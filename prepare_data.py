import os
import time
import copy
import pandas as pd
from tqdm import tqdm
import csv
import re

from sklearn.metrics import f1_score

import torch
import torch.nn as nn

from src.utils import load_CIFAR, load_MNIST, load_ImageNet, parse_args, get_model, get_trainable_modules_main, extract_class_to_dataloder, patch_resnet_imagenet, eval_model_dataloder, _configure_logging
from src.attribution import get_relevance_scores, get_relevance_scores_for_all_layers
from src.pruning_methods import prune_neurons, prune_layers

from models_info.models_cv.resnet_imagenet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


### Example usage (ResNet - ImageNet pair):
# python3 prepare_data.py --data-path /data/shenghao/dataset --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_IMAGENET_whole.pth' --dataset imagenet --batch-size 4 --end2end --model resnet18 --top-m-neurons 6 --csv-file resnet18_imagenet_b4 --device 'cuda:0'
# python3 prepare_data.py --data-path /data/shenghao/dataset --saved-model '/torch-deepimportance/models_info/saved_models/resnet152_IMAGENET_whole.pth' --dataset imagenet --batch-size 4 --end2end --model resnet152 --top-m-neurons 6 --csv-file resnet152_imagenet_b4 --device 'cuda:0'
### Example usage (VGG16 - CIFAR10 pair):
# python3 prepare_data.py --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --batch-size 2 --end2end --model vgg16 --top-m-neurons 6 --csv-file vgg16_cifar_t6 --inordered-dataset --device 'cuda:0'
# python3 prepare_data.py --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --batch-size 2 --end2end --model vgg16 --top-m-neurons 6 --csv-file vgg16_testonly --inordered-dataset --device 'cuda:0'


SKIPTEST = True

def save_to_csv(train_images, train_labels, layer_info, optimal_method, csv_file):
    # Flatten the train_images and convert to list
    flattened_images = train_images.squeeze(0).flatten().tolist()
    
    # Convert train_labels and layer_info to list    
    train_label = train_labels.item()
    layer_info_list = layer_info.tolist()
    
    # Create a dictionary to hold the data
    data = {
        'image': [flattened_images],
        'label': [train_label],
        'layer_info': [layer_info_list],
        'optimal_method': [optimal_method]
    }
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

def save_layer_scores_to_csv(layer_scores, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(['LayerName', 'NeuronIndex', 'Score'])
        
        # Write layer scores
        for layer_name, scores in layer_scores.items():
            for neuron_index, score in enumerate(scores):
                writer.writerow([layer_name, neuron_index, score])

def save_inter_layer_scores_to_csv(layer_scores, labels, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(['LayerName', 'NeuronIndex', 'Score', 'Label'])
        
        # Write layer scores
        for layer_name, scores in layer_scores.items():
            for neuron_index, score in enumerate(scores):
                writer.writerow([layer_name, neuron_index, score, labels])

def prapared_parameters(args):
    # Logger settings
    logger = _configure_logging(args.logging, args, 'debug')
    
    # Model settings
    model_path = os.getenv("HOME") + args.saved_model
    if args.dataset == 'imagenet' and re.match(r'resnet\d+', args.model):
        model_mapping = {
            'resnet18': ResNet18,
            'resnet34': ResNet34,
            'resnet50': ResNet50,
            'resnet101': ResNet101,
            'resnet152': ResNet152
        }
        model_fit = model_mapping.get(args.model, ResNet18)(num_classes=1000)  # Default to ResNet18 if not found
        model_fit = model_fit.eval()
        model = patch_resnet_imagenet(model_fit, model_path, model_name=args.model)
        module_name = []
        module = []
        for name, layer in model.named_modules():
            module_name.append(name)
            module.append(layer)    
    else:
        # Model loading
        model, module_name, module = get_model(model_path)
    
    trainable_module, trainable_module_name = get_trainable_modules_main(model)

    return model, module_name, module, trainable_module, trainable_module_name, logger

def assign_ranking_scores(important_neurons_dict):
    ranking_scores = {}

    # Iterate through each attribution method in the order of the dictionary
    for method, neurons in important_neurons_dict.items():
        # Assign scores based on the position in the list (higher position = higher score)
        for rank, neuron in enumerate(neurons):
            score = len(neurons) - rank  # Higher rank gets higher score
            if neuron not in ranking_scores:
                ranking_scores[neuron] = 0
            ranking_scores[neuron] += score  # Sum scores if neuron appears in multiple methods

    return ranking_scores


def rank_and_select_top_neurons(important_neurons_dict, top_n):
    ranking_scores = assign_ranking_scores(important_neurons_dict)

    # Sort neurons by their cumulative scores in descending order
    ranked_neurons = sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)
    return [neuron for neuron, score in ranked_neurons[:top_n]]

def weighted_top_neurons(important_neurons_dict, loss_gains, top_k=10):
    """
    Compute weighted scores for neurons based on loss_gains and return the top K neurons.

    Args:
        important_neurons_dict (dict): Dictionary containing neurons and their scores for each attribution method.
        loss_gains (dict): Dictionary containing loss gains for each attribution method.
        top_k (int): Number of top neurons to return.

    Returns:
        list: Top K neurons with their weighted scores.
    """
    # Normalize loss_gains to compute weights
    total_loss_gain = sum(loss_gains.values())
    weights = {method: gain / total_loss_gain for method, gain in loss_gains.items()}

    # Compute weighted scores for neurons
    weighted_scores = {}
    for method, neurons in important_neurons_dict.items():
        weight = weights.get(method, 0)
        for layer_name, score, index in neurons:
            neuron_key = (layer_name, index)
            if neuron_key not in weighted_scores:
                weighted_scores[neuron_key] = 0
            weighted_scores[neuron_key] += score * weight

    # Sort neurons by their weighted scores in descending order
    sorted_neurons = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)

    # Return the top K neurons
    return sorted_neurons[:top_k]

# Step - 1
def get_layer_info(layer_name, trainable_module_name):
    layer_info = torch.zeros(len(trainable_module_name))
    layer_info[trainable_module_name.index(layer_name)] = 1
    return layer_info

def prepare_data(args):
    ### Dataset settings
    if args.dataset == 'cifar10':
        trainloader, testloader, train_dataset, test_dataset, classes = load_CIFAR(batch_size=args.batch_size, root=args.data_path, shuffle=True)
    elif args.dataset == 'mnist':
        trainloader, testloader, train_dataset, test_dataset, classes = load_MNIST(batch_size=args.batch_size, root=args.data_path)
    elif args.dataset == 'imagenet':
        trainloader, testloader, train_dataset, test_dataset, classes = load_ImageNet(batch_size=args.batch_size,  root=args.data_path+'/ImageNet',  num_workers=2, use_val=False)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    return trainloader, testloader, train_dataset, test_dataset, classes

def test_model(model, device, inputs, labels):
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    inputs, labels = inputs.to(device), labels.to(device)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    total_loss += loss.item() * inputs.size(0)
    _, predicted = torch.max(outputs, 1)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)
    accuracy = correct / total * 100
    
    f1_macro = f1_score(predicted.cpu().numpy(), labels.cpu().numpy(), average='macro')
    
    return accuracy, total_loss, f1_macro

def select_top_neurons(importance_scores, top_m_neurons=5):
    if top_m_neurons == -1:
        print("Selecting all neurons.")
        if importance_scores.dim() == 1:
            _, indices = torch.sort(importance_scores, descending=True)
            return indices
        else:
            return None
    else:
        if importance_scores.dim() == 1:
            print("Selecting top {} neurons (FC layer).".format(top_m_neurons))
            _, indices = torch.topk(importance_scores, top_m_neurons)
            return indices
        else:
            mean_attribution = torch.mean(importance_scores, dim=[1, 2])
            if mean_attribution.shape[0] < top_m_neurons:
                print("Selecting all the neurons (Conv2D layer).")
                return None
            else:
                _, indices = torch.topk(mean_attribution, top_m_neurons)
                return indices

def select_top_neurons_all(importance_scores_dict, top_m_neurons=5, filter_neuron=None):
    flattened_importance = []

    # Flatten and collect importance scores across all layers
    for layer_name, importance_scores in importance_scores_dict.items():
        # Fully connected layer
        if importance_scores.dim() == 1:  
            for idx, score in enumerate(importance_scores):
                flattened_importance.append((layer_name, score.item(), idx))
        # Convolutional layer
        else:  
            # Compute mean importance per channel
            mean_attribution = torch.mean(importance_scores, dim=[1, 2])
            for idx, score in enumerate(mean_attribution):
                flattened_importance.append((layer_name, score.item(), idx))
                
    # Filter out the last layer
    if filter_neuron is not None:
        # Filter out the specific layer (e.g., 'fc3')
        filtered_importance = [item for item in flattened_importance if item[0] != filter_neuron]
    else:
        filtered_importance = flattened_importance
        
    # Sort by importance score in descending order
    flattened_importance = sorted(filtered_importance, key=lambda x: x[1], reverse=True)
        
    # Select top-m neurons across all layers
    if top_m_neurons == -1:
        selected = flattened_importance
    else:
        selected = flattened_importance[:top_m_neurons]
                
    # Group selected neurons by layer
    selected_indices = {}
    for layer_name, _, index in selected:
        if layer_name not in selected_indices:
            selected_indices[layer_name] = []
        selected_indices[layer_name].append(index)
        
    # Convert lists to tensors
    for layer_name in selected_indices:
        selected_indices[layer_name] = torch.tensor(selected_indices[layer_name])
        
    # Neurons Index without and with scores
    return selected_indices, selected

# Step -2
def extract_features(model, inputs):
    features = []
    def hook_fn(module, input, output):
        features.append(output.detach())
    handle = model.fc2.register_forward_hook(hook_fn)
    model.eval()

    with torch.no_grad():
        model(inputs)

    # Remove the hook
    handle.remove()
    return features

# Step - 3
def get_important_dict(attributions, model, device, train_inputs, train_labels, classes, net_layer, layer_name, top_m_neurons, final_layer, end2end=False):
    important_neurons_dict = {}
    for attribution_method in attributions:
        if end2end:
            # print("Relevance scores for all layers.")
            layer_importance_scores = get_relevance_scores_for_all_layers(model, train_inputs, train_labels, device, attribution_method=attribution_method)
            important_neurons, inorderd_neuron_indices = select_top_neurons_all(layer_importance_scores, top_m_neurons, final_layer)
            important_neurons_dict[attribution_method] = inorderd_neuron_indices
        else:
            # print("Relevance scores for layer: {}".format(layer_name))
            _, layer_importance_scores = get_relevance_scores(model, 
                                                            train_inputs, 
                                                            train_labels, 
                                                            classes, 
                                                            net_layer,
                                                            layer_name=layer_name, 
                                                            attribution_method=attribution_method)

    
            important_neurons  = select_top_neurons(layer_importance_scores, top_m_neurons)
            important_neurons_dict[attribution_method] = important_neurons
    
    if end2end:
        top_neurons_dict = rank_and_select_top_neurons(important_neurons_dict, top_m_neurons)
        
    return important_neurons_dict

# Step - 4
def identify_optimal_method(model, device, original_state, classes, inputs, labels, important_neurons_dict, layer_index, logger, end2end):
    
    pruned_model = copy.deepcopy(model)
    trainable_module_pruned, trainable_module_name_pruned = get_trainable_modules_main(pruned_model)
    layer_name = trainable_module_name_pruned[layer_index]
    net_layer = trainable_module_pruned[layer_index]
    # print("prunned layer name: ", layer_name)
    
    original_accuracy, original_loss, f1_score = test_model(model, device, inputs, labels)
    accuracy_drops = {}
    loss_gains = {}
        
    for method, neurons_to_prune in important_neurons_dict.items():
        if end2end:
            prune_layers(pruned_model, neurons_to_prune)
        else:
            prune_neurons(pruned_model, net_layer, neurons_to_prune)
        
        # Test the pruned model
        pruned_accuracy, pruned_loss, f1_score = test_model(pruned_model, device, inputs, labels)
        accuracy_drop = original_accuracy - pruned_accuracy
        accuracy_drops[method] = accuracy_drop
        loss_gains[method] = pruned_loss - original_loss
        
        pruned_model.load_state_dict(original_state)
        # print("Model restored to original state.")
    
    # Find the method with the [MAX(accuracy drop)] or [MAX(loss gain)]
    # optimal_method = max(accuracy_drops, key=accuracy_drops.get)
    sorted_neurons = weighted_top_neurons(important_neurons_dict, loss_gains)
    optimal_method = max(loss_gains, key=loss_gains.get)
    sorted_neurons_ = important_neurons_dict[optimal_method]
    sorted_neurons_opti = [((layer_name, index), score) for layer_name, score, index in sorted_neurons_]
    
    logger.info(f"Label: {classes[labels[0]]}, Optimal method: {optimal_method}, Accuracy drop: {accuracy_drops[optimal_method]:.4f}, Loss gain: {loss_gains[optimal_method]:.4f}")

    return optimal_method, accuracy_drops, sorted_neurons, sorted_neurons_opti

def save_intersection(important_neurons_dict, logger):
    sets = {k: set(v.tolist()) for k, v in important_neurons_dict.items()}
    intersection = list(set.intersection(*sets.values()))
    logger.info("Common Neurons: {}".format(intersection))


def voting_init(layer_scores, trainable_module_name, trainable_module, excluded_layer=None):    
    for layer_name, module in zip(trainable_module_name, trainable_module):
        if layer_name == excluded_layer:
            continue
        
        # Initialize scores list for each layer
        if isinstance(module, torch.nn.Linear):
            layer_scores[layer_name] = [0] * module.out_features
        elif isinstance(module, torch.nn.Conv2d):
            layer_scores[layer_name] = [0] * module.out_channels
            
    return layer_scores

def voting_neurons(layer_index_pairs, layer_scores):
    # Assign scores in reverse order (higher order = higher score)
    for rank, (layer_name, neuron_index) in enumerate(reversed(layer_index_pairs), start=1):
        if layer_name in layer_scores:
            layer_scores[layer_name][neuron_index] += rank

    return layer_scores

def create_train_data(attributions, 
               model,
               device,
               classes, 
               net_layer, 
               layer_name, 
               top_m_neurons, 
               original_state, 
               layer_info,
               layer_index,
               trainloader,
               logger,
               end2end,
               final_layer,
               across_attr_method,
               csv_file,
               get_intersection=False):
        
    layer_scores = {}
    init_count = 0
    
    for train_images, train_labels in tqdm(trainloader):
                
        ### Obtain important neurons using different attribution methods
        important_neurons_dict = get_important_dict(attributions, 
                                                    model,
                                                    device,
                                                    train_images, 
                                                    train_labels, 
                                                    classes, 
                                                    net_layer, 
                                                    layer_name, 
                                                    top_m_neurons,
                                                    final_layer,
                                                    end2end)
        
        if get_intersection:
            save_intersection(important_neurons_dict, logger)
            
        optimal_method, accuracy_drops, sorted_neurons, sorted_neurons_opti = identify_optimal_method(model,
                                                                device,
                                                                original_state,
                                                                classes, 
                                                                train_images, 
                                                                train_labels, 
                                                                important_neurons_dict, 
                                                                layer_index,
                                                                logger,
                                                                end2end)
        layer_index_pairs = [neuron[0] for neuron in sorted_neurons]
        layer_index_pairs_opti = [neuron[0] for neuron in sorted_neurons_opti]
        
        if end2end:
            if init_count == 0:
                voting_init(layer_scores, trainable_module_name, trainable_module, trainable_module_name[-1])
                init_count += 1
                if across_attr_method:
                    layer_scores = voting_neurons(layer_index_pairs, layer_scores)
                else:
                    layer_scores = voting_neurons(layer_index_pairs_opti, layer_scores)
            else:
                init_count += 1
                if across_attr_method:
                    layer_scores = voting_neurons(layer_index_pairs, layer_scores)
                else:
                    layer_scores = voting_neurons(layer_index_pairs_opti, layer_scores)
            
        else:
            # Save to [train_images, train_labels, layer_info, optimal_method]
            if train_images.size(0) == 1:
                save_to_csv(train_images, train_labels, layer_info, optimal_method, "prepared_data_train_cifar_{}.csv".format(model.__module__[10:]))
            else:
                for i in range(train_images.size(0)):
                    save_to_csv(train_images[i].unsqueeze(0), train_labels[i].unsqueeze(0), layer_info, optimal_method, "prepared_data_train_cifar_{}.csv".format(model.__module__[10:]))
        
        b_accuracy, b_total_loss, b_f1_score = test_model(model, device, train_images, train_labels)
        
        logger.info("Optimal method: {}".format(optimal_method))
        logger.info("Accuracy drop: {}".format(accuracy_drops))
        logger.info("Before Acc: {:.2f}%, Before Loss: {:.2f}, Before F1 Score: {:.2f}".format(b_accuracy, b_total_loss, b_f1_score))
    
    if end2end:
        save_layer_scores_to_csv(layer_scores, "./saved_files/pre_csv/{}.csv".format(csv_file))    
        print("Layer scores saved to CSV.")
    
    logger.info("Trainloader DONE.") 

if __name__ == '__main__':
    args = parse_args()
    model, module_name, module, trainable_module, trainable_module_name, logger = prapared_parameters(args)
        
    ### Data settings
    trainloader, testloader, train_dataset, test_dataset, classes = prepare_data(args)    
    
    ### Device settings
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")    
    
    # attributions = ['lc', 'la', 'ii', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp']
    attributions = ['lc', 'la', 'ii', 'ldl', 'ldls', 'lgs', 'lig', 'lrp']
        
    ## Saved the index for comparing the Common Neurons across the attributions
    # {'lc': [1, 2, 3, 4, 5], 'la': [1, 2, 3, 4, 5], ...}
    attr_dict = {key: [] for key in attributions}
    original_state = copy.deepcopy(model.state_dict())
    layer_info = get_layer_info(trainable_module_name[args.layer_index], trainable_module_name)
    
    if args.inordered_dataset:
        trainloader = extract_class_to_dataloder(train_dataset, classes, args.batch_size)
        testloader = extract_class_to_dataloder(test_dataset, classes, args.batch_size)
    
    # Skip final classifier layer
    final_layer = trainable_module_name[-1]
    
    if not SKIPTEST:
        acc, loss, f1 = eval_model_dataloder(model, testloader, device=device)
        logger.info(f'Original Model testing Accuracy: {acc}, Loss: {loss}, F1: {f1}')
        
    # set across_attr_method = True to get the top-K neurons across all the attribution methods
    create_train_data(attributions=attributions, 
                   model=model,
                   device=device,
                   classes=classes, 
                   net_layer=trainable_module[args.layer_index], 
                   layer_name=trainable_module_name[args.layer_index], 
                   top_m_neurons=args.top_m_neurons, 
                   original_state=original_state, 
                   layer_info=layer_info,
                   layer_index=args.layer_index,
                   trainloader=trainloader,
                   logger=logger,
                   final_layer=final_layer,
                   end2end=args.end2end,
                   across_attr_method=True,
                   csv_file=args.csv_file,
                   get_intersection=False)
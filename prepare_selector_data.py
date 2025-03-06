import os
import time
import copy
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add src directory to sys.path
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

from utils import load_CIFAR, load_MNIST, load_ImageNet, parse_args, get_model, get_trainable_modules_main, extract_class_to_dataloder, Logger
from attribution import get_relevance_scores
from pruning_methods import prune_neurons

class CustomDataset(Dataset):
    def __init__(self, csv_file, attributions):
        self.data = pd.read_csv(csv_file)
        self.attributions = attributions
        self.method_to_idx = {method: idx for idx, method in enumerate(attributions)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.tensor(eval(self.data.iloc[idx]['image']), dtype=torch.float32).view(3, 32, 32)
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)
        layer_info = torch.tensor(eval(self.data.iloc[idx]['layer_info']), dtype=torch.float32)
        optimal_method = self.data.iloc[idx]['optimal_method']
        optimal_method_idx = self.method_to_idx[optimal_method]
        optimal_method_one_hot = torch.zeros(len(self.attributions))
        optimal_method_one_hot[optimal_method_idx] = 1
        
        return image, label, layer_info, optimal_method_one_hot

def test_new_dataset_loading(csv_file='prepared_data.csv', attributions=['lfa', 'ldl']):
    
    dataset = CustomDataset(csv_file, attributions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # Training loop
    for epoch in range(3):
        for images, labels, layer_infos, optimal_methods in dataloader:
            # Your training code here
            print(images.shape, labels.shape, layer_infos.shape, optimal_methods.shape)
            break

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

def prapared_parameters(args):
    ### Logger settings
    if args.logging:
        start_time = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
        saved_log_name = args.log_path + '-{}-{}-{}-{}.log'.format(args.model, args.dataset, args.layer_index, timestamp)
        # saved_log_name = args.log_path + 'PrepareData-{}-{}-L{}-{}.log'.format(args.model, args.dataset, args.layer_index, timestamp)
        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[=== Model: {}, Dataset: {}, Layers_Index: {}, TopK: {} ==]".format(args.model, args.dataset, args.layer_index, args.top_m_neurons))
    else:
        log = None

    ### Model settings
    model_path = os.getenv("HOME") + args.saved_model
    
    ### Model loading
    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)

    return model, module_name, module, trainable_module, trainable_module_name, log

# Step - 1
def get_layer_info(layer_name, trainable_module_name):
    layer_info = torch.zeros(len(trainable_module_name))
    layer_info[trainable_module_name.index(layer_name)] = 1
    return layer_info

def prepare_data(args):
    ### Dataset settings
    if args.dataset == 'cifar10':
        trainloader, testloader, train_dataset, test_dataset, classes = load_CIFAR(batch_size=args.batch_size, root=args.data_path, large_image=args.large_image, shuffle=True)
    elif args.dataset == 'mnist':
        trainloader, testloader, train_dataset, test_dataset, classes = load_MNIST(batch_size=args.batch_size, root=args.data_path)
    elif args.dataset == 'imagenet':
        # batch_size=32, root='/data/shenghao/dataset/ImageNet', num_workers=2, use_val=False
        trainloader, testloader, train_dataset, test_dataset, classes = load_ImageNet(batch_size=args.batch_size, 
                                                         root=args.data_path + '/ImageNet', 
                                                         num_workers=2, 
                                                         use_val=False)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    return trainloader, testloader, train_dataset, test_dataset, classes

# Evaluate the model on the given dataloader and compute accuracy, loss, and F1 score.
def test_model_dataloder(model, dataloader, device='cpu'):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)

            # Store labels and predictions for metric computation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute average loss
    avg_loss = running_loss / len(dataloader.dataset)

    # Compute accuracy
    correct_predictions = sum(p == t for p, t in zip(all_preds, all_labels))
    accuracy = correct_predictions / len(all_labels)

    # Compute F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, avg_loss, f1

def test_model(model, inputs, labels):
    criterion = nn.CrossEntropyLoss()
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
    
    f1_macro = f1_score(predicted, labels, average='macro')
    
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
def get_important_dict(attributions, model, train_inputs, train_labels, classes, net_layer, layer_name, top_m_neurons):
    important_neurons_dict = {}
    for attribution_method in attributions:
        _, layer_importance_scores = get_relevance_scores(model, 
                                                        train_inputs, 
                                                        train_labels, 
                                                        classes, 
                                                        net_layer,
                                                        layer_name=layer_name, 
                                                        attribution_method=attribution_method)
    
        important_neurons  = select_top_neurons(layer_importance_scores, top_m_neurons)
        important_neurons_dict[attribution_method] = important_neurons
    return important_neurons_dict

# Step - 4
def identify_optimal_method(model, original_state, classes, inputs, labels, important_neurons_dict, layer_index, log):
    
    pruned_model = copy.deepcopy(model)
    trainable_module_pruned, trainable_module_name_pruned = get_trainable_modules_main(pruned_model)
    layer_name = trainable_module_name_pruned[layer_index]
    net_layer = trainable_module_pruned[layer_index]
    print("prunned layer name: ", layer_name)
    
    original_accuracy, original_loss, f1_score = test_model(model, inputs, labels)
    accuracy_drops = {}
    loss_gains = {}
    
    for method, neurons_to_prune in important_neurons_dict.items():        
        prune_neurons(pruned_model, net_layer, neurons_to_prune)
        # Test the pruned model
        pruned_accuracy, pruned_loss, f1_score = test_model(pruned_model, inputs, labels)
        accuracy_drop = original_accuracy - pruned_accuracy
        accuracy_drops[method] = accuracy_drop
        loss_gains[method] = pruned_loss - original_loss
        pruned_model.load_state_dict(original_state)
        print("Model restored to original state.")

    # Find the method with the MAX accuracy drop
    # optimal_method = max(accuracy_drops, key=accuracy_drops.get)
    optimal_method = max(loss_gains, key=loss_gains.get)
    if log is not None:
        log.logger.info(f"Label: {classes[labels[0]]}, Optimal method: {optimal_method}, Accuracy drop: {accuracy_drops[optimal_method]:.4f}, Loss gain: {loss_gains[optimal_method]:.4f}")
    else:
        print(f"Label: {classes[labels[0]]}, Optimal method: {optimal_method}, Accuracy drop: {accuracy_drops[optimal_method]:.4f}, Loss gain: {loss_gains[optimal_method]:.4f}")
    
    return optimal_method, accuracy_drops

def create_data(attributions, 
               model,
               classes, 
               net_layer, 
               layer_name, 
               top_m_neurons, 
               original_state, 
               layer_info,
               layer_index,
               trainloader,
               testloader,
               log):
    
    model.eval()
    for train_images, train_labels in tqdm(trainloader):
        ### Obtain important neurons using different attribution methods
        important_neurons_dict = get_important_dict(attributions, 
                                                    model,
                                                    train_images, 
                                                    train_labels, 
                                                    classes, 
                                                    net_layer, 
                                                    layer_name, 
                                                    top_m_neurons)
        
        sets = {k: set(v.tolist()) for k, v in important_neurons_dict.items()}
        intersection = list(set.intersection(*sets.values()))
        if log is not None:
            log.logger.info("Common Neurons: {}".format(intersection))
        else:
            print("Common Neurons: ", intersection)
            
        optimal_method, accuracy_drops = identify_optimal_method(model, 
                                                                original_state,
                                                                classes, 
                                                                train_images, 
                                                                train_labels, 
                                                                important_neurons_dict, 
                                                                layer_index,
                                                                log)
        
        # Save to [train_images, train_labels, layer_info, optimal_method]
        if train_images.size(0) == 1:
            save_to_csv(train_images, train_labels, layer_info, optimal_method, "prepared_data_train_cifar_{}.csv".format(model.__module__[10:]))
        else:
            for i in range(train_images.size(0)):
                save_to_csv(train_images[i].unsqueeze(0), train_labels[i].unsqueeze(0), layer_info, optimal_method, "prepared_data_train_cifar_{}.csv".format(model.__module__[10:]))
        
        b_accuracy, b_total_loss, f1_score = test_model(model, train_images, train_labels)
        
        if log is not None:
            log.logger.info("Optimal method: {}".format(optimal_method))
            log.logger.info("Accuracy drop: {}".format(accuracy_drops))
            log.logger.info("Before Acc: {:.2f}%, Before Loss: {:.2f}".format(b_accuracy, b_total_loss))
        else:
            print("Optimal method: {}".format(optimal_method))
            print("Accuracy drop: {}".format(accuracy_drops))
            print("Before Acc: {:.2f}%, Before Loss: {:.2f}".format(b_accuracy, b_total_loss))
    
    print("Trainloader Done.")
    
    for test_images, test_labels in testloader:
        ### Obtain important neurons using different attribution methods
        important_neurons_dict = get_important_dict(attributions, 
                                                    model,
                                                    test_images, 
                                                    test_labels, 
                                                    classes, 
                                                    net_layer, 
                                                    layer_name, 
                                                    top_m_neurons)
        
        optimal_method, accuracy_drops = identify_optimal_method(model, 
                                                                original_state,
                                                                classes,
                                                                test_images, 
                                                                test_labels, 
                                                                important_neurons_dict, 
                                                                layer_index,
                                                                log)
        if test_images.size(0) == 1:
            save_to_csv(test_images, test_labels, layer_info, optimal_method, "prepared_data_train_cifar_{}.csv".format(model.__module__[10:]))
        else:
            for i in range(test_images.size(0)):
                save_to_csv(test_images[i].unsqueeze(0), test_labels[i].unsqueeze(0), layer_info, optimal_method, "prepared_data_train_cifar_{}.csv".format(model.__module__[10:]))
        
        b_accuracy, b_total_loss, f1_score = test_model(model, test_images, test_labels)
        
        if log is not None:
            log.logger.info("Optimal method: {}".format(optimal_method))
            log.logger.info("Accuracy drop: {}".format(accuracy_drops))
            log.logger.info("Before Acc: {:.2f}%, Before Loss: {:.2f}".format(b_accuracy, b_total_loss))
        else:
            print("Optimal method: {}".format(optimal_method))
            print("Accuracy drop: {}".format(accuracy_drops))
            print("Before Acc: {:.2f}%, Before Loss: {:.2f}".format(b_accuracy, b_total_loss))
    
    print("Testloader Done.")
    

def train_data_layer_check(attributions, 
               model,
               classes, 
               net_layer, 
               layer_name, 
               top_m_neurons, 
               original_state, 
               trainloader,
               log):
    
    model.eval()
    
    tested_label = -1
    
    for train_images, train_labels in tqdm(trainloader):
        
        if tested_label == train_labels[0].item():
            continue
        else:
            tested_label = train_labels[0].item()
        
        for index, layer in enumerate(net_layer[1:-1]):
            ### Obtain important neurons using different attribution methods
            important_neurons_dict = get_important_dict(attributions, 
                                                        model,
                                                        train_images, 
                                                        train_labels, 
                                                        classes, 
                                                        layer, 
                                                        layer_name[index + 1], 
                                                        top_m_neurons)
        
            sets = {k: set(v.tolist()) for k, v in important_neurons_dict.items()}
            intersection = list(set.intersection(*sets.values()))
        
            if log is not None:
                log.logger.info("Layer: {}, Common Neurons: {}".format(layer_name[index + 1], intersection))
            else:
                print("Layer: {}, Common Neurons: {}".format(layer_name[index + 1], intersection))
            
            optimal_method, accuracy_drops = identify_optimal_method(model, 
                                                                original_state,
                                                                classes, 
                                                                train_images, 
                                                                train_labels, 
                                                                important_neurons_dict, 
                                                                index + 1,
                                                                log)
        
        
            b_accuracy, b_total_loss, f1_score = test_model(model, train_images, train_labels)
            
            if log is not None:
                log.logger.info("Optimal method: {}".format(optimal_method))
                log.logger.info("Accuracy drop: {}".format(accuracy_drops))
                log.logger.info("Before Acc: {:.2f}%, Before Loss: {:.2f}".format(b_accuracy, b_total_loss))
            else:
                print("Optimal method: {}".format(optimal_method))
                print("Accuracy drop: {}".format(accuracy_drops))
                print("Before Acc: {:.2f}%, Before Loss: {:.2f}".format(b_accuracy, b_total_loss))
            
    print("Trainloader Done.")    

if __name__ == '__main__':
    args = parse_args()
    args.batch_size = 1
    model, module_name, module, trainable_module, trainable_module_name, log = prapared_parameters(args)
        
    ### Data settings
    trainloader, testloader, train_dataset, test_dataset, classes = prepare_data(args)    
    
    # attributions = ['lc', 'la', 'ii', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp']
    # attributions = ['lc', 'la', 'ii', 'ldl', 'lgs', 'lig', 'lfa', 'lrp']
    attributions = ['lc', 'la', 'ii', 'ldl', 'ldls', 'lgs', 'lig', 'lrp']
    # attributions = ['la', 'ii']
        
    ## Saved the index for comparing the Common Neurons across the attributions
    # {'lc': [1, 2, 3, 4, 5], 'la': [1, 2, 3, 4, 5], ...}
    attr_dict = {key: [] for key in attributions}
    original_state = copy.deepcopy(model.state_dict())
    layer_info = get_layer_info(trainable_module_name[args.layer_index], trainable_module_name)
    
    ordered_loader_train = extract_class_to_dataloder(train_dataset, classes, 100)
    ordered_loader_test = extract_class_to_dataloder(test_dataset, classes, 100)
    
    train_data_layer_check(attributions=attributions, 
               model=model,
               classes=classes, 
               net_layer=trainable_module, 
               layer_name=trainable_module_name, 
               top_m_neurons=args.top_m_neurons, 
               original_state=original_state, 
               trainloader=ordered_loader_train,
               log=log)
    
    # create_data(attributions=attributions, 
    #            model=model,
    #            classes=classes, 
    #            net_layer=trainable_module[args.layer_index], 
    #            layer_name=trainable_module_name[args.layer_index], 
    #            top_m_neurons=args.top_m_neurons, 
    #            original_state=original_state, 
    #            layer_info=layer_info,
    #            layer_index=args.layer_index,
    #            trainloader=ordered_loader_train,
    #            testloader=ordered_loader_test,
    #            log=log)
    
    # test_new_dataset_loading("prepared_data_train_cifar_{}.csv".format(model.__module__[10:]), attributions)
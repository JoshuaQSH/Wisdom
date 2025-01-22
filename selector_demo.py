import os
import time
import copy
from pathlib import Path
import sys

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim

# Add src directory to sys.path
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

from utils import load_CIFAR, load_MNIST, load_ImageNet, get_class_data, parse_args, get_model, get_model_cifar, get_trainable_modules_main, SelectorDataset, Logger
from attribution import get_relevance_scores, get_relevance_scores_for_all_layers
from pruning_methods import prune_neurons

def save_model(model, model_name):
    torch.save(model.state_dict(), model_name + '.pt')
    print("Model saved as", model_name + '.pt')

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
    
    return accuracy, total_loss


def prapared_parameters(args):
    ### Logger settings
    if args.logging:
        start_time = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
        saved_log_name = args.log_path + 'Selector-{}-{}-L{}-{}.log'.format(args.model, args.dataset, args.layer_index, timestamp)
        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[=== Model: {}, Dataset: {}, Layers_Index: {}, TopK: {} ==]".format(args.model, args.dataset, args.layer_index, args.top_m_neurons))
    else:
        log = None
    
    ### Model settings
    if args.model_path != 'None':
        model_path = args.model_path
    else:
        model_path = os.getenv("HOME") + '/torch-deepimportance/models_info/saved_models/'
    model_path += args.saved_model
    
    ## Loading models - either 1) from scratch or 2) pretrained
    if args.dataset == 'cifar10' and args.model != 'lenet':
        model, module_name, module = get_model_cifar(model_name=args.model, load_model_path=model_path)
    else:
        # We aussume that the SOTA models are pretrained with IMAGENET
        model, module_name, module = get_model(model_name=args.model)
    
    # TODO: A Hack here for model loading
    model.load_state_dict(torch.load(model_path))
    selector_model = copy.deepcopy(model)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)
    
    ### Device settings    
    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    return model, selector_model, module_name, module, trainable_module, trainable_module_name, device, log

def prepare_data(args):
    ### Dataset settings
    if args.dataset == 'cifar10':
        trainloader, testloader, test_dataset, classes = load_CIFAR(batch_size=args.batch_size, root=args.data_path, large_image=args.large_image, shuffle=True)
    elif args.dataset == 'mnist':
        trainloader, testloader, test_dataset, classes = load_MNIST(batch_size=args.batch_size, root=args.data_path)
    elif args.dataset == 'imagenet':
        # batch_size=32, root='/data/shenghao/dataset/ImageNet', num_workers=2, use_val=False
        trainloader, testloader, test_dataset, classes = load_ImageNet(batch_size=args.batch_size, 
                                                         root=args.data_path + '/ImageNet', 
                                                         num_workers=2, 
                                                         use_val=False)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    return trainloader, testloader, classes

# Step 1: Extract layer information
def get_layer_info(layer_name, trainable_module_name):
    layer_info = torch.zeros(len(trainable_module_name))
    layer_info[trainable_module_name.index(layer_name)] = 1
    return layer_info

# Step 2: Extract features using transfer learning (freeze feature extractor)
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

# Step 3: Obtain important neurons using different attribution methods
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

# Step 4: Identify the optimal attribution method by evaluating accuracy drop
def identify_optimal_method(model, original_state, inputs, labels, important_neurons_dict, layer_name):
    
    pruned_model = copy.deepcopy(model)
    original_accuracy, _ = test_model(model, inputs, labels)
    accuracy_drops = {}
    
    for method, neurons_to_prune in important_neurons_dict.items():        
        prune_neurons(pruned_model, layer_name, neurons_to_prune)
        # Test the pruned model
        pruned_accuracy, _ = test_model(pruned_model, inputs, labels)
        accuracy_drop = original_accuracy - pruned_accuracy
        accuracy_drops[method] = accuracy_drop

        pruned_model.load_state_dict(original_state)
        print("Model restored to original state.")

    # Find the method with the MAX accuracy drop
    optimal_method = max(accuracy_drops, key=accuracy_drops.get)
    print(f"Optimal method: {optimal_method}, Accuracy drop: {accuracy_drops[optimal_method]:.4f}")
    
    return optimal_method, accuracy_drops

def string_to_one_hot(attribution_methods, string_labels):
    method_to_idx = {method: idx for idx, method in enumerate(attribution_methods)}
    integer_label = method_to_idx[string_labels]
    return torch.tensor(integer_label, dtype=torch.long)

# attributions, model, classes, net_layer, layer_name, top_m_neurons
def train_demo(attributions, 
               model,
               selector_model,
               num_epochs,
               classes, 
               net_layer, 
               layer_name, 
               top_m_neurons, 
               original_state, 
               layer_info,
               train_images,
               train_labels,
               log):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(selector_model.parameters(), lr=0.001)
    # optimizer = optim.SGD(selector_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    selector_model.train()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selector_model.to(device)
    
    ### Obtain important neurons using different attribution methods
    important_neurons_dict = get_important_dict(attributions, 
                                                model,
                                                train_images, 
                                                train_labels, 
                                                classes, 
                                                net_layer, 
                                                layer_name, 
                                                top_m_neurons)
    
    optimal_method, accuracy_drops = identify_optimal_method(model, 
                                                            original_state, 
                                                            train_images, 
                                                            train_labels, 
                                                            important_neurons_dict, 
                                                            layer_name)
    
    attr_label = string_to_one_hot(attributions, optimal_method)
    attr_label_ex = attr_label.repeat(train_images.shape[0], 1).squeeze(1)
    attr_label_ex = attr_label_ex.to(device)
    
    test_model(model, train_images, train_labels)
    b_accuracy, b_total_loss = test_model(model, train_images, train_labels)
    if log is not None:
        log.logger.info("Optimal method: {}".format(optimal_method))
        log.logger.info("Accuracy drop: {}".format(accuracy_drops))
        log.logger.info("Before Acc: {:.2f}%, Before Loss: {:.2f}".format(b_accuracy, b_total_loss))
    else:
        print("Optimal method: {}".format(optimal_method))
        print("Accuracy drop: {}".format(accuracy_drops))
        print("Before Acc: {:.2f}%, Before Loss: {:.2f}".format(b_accuracy, b_total_loss))
    
    ### Extract features using transfer learning (freeze feature extractor)
    features = extract_features(model, train_images)
    layer_info_repeated = layer_info.unsqueeze(0).repeat(features[0].shape[0], 1)
    combined_features = torch.cat((features[0], layer_info_repeated), dim=1)
        
    for epoch in range(num_epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        ## Train the selector model
        # outputs = selector_model(train_images.to(device))
        
        ##  Train the final fc layer only
        outputs = selector_model(combined_features.to(device))
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, attr_label_ex)
        loss.backward()
        optimizer.step()
            
        # running_loss = loss.item() * train_images.size(0)
        running_loss = loss.item()
        running_corrects = torch.sum(preds == attr_label_ex.data)
                
        if log is not None:
            log.logger.info("Epoch [{}/{}], Loss: {:.4f} Acc: {:.4f}".format(epoch+1, num_epochs, running_loss, running_corrects / train_images.shape[0]))
        else:
            print('Epoch [{}/{}], Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, running_loss, running_corrects / train_images.shape[0]))
    
    return selector_model

def test_demo(selector_model, model, test_images, test_labels, classes, attributions, net_layer, layer_name, top_m_neurons, log):
    selector_model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selector_model.to(device)
    
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
                                                            train_images, 
                                                            train_labels, 
                                                            important_neurons_dict, 
                                                            layer_name)
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        outputs = selector_model(test_images.to(device))
        _, preds = torch.max(outputs, 1)
        attr_label = string_to_one_hot(attributions, optimal_method)
        attr_label_ex = attr_label.repeat(test_images.shape[0], 1).squeeze(1)
        attr_label_ex = attr_label_ex.to(device)
        loss = criterion(outputs, attr_label_ex)
        running_corrects = torch.sum(preds == attr_label_ex.data)

        acc = running_corrects / test_images.shape[0]
        
        if log is not None:
            log.logger.info("Test Loss: {:.4f} Acc: {:.4f}".format(loss.item(), acc))
        else:
            print("Test Loss: {:.4f} Acc: {:.4f}".format(loss.item(), acc))
        
        return acc

def test_demo_with_layerinfo(model, selector_model, test_images, test_labels, layer_info, attributions, net_layer, layer_name, extract_layer_name, top_m_neurons, log):
    mid_features = []
    selector_model.cpu()
    
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
                                                            train_images, 
                                                            train_labels, 
                                                            important_neurons_dict, 
                                                            layer_name)
    
    def hook_fn(module, input, output):
        mid_features.append(output.detach())  # Store the output
        
    # Register the hook on the specified layer
    handle = None
    for name, module in model.named_modules():
        if name == extract_layer_name:
            handle = module.register_forward_hook(hook_fn)
            break

    if handle is None:
        raise ValueError(f"Layer {extract_layer_name} not found in the model.")

    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        outputs_class = model(test_images)
        layer_info_repeated = layer_info.unsqueeze(0).repeat(mid_features[0].shape[0], 1)
        combined_features = torch.cat((mid_features[0], layer_info_repeated), dim=1)
        outputs_attr = selector_model.fc3.forward(combined_features)
        
        _, preds_class = torch.max(outputs_class, 1)
        _, preds_attr = torch.max(outputs_attr, 1)
        
        attr_label = string_to_one_hot(attributions, optimal_method)
        attr_label_ex = attr_label.repeat(test_images.shape[0], 1).squeeze(1)
        
        loss = criterion(outputs_attr, attr_label_ex)
        running_corrects_class = torch.sum(preds_class == test_labels.data)
        running_corrects_attr = torch.sum(preds_attr == attr_label_ex.data)
        
        acc_class = running_corrects_class / test_images.shape[0]
        acc_attr = running_corrects_attr / test_images.shape[0]
        
        if log is not None:
            log.logger.info("Test Loss: {:.4f} Acc: {:.4f}".format(loss.item(), acc_attr))
        else:
            print("Test Loss: {:.4f} Acc: {:.4f}".format(loss.item(), acc_attr))
    
    handle.remove()
    return acc_class, acc_attr

if __name__ == '__main__':
    args = parse_args()
    model, selector_model, module_name, module, trainable_module, trainable_module_name, device, log = prapared_parameters(args)

    ### Data settings
    trainloader, testloader, classes = prepare_data(args)    
    
    # attributions = ['lc', 'la', 'ii', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp']
    attributions = ['lc', 'la', 'ii', 'ldl', 'lgs', 'lig', 'lfa', 'lrp']
    # attributions = ['lfa', 'ldl']
        
    ## Saved the index for comparing the Common Neurons across the attributions
    # {'lc': [1, 2, 3, 4, 5], 'la': [1, 2, 3, 4, 5], ...}
    attr_dict = {key: [] for key in attributions}
        
    # Save the original model state (Before pruning)
    original_state = copy.deepcopy(model.state_dict())
    
    # Step 1: Extract layer information
    layer_info = get_layer_info(trainable_module_name[args.layer_index], trainable_module_name)
    
    num_out = len(attributions)
    # num_ft = trainable_module[-1].in_features
    selector_model = copy.deepcopy(model)
    num_epochs = 40
    train_iters = 2
    for param in selector_model.parameters():
        param.requires_grad = False
    
    selector_model.fc3 = nn.Linear(trainable_module[-1].in_features+layer_info.shape[0], num_out)
    trained_fc3 = selector_model.fc3

    best_acc = 0
    for epoch in range(num_epochs):
        for test_class in classes:
            train_images, train_labels = get_class_data(trainloader, classes, test_class)
            selector_model_choice = train_demo(attributions=attributions, 
                    model=model,
                    selector_model=trained_fc3,
                    num_epochs=train_iters,
                    classes=classes,
                    net_layer=trainable_module[args.layer_index], 
                    layer_name=trainable_module_name[args.layer_index], 
                    top_m_neurons=args.top_m_neurons, 
                    original_state=original_state, 
                    layer_info=layer_info, 
                    train_images=train_images,
                    train_labels=train_labels,
                    log=log)

        if epoch % 5 == 0:
            if log is not None:
                log.logger.info("Label (During Training): {}".format(test_class))
                log.logger.info("Best Acc: {:.4f}, Epoch: {}".format(best_acc, epoch))
            else:
                print("Label (During Training): {}, Epoch: {}".format(test_class, epoch))
                print("Best Acc: {:.4f}".format(best_acc))
            test_images, test_labels = get_class_data(testloader, classes, test_class)
            acc_class, test_acc = test_demo_with_layerinfo(model, selector_model, test_images, test_labels, layer_info, attributions, trainable_module[args.layer_index], trainable_module_name[args.layer_index], trainable_module_name[-2], args.top_m_neurons, log)
            # test_ac = test_demo(selector_model, model, test_images, test_labels, classes, attributions, trainable_module[args.layer_index], trainable_module_name[args.layer_index], args.top_m_neurons, log)
            if best_acc < test_acc:
                # save_model(selector_model, 'selector_lenet_cifar10')
                best_acc = test_acc
                print("Class Acc: {}, Attr Acc: {}".format(acc_class, test_acc))
            
    for test_class in classes:
        if log is not None:
            log.logger.info("Label (Testing): {}".format(test_class))
        else:
            print("Label (Testing): {}".format(test_class))
        test_images, test_labels = get_class_data(testloader, classes, test_class)
        # test_acc = test_demo(selector_model, test_images, test_labels, classes, attributions, log)
        # test_acc = test_demo(selector_model, model, test_images, test_labels, classes, attributions, trainable_module[args.layer_index], trainable_module_name[args.layer_index], args.top_m_neurons, log)
        acc_class, test_acc = test_demo_with_layerinfo(model, selector_model, test_images, test_labels, layer_info, attributions, trainable_module[args.layer_index], trainable_module_name[args.layer_index], trainable_module_name[-2], args.top_m_neurons, log)

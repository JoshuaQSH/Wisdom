# wisdom/utils/common.py
import os
import json
import pickle
import hashlib

import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.utils.data import Subset

import numpy as np
from models_info.models_cv import *


def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def stable_selection_hash(selected, impl, cluster_cfg) -> str:
    h = hashlib.sha1()
    h.update(repr(sorted((k, tuple(sorted(v))) for k, v in selected.items())).encode())
    h.update(str(impl).encode())
    h.update(repr(cluster_cfg).encode())
    return h.hexdigest()[:16]
        
# Decide which testing mode is active
def _select_testing_mode(args) -> dict:
    # Return a dictionary with boolean values for each mode
    testing_mode =  {
        'end2end': bool(args.end2end),
        'all_class': bool(args.all_class),
        'class_iters': bool(args.class_iters)
    }
    
    # Build list of active modes with alternative descriptions for False cases
    mode_descriptions = []
    if testing_mode['end2end']:
        mode_descriptions.append('End2End-Testing')
    else:
        mode_descriptions.append('Single-Layer-Testing')
        
    if testing_mode['all_class']:
        mode_descriptions.append('All-Class-Testing')
    else:
        mode_descriptions.append('Class-Wise-Testing')
        
    if testing_mode['class_iters']:
        mode_descriptions.append('Iterating-All-Class: On')
    else:
        mode_descriptions.append('Iterating-All-Class: Off')
        
    return testing_mode, mode_descriptions

def convert_tensors(obj):
    """Recursively convert Tensors to lists"""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert tensor to a list
    elif isinstance(obj, dict):
        return {k: convert_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors(v) for v in obj]
    else:
        return obj

def save_json(filename, saved_data):
    with open(filename, 'w') as json_file:
        json.dump(saved_data, json_file, indent=4)

def load_json(filename):
    with open(filename, 'r') as json_file:
        saved_data = json.load(json_file)
    return saved_data

def normalize_tensor(featrues):
    featrues -= featrues.min()
    featrues /= featrues.max()
    return featrues


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

# Save the torch (DNN) model
def save_model(model, model_name):
    torch.save(model.state_dict(), model_name + '.pt')
    torch.save(model, model_name + '_whole.pth')
    print("Model state saved as", model_name + '.pt')
    print("Whole model saved as", model_name + '_whole.pth')


def save_cluster_groups(cluster_groups, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(cluster_groups, f)

def load_cluster_groups(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


#------------
# Trainable modules and model loading
#------------

def get_trainable_modules_main(model, prefix=''):
    
    trainable_module = []
    trainable_module_name = []
    
    def get_trainable_modules(model, prefix=''):
        for name, layer in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)) and any(p.requires_grad for p in layer.parameters()):
                trainable_module_name.append(full_name)
                trainable_module.append(layer)
            get_trainable_modules(layer, full_name)
    get_trainable_modules(model)
    return trainable_module, trainable_module_name

def get_layer_by_name(model, layer_name):
    parts = layer_name.split('.')
    layer = model
    for part in parts:
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)
    return layer

def get_model(load_model_path='./models_info/saved_models/lenet_CIFAR10_whole.pth'):
    module_name = []
    module = []
    model = torch.load(load_model_path, weights_only=False)
    
    # Alternatively, to get all submodule names (including nested ones)
    for name, layer in model.named_modules():
        module_name.append(name)
        module.append(layer)

    return model, module_name, module

def mnist_model_state2whole():
    load_model_path=['./models_info/saved_models/lenet_MNIST.pt']
    model_classes = { 'lenet': LeNet}
    for i, model_name in enumerate(model_classes):
        model = model_classes[model_name]()

        data_parallel_dict = torch.load(load_model_path[i])
        new_state_dict = {}
        for key, value in data_parallel_dict.items():
            new_key = key.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[new_key] = value    
        
        model.load_state_dict(new_state_dict)
        torch.save(model, load_model_path[i].replace('.pt', '_whole.pth'))
        print("Done with ", model_name)

def cifar_model_state2whole():
    load_model_path=['./models_info/saved_models/lenet_CIFAR10.pt', 
                     './models_info/saved_models/vgg16_CIFAR10.pt', 
                     './models_info/saved_models/resnet18_CIFAR10.pt',
                     './models_info/saved_models/densenet_CIFAR10.pt',
                     './models_info/saved_models/mobilenetv2_CIFAR10.pt',
                     './models_info/saved_models/shufflenetv2_CIFAR10.pt',
                     './models_info/saved_models/efficientnet_CIFAR10.pt']
    
    model_classes = {
        'lenet': LeNet,
        'vgg16': lambda: VGG('VGG16'),
        'resnet18': ResNet18,
        # 'googlenet': GoogLeNet,
        'densenet': DenseNet121,
        # 'resnext29': ResNeXt29_2x64d,
        'mobilenetv2': MobileNetV2,
        'shufflenetv2': lambda: ShuffleNetV2(1),
        # 'senet': SENet18,
        # 'preresnet': PreActResNet18,
        # 'mobilenet': MobileNet,
        # 'DPN92': DPN92,
        'efficientnet': EfficientNetB0,
        # 'regnet': RegNetX_200MF,
        # 'simpledla': SimpleDLA,
    }

    for i, model_name in enumerate(model_classes):
        model = model_classes[model_name]()

        data_parallel_dict = torch.load(load_model_path[i])
        new_state_dict = {}
        for key, value in data_parallel_dict.items():
            new_key = key.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[new_key] = value    
        
        model.load_state_dict(new_state_dict)
        torch.save(model, load_model_path[i].replace('.pt', '_whole.pth'))
        print("Done with ", model_name)

def imagenet_model_state2whole():
    # Hardcoded model names for now
    offer_moder_name = ['vgg16', 
                        'convnext_base', 
                        'efficientnet_v2_s', 
                        'efficientnet_v2_m', 
                        'mnasnet1_0', 
                        'googlenet',
                        'inception_v3',
                        'mobilenet_v3_small',
                        'resnet18',
                        'resnet152',
                        'resnext101_32x8d',
                        'vit_b_16']
    
     # Check if model_name is in the list
    for model_name in offer_moder_name:
        # Dynamically get the model function from torchvision.models
        model_func = getattr(models, model_name)
        
        # Dynamically get the weights attribute for the model
        # "IMAGENET1K_V2", "IMAGENET1K_V1"
        model = model_func(weights="IMAGENET1K_V1")        
        print(f"{model_name} model loaded with weights.")
        torch.save(model, f"./models_info/saved_models/{model_name}_IMAGENET_whole.pth")
        print("Done with ", model_name)

def get_class_data(dataloader, classes, target_class):
    max_test_sample = 8000
    class_index = classes.index(target_class)

    filtered_data = []
    filtered_labels = []
    for inputs, labels in dataloader:
        for i, l in zip(inputs, labels):
            if l == class_index:
                filtered_data.append(i)
                filtered_labels.append(l)  
        if len(filtered_data) >= max_test_sample:
            break
    
    if filtered_data:
        return torch.stack(filtered_data), torch.tensor(filtered_labels)
    else:
        return None, None

def extract_class_to_dataloder(dataset, classes, batch_size=100, target_class_name=None):
    # If no specific class is requested, return ordered loader with all classes
    if target_class_name is None:
        class_indices = {i: [] for i in range(len(classes))}
        
        # Populate the dictionary with indices
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
        
        ordered_indices = [idx for class_id in range(len(classes)) for idx in class_indices[class_id]]
        ordered_subset = Subset(dataset, ordered_indices)
        ordered_loader = DataLoader(ordered_subset, batch_size=batch_size, shuffle=False)
        
        return ordered_loader
    
    # Find the class index for the target class name
    if target_class_name not in classes:
        raise ValueError(f"Class '{target_class_name}' not found in classes list")
    
    target_class_index = classes.index(target_class_name)
    
    # Find all indices that belong to the target class
    target_indices = []
    for idx, (_, label) in enumerate(dataset):
        if label == target_class_index:
            target_indices.append(idx)
    
    if not target_indices:
        raise ValueError(f"No samples found for class '{target_class_name}'")
    
    # Create subset with only the target class data
    target_subset = Subset(dataset, target_indices)
    target_loader = DataLoader(target_subset, batch_size=batch_size, shuffle=False)
    
    return target_loader

## An end-to-end test for the model (randomly pickup a bunch of images)
def extract_random_class(test_dataset, test_all=False, num_samples=1000):

    if test_all:
        subset_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    else:
        indices = torch.randperm(len(test_dataset))[:num_samples]
        subset = Subset(test_dataset, indices)
        subset_loader = DataLoader(subset, batch_size=1, shuffle=False)
    
    test_image = []
    test_label = []

    # Iterate through the DataLoader
    for images, labels in subset_loader:
        test_image.append(images)
        test_label.append(labels)

    # Concatenate all batches into single tensors
    test_image = torch.cat(test_image, dim=0)
    test_label = torch.cat(test_label, dim=0)

    return subset_loader, test_image, test_label

# Evaluate the model on the given dataloader and compute accuracy, loss, and F1 score.
def eval_model_dataloder(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    from sklearn.metrics import f1_score
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

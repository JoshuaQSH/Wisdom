import argparse
import os
import urllib
import urllib.request
import json
import logging
from logging import handlers
import time
import pickle

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Subset

import matplotlib.pyplot as plt
from captum.attr import visualization as viz
import numpy as np

from sklearn.model_selection import train_test_split

from models_info.models_cv import *
# from models_info.YOLOv5.yolo import *
# from models_info.YOLOv5.datasets import *



#------------
# Argument parsing
#------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lenet', help='Model to use for training.')
    parser.add_argument('--saved-model', type=str, default='/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth', help='Saved model name.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'imagenet'], help='The dataset to use for training and testing.')
    parser.add_argument('--data-path', type=str, default='./datasets/', help='Path to the data directory.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training.')
    parser.add_argument('--random-prune', action='store_true', help='Randomly prune the neurons.')
    parser.add_argument('--use-silhouette', action='store_true', help='Whether to use silhouette score for clustering.')
    parser.add_argument('--n-clusters', type=int, default=2, help='Number of clusters to use for KMeans.')
    parser.add_argument('--top-m-neurons', type=int, default=5, help='Number of top neurons to select.')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training.')

    # Testing Mode arguments
    parser.add_argument('--test-image', type=str, default='1', help='Test image name. For the single image testing. (against with the `all-class`).')
    parser.add_argument('--all-class', action='store_true', help='Attributions collected for all the classes. When activated, it will equal to batch testing.')
    parser.add_argument('--class-iters', action='store_true', help='Only valided when doing class-wise testing. If set, the model will be tested for each class separately.')
    parser.add_argument('--end2end', action='store_true', help='End to end testing for the whole model.')
    parser.add_argument('--idc-test-all', action='store_true', help='Using all the test images for the Coverage testing. Other wise will only sample some images from the test set.')
    parser.add_argument('--num-samples', type=int, default=0, help='Sampling number for the test images (against with the `idc-test-all`).')
    parser.add_argument('--attr', type=str, default='lc', choices=['lc', 'la', 'ii', 'lgxa', 'lgc', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp', 'random', 'wisdom'],  help='The attribution method to use.')
    parser.add_argument('--layer-index', type=int, default=1, help='Get the layer index for the model, should start with 1')

    
    # General arguments
    # parser.add_argument('--vis-attributions', action='store_true', help='Visualize the attributions.')
    # parser.add_argument('--viz', action='store_true', help='Visualize the input and its relevance.')
    parser.add_argument('--logging', action="store_true", help="Whether to log the training process")
    parser.add_argument('--log-path', type=str, default='./logs/TestLog', help='Path (and name) to save the log file.')
    parser.add_argument('--inordered-dataset', action='store_true', help='Whether the dataset is ordered.')
    parser.add_argument('--csv-file', type=str, default='demo_layer_scores.csv', help='The file to save the layer scores.')

    args = parser.parse_args()
    # print(args)
    
    return args


#------------
# Logger configuration
#------------
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }
    def __init__(self, 
                 filename, 
                 level='info',
                 when='D',
                 backCount=3,
                 fmt='%(asctime)s - %(pathname)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,
                                               when=when,
                                               backupCount=backCount,
                                               encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)

# Logger configuration
def _configure_logging(enable_logging: bool, args, level: str = "info") -> logging.Logger:
    if not enable_logging:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        return logging.getLogger(__name__)

    log_level = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "crit": logging.CRITICAL,
    }.get(level.lower(), logging.INFO)
    
    start_ms = int(time.time() * 1000)
    timestamp = time.strftime("%Y%m%d‑%H%M%S", time.localtime(start_ms / 1000))
    logfile = f"{args.log_path}-{args.model}-{args.dataset}-{timestamp}.log"

    logger = logging.getLogger("Wisdom")
    logger.setLevel(log_level)
    
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.debug(
        "[=== Model: %s, Dataset: %s, Layers_Index: %s, Topk: %s ===]",
        args.model,
        args.dataset,
        args.layer_index,
        args.top_m_neurons,
    )
    return logger


#------------
# Helper functions
#------------
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


#------------
# Dataloader and dataset functions
#------------

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets, datasets['train'], datasets['val']

def data_loader(root, batch_size=256, workers=1, pin_memory=True, shuffle=False):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, train_dataset, val_dataset

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets

#------------
# Loading datasets
#------------
# Load the ImageNet dataset
def load_ImageNet(batch_size=32, root='./datasets/ImageNet', num_workers=2, use_val=False, label_path='./datasets/imagenet_labels.json'):
    
    val_path = os.path.join(root, 'val/')
    
    if not os.path.exists(label_path):
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        urllib.request.urlretrieve(url, "imagenet_labels.json")
        label_path = "imagenet_labels.json"

    # Load the labels from the JSON file
    with open(label_path) as f:
        classes = json.load(f)
    
    if use_val:
        # Optional: use val_dataset as the training dataset for shorter training time
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
        datasets, train_dataset, val_dataset  = train_val_dataset(val_dataset, val_split=0.25)
        dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                       for x in ['train', 'val']}
        trainloader = dataloaders['train']
        testloader = dataloaders['val']
    else:
        trainloader, testloader, train_dataset, val_dataset = data_loader(root, batch_size, num_workers, True)

    return trainloader, testloader, train_dataset, val_dataset, classes

#  Load the CIFAR-10 dataset
def load_CIFAR(batch_size=32, root='./datasets', shuffle=True):

    transform = transforms.Compose([
         transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, train_dataset, test_dataset, classes


#  Load the MNIST dataset
def load_MNIST(batch_size=32, root='./datasets', channel_first=False, train_all=False):
    # transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    
    transform_list = [
        transforms.Resize(32),  # Upscale from 28x28 to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

    if channel_first:
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))  # If you want 3 channels
    transform = transforms.Compose(transform_list)

    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)
    
    if train_all:
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    return train_loader, test_loader, train_dataset, test_dataset, classes


def get_data(dataset_name, batch_size, data_path):
    ### Dataset settings
    if dataset_name == 'cifar10':
        trainloader, testloader, train_dataset, test_dataset, classes = load_CIFAR(batch_size=batch_size, root=data_path, shuffle=True)
    elif dataset_name == 'mnist':
        trainloader, testloader, train_dataset, test_dataset, classes = load_MNIST(batch_size=batch_size, root=data_path)
    elif dataset_name == 'imagenet':
        trainloader, testloader, train_dataset, test_dataset, classes = load_ImageNet(batch_size=batch_size, 
                                                         root=data_path + '/ImageNet', 
                                                         num_workers=2, 
                                                         use_val=False)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    
    return trainloader, testloader, train_dataset, test_dataset, classes

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#------------
# Visualize the attributions
#------------
def viz_attr(img, attr, dataset_name, model_name, with_original=False):
    attr_np = attr.cpu().detach().numpy()
    img_np  = img.cpu().detach().numpy()
    
    if with_original:
        fig, ax = viz.visualize_image_attr_multiple(np.transpose(attr_np, (1, 2, 0)),
                                        np.transpose(img_np, (1, 2, 0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        show_colorbar=True,
                                        outlier_perc=2,
                                        use_pyplot=False)
    else:
        fig, ax = viz.visualize_image_attr(np.transpose(attr_np, (1, 2, 0)),
                                        np.transpose(img_np, (1, 2, 0)),
                                        "heat_map",
                                        "positive",
                                        show_colorbar=True,
                                        outlier_perc=2,
                                        use_pyplot=False)

    fig.tight_layout()
    fig.savefig(f"{dataset_name}_{model_name}.png",
                dpi=300,
                bbox_inches="tight")
    plt.close(fig)

# This is for forming a custom dataset
class SelectorDataset(torch.utils.data.Dataset):
    def __init__(self, image_dataset, layer_info, attribution_labels, attribution_methods):
        super(SelectorDataset, self).__init__()
        self.image_dataset = image_dataset
        self.layer_info = layer_info
        self.attribution_labels = attribution_labels
        # Map method to index
        self.method_to_idx = {method: idx for idx, method in enumerate(attribution_methods)}  

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # Get image and label from the original dataset
        image, _ = self.image_dataset[idx]
        
        # Get layer information for the current sample
        layer_info = self.layer_info[idx]
        
        # Convert attribution method to index
        attribution_label = self.method_to_idx[self.attribution_labels[idx]]
        
        return image, layer_info, torch.tensor(attribution_label, dtype=torch.long)


#------------
# Patching and saving models
#------------
# Quick patch for torchvision resnet models
# ResNet implmented in torchvision is not suitable for the Captum attribution methods, a little customizations are made in the model definition.
def patch_resnet_torchvision(src, dst_model):
    src_sd = src.state_dict() if isinstance(src, torch.nn.Module) else src
    dst_sd = dst_model.state_dict()

    def rename(key: str) -> str:
        return (key
                .replace(".downsample.0.", ".shortcut_conv.")
                .replace(".downsample.1.", ".shortcut_bn."))

    patched = {rename(k): v for k, v in src_sd.items()
               if rename(k) in dst_sd and v.shape == dst_sd[rename(k)].shape}

    merged = dst_sd.copy()
    merged.update(patched)
    return merged

def patch_resnet_imagenet(dst_model, saved_model_weight='./models_info/saved_models/resnet18_IMAGENET_whole.pth', model_name='resnet18'):
    
    if os.path.exists(saved_model_weight):
         src_model, _, _ = get_model(saved_model_weight)
    else:
        print(f"Model weight file {saved_model_weight} does not exist, try to download it first.")
        model_func = getattr(models, model_name)
        src_model = model_func(weights="IMAGENET1K_V1") 
    
    src_model.eval()
    dst_model.eval()
    
    patched_sd  = patch_resnet_torchvision(src_model, dst_model)
    incompat = dst_model.load_state_dict(patched_sd, strict=False)
    assert incompat.missing_keys == [], f"Missing keys in destination model: {incompat.missing_keys}"
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        d = (src_model(x) - dst_model(x)).abs().max().item()
        assert d < 1e-5, f"Model patching failed, max diff: {d}"
    
    torch.save(dst_model.state_dict(), './models_info/saved_models/{}_IMAGENET_patched.pt'.format(model_name))
    torch.save(dst_model, './models_info/saved_models/{}_IMAGENET_patched_whole.pth'.format(model_name))
    
    return dst_model 
    

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

# See if a neuron is activated
def is_neuron_activated(activation_value, threshold=0):
    return activation_value > threshold

def is_conv2d_neuron_activated(activation_map, threshold=0):
    return (activation_map > threshold).any().item()


if __name__ == '__main__':
    # unit test - model
    args = parse_args()



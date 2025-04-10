import argparse
import os
import sys
from pathlib import Path
import urllib
import urllib.request
import json
import random
import logging
from logging import handlers

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

from sklearn.model_selection import train_test_split

# Add src directory to sys.path
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

import model_hub
# from model_hub import LeNet, Net
from models_cv import *
from YOLOv5.yolo import *
from YOLOv5.datasets import *

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
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lenet', help='Model to use for training.')
    parser.add_argument('--saved-model', type=str, default='lenet_CIFAR10.pth', help='Saved model name.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'imagenet'], help='The dataset to use for training and testing.')
    parser.add_argument('--data-path', type=str, default='/data/shenghao/dataset/', help='Path to the data directory.')
    parser.add_argument('--importance-file', type=str, default='./saved_files/plane_lenet_importance.json', help='The file to save the importance scores.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training.')
    parser.add_argument('--large-image', action='store_true', help='Use CIFAR-10 dataset with the resized images.')
    parser.add_argument('--random-prune', action='store_true', help='Randomly prune the neurons.')
    parser.add_argument('--use-silhouette', action='store_true', help='Whether to use silhouette score for clustering.')
    parser.add_argument('--n-clusters', type=int, default=2, help='Number of clusters to use for KMeans.')
    parser.add_argument('--top-m-neurons', type=int, default=5, help='Number of top neurons to select.')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training.')

    # IDC Testing 
    parser.add_argument('--test-image', type=str, default='plane', help='Test image name. For the single image testing.')
    parser.add_argument('--all-class', action='store_true', help='Attributions collected for all the classes.')
    parser.add_argument('--idc-test-all', action='store_true', help='Using all the test images for the Coverage testing.')
    parser.add_argument('--num-samples', type=int, default=0, help='Sampling number for the test images (against with the `idc-test-all`).')

    parser.add_argument('--attr', type=str, default='lc', choices=['lc', 'la', 'ii', 'lgxa', 'lgc', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp'],  help='The attribution method to use.')
    parser.add_argument('--layer-index', type=int, default=1, help='Get the layer index for the model, should start with 1')
    parser.add_argument('--layer-by-layer', action='store_true', help='Capturing all the module layer in the model, same as end2end.')
    parser.add_argument('--end2end', action='store_true', help='End to end testing for the whole model.')
    
    # General arguments
    # parser.add_argument('--vis-attributions', action='store_true', help='Visualize the attributions.')
    # parser.add_argument('--viz', action='store_true', help='Visualize the input and its relevance.')
    parser.add_argument('--logging', action="store_true", help="Whether to log the training process")
    parser.add_argument('--log-path', type=str, default='./logs/TestLog', help='Path (and name) to save the log file.')
    parser.add_argument('--csv-file', type=str, default='demo_layer_scores.csv', help='The file to save the layer scores.')

    args = parser.parse_args()
    # print(args)
    
    return args

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

# Load COCO dataset
def load_COCO_old(batch_size=32, root='/home/shenghao/torch-deepimportance/yolov5/data/coco', num_workers=2):
    # Define paths to annotations
    ann_train = os.path.join(root, 'annotations', 'instances_train2017.json')
    ann_val = os.path.join(root, 'annotations', 'instances_val2017.json')

    # Define image directories
    img_dir_train = os.path.join(root, 'images', 'train2017')
    img_dir_val = os.path.join(root, 'images', 'val2017')

    # Define transformations
    # 1, 3, 640, 640
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = CocoDetection(root=img_dir_train, annFile=ann_train, transform=transform)
    val_dataset = CocoDetection(root=img_dir_val, annFile=ann_val, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # COCO class names
    classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    return train_loader, val_loader, classes

def load_COCO(batch_size=32, data_path='../data/elephant.yaml', img_size=[640, 640], 
              cfg='models/yolov5s.yaml',
              model_stride=[8, 16, 32],
              single_cls=True, 
              cache_images=True, 
              rect=True, 
              num_workers=2):
    
    hyp = {'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 5e-4,  # optimizer weight decay
       'giou': 0.05,  # giou loss gain
       'cls': 0.58,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'anchor_t': 4.0,  # anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)
    
    data_path = glob.glob('./**/' + data_path, recursive=True)[0]
    with open(data_path) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])

    # Image sizes
    gs = int(max(model_stride))  # grid size (max stride)
    if any(x % gs != 0 for x in img_size):
        print('WARNING: --img-size %g,%g must be multiple of %s max stride %g' % (*img_size, cfg, gs))
    imgsz, imgsz_test = [make_divisible(x, gs) for x in img_size]  # image sizes (train, test)

    dataset = LoadImagesAndLabels(train_path, imgsz, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=rect,  # rectangular training
                                  cache_images=cache_images,
                                  single_cls=single_cls)
    
    trainloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=not rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=cache_images,
                                                                 single_cls=single_cls),
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # class frequency
    labels = np.concatenate(dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes

    return trainloader, testloader, c

# Load the ImageNet dataset
def load_ImageNet(batch_size=32, root='/data/shenghao/dataset/ImageNet', num_workers=2, use_val=False):
    
    val_path = os.path.join(root, 'val/')
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    urllib.request.urlretrieve(url, "imagenet_labels.json")

    # Load the labels from the JSON file
    with open("imagenet_labels.json") as f:
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
def load_CIFAR(batch_size=32, root='./data', large_image=False, shuffle=True):

    if large_image:
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    else:
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
def load_MNIST(batch_size=32, root='./data', channel_first=False, train_all=False):
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

# Save the torch (DNN) model
def save_model(model, model_name):
    torch.save(model.state_dict(), model_name + '.pt')
    torch.save(model, model_name + '_whole.pth')
    print("Model state saved as", model_name + '.pt')
    print("Whole model saved as", model_name + '_whole.pth')


# Save the k-means model
def save_kmeans_model(kmeans, filename='kmeans_model.pkl'):
    # joblib.dump(kmeans, filename)
    import pickle
    with open(filename, "wb") as file:
        pickle.dump(kmeans, file)

# Load the k-means model
def load_kmeans_model(filename='kmeans_model.pkl'):
    # return joblib.load(filename)
    import pickle
    with open(filename, "rb") as file:
        kmeans_models_loaded = pickle.load(file)
    return kmeans_models_loaded

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

def get_model(load_model_path='/home/shenghao/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth'):
    module_name = []
    module = []
    model = torch.load(load_model_path)
    
    # Alternatively, to get all submodule names (including nested ones)
    for name, layer in model.named_modules():
        module_name.append(name)
        module.append(layer)

    return model, module_name, module

def mnist_model_state2whole():
    load_model_path=['/home/shenghao/torch-deepimportance/models_info/saved_models/lenet_MNIST.pt']
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
    load_model_path=['/home/shenghao/torch-deepimportance/models_info/saved_models/lenet_CIFAR10.pt', 
                     '/home/shenghao/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10.pt', 
                     '/home/shenghao/torch-deepimportance/models_info/saved_models/resnet18_CIFAR10.pt',
                     '/home/shenghao/torch-deepimportance/models_info/saved_models/densenet_CIFAR10.pt',
                     '/home/shenghao/torch-deepimportance/models_info/saved_models/mobilenetv2_CIFAR10.pt',
                     '/home/shenghao/torch-deepimportance/models_info/saved_models/shufflenetv2_CIFAR10.pt',
                     '/home/shenghao/torch-deepimportance/models_info/saved_models/efficientnet_CIFAR10.pt']
    
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
        torch.save(model, f"/home/shenghao/torch-deepimportance/models_info/saved_models/{model_name}_IMAGENET_whole.pth")
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

def extract_class_to_dataloder(dataset, classes, batch_size=100):
    class_indices = {i: [] for i in range(len(classes))}
    
    # Populate the dictionary with indices
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    ordered_indices = [idx for class_id in range(len(classes)) for idx in class_indices[class_id]]
    ordered_subset = Subset(dataset, ordered_indices)
    ordered_loader = DataLoader(ordered_subset, batch_size=batch_size, shuffle=False)
    
    return ordered_loader

# Evaluate the model on the given dataloader and compute accuracy, loss, and F1 score.
def test_model_dataloder(model, dataloader, device='cpu'):
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

## An end-to-end test for the model (randomly pickup a bunch of images)
def test_random_class(test_dataset, test_all=False, num_samples=1000):

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

# See if a neuron is activated
def is_neuron_activated(activation_value, threshold=0):
    return activation_value > threshold

def is_conv2d_neuron_activated(activation_map, threshold=0):
    return (activation_map > threshold).any().item()

## For the unit test
def test_model():
    offer_model_name = ['vgg16', 'lenet',
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
    offer_model_name = ['mobilenet_v3_small', 'efficientnet_v2_s', 'convnext_base']
    for model_name in offer_model_name:
         model, module_name, module = get_model(model_name=model_name)
         print(model_name, len(module_name))
         # print(model_name, module_name)
         total_params = sum(p.numel() for p in model.parameters())
         print(f"Total number of parameters: {total_params}")

# Test the CIFAR models
def test_cifar_models():
    model_classes = {
    'lenet': LeNet,
    'vgg16': lambda: VGG('VGG16'),
    'resnet18': ResNet18,
    'googlenet': GoogLeNet,
    'densenet': DenseNet121,
    'resnext29': ResNeXt29_2x64d,
    'mobilenetv2': MobileNetV2,
    'shufflenetv2': lambda: ShuffleNetV2(1),
    'senet': SENet18,
    'preresnet': PreActResNet18,
    'mobilenet': MobileNet,
    'DPN92': DPN92,
    'efficientnet': EfficientNetB0,
    'regnet': RegNetX_200MF,
    'simpledla': SimpleDLA,}

    model_list = ['lenet', 'vgg16', 'resnet18', 'googlenet', 'mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2', 'senet', 'preresnet', 'densenet', 'resnext']

    for model_name in model_list:
        if model_name in model_classes:
            model = model_classes[model_name]()
            print(model_name)    
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters: {total_params}")
            x = torch.randn(1, 3, 32, 32)
            y = model(x)
            print(y.size())

def test_selector_dataloader(root):
    
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)

    layer_info = torch.tensor([0., 0., 1., 0., 0.]).repeat(len(train_dataset), 1)  # One-hot vector for all images
    # Example list of attribution methods
    attribution_labels = ['lc', 'la', 'ii', 'lc', 'ldl', 'lig', 'lgs', 'lgxa', 'lrp', 'lfa'] * 5000
    attribution_methods = ['lc', 'la', 'ii', 'lgxa', 'lgc', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp']

    # Create the custom dataset
    selector_train_dataset = SelectorDataset(train_dataset, layer_info, attribution_labels, attribution_methods)

    # Create DataLoader for training
    trainloader = DataLoader(selector_train_dataset, batch_size=32, shuffle=True, num_workers=2)

    # Create DataLoader for testing (similarly, you need to create the corresponding test dataset)
    selector_test_dataset = SelectorDataset(test_dataset, layer_info, attribution_labels, attribution_methods)
    testloader = torch.utils.data.DataLoader(selector_test_dataset, batch_size=32, shuffle=False, num_workers=2)
    for images, layer_info, labels in trainloader:
        print(f"Images shape: {images.shape}")
        print(f"Layer info shape: {layer_info.shape}")
        print(f"Labels shape: {labels.shape}")
        break

if __name__ == '__main__':
    # unit test - model
    args = parse_args()
    # subset_loader, test_image, test_label = test_random_class(test_dataset, test_all=True, num_samples=1000)
    # trainloader, testloader, test_dataset, classes = load_ImageNet()
    # print("Classes: ", classes)
    # test_model()
    # test_cifar_models()
    # test_selector_dataloader(root = args.data_path)
    # trainloader, testloader, c = load_COCO()
    # model, module_name, module = get_model('/home/shenghao/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth')
    # print(model)
    # print(module)    

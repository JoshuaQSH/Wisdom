import argparse
import os
import sys
from pathlib import Path
import urllib
import urllib.request
import json
import logging
from logging import handlers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score


# Add src directory to sys.path
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

from model_hub import LeNet, Net

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
    parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'vgg16', 'custom'], help='Model to use for training.')
    parser.add_argument('--model-path', type=str, default='None', help='Path to the trained model.')
    parser.add_argument('--saved-model', type=str, default='lenet_cifar10.pt', help='Saved model name.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'imagenet'], help='The dataset to use for training and testing.')
    parser.add_argument('--data-path', type=str, default='/data/shenghao/dataset/', help='Path to the data directory.')
    parser.add_argument('--importance-file', type=str, default='./saved_files/plane_lenet_importance.json', help='The file to save the importance scores.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training.')
    parser.add_argument('--large-image', action='store_true', help='Use CIFAR-10 dataset with the resized images.')
    parser.add_argument('--test-image', type=str, default='plane', help='Test image name.')
    parser.add_argument('--test-all', action='store_true', help='Test all the images.')
    parser.add_argument('--attr', type=str, default='lc', choices=['lc', 'la', 'ii', 'lgxa', 'lgc', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp'],  help='The attribution method to use.')
    parser.add_argument('--layer-index', type=int, default=1, help='Get the layer index for the model, should start with 1')
    parser.add_argument('--capture-all', action='store_true', help='Capture all the layers.')
    parser.add_argument('--end2end', action='store_true', help='End to end testing for the whole model.')
    parser.add_argument('--random-prune', action='store_true', help='Randomly prune the neurons.')
    parser.add_argument('--use-silhouette', action='store_true', help='Whether to use silhouette score for clustering.')
    parser.add_argument('--n-clusters', type=int, default=5, help='Number of clusters to use for KMeans.')
    parser.add_argument('--cluster-scores', action='store_true', help='Cluserting the importance scores rather than using actiation values.')
    parser.add_argument('--top-m-neurons', type=int, default=5, help='Number of top neurons to select.')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training.')
    
    # General arguments
    parser.add_argument('--vis-attributions', action='store_true', help='Visualize the attributions.')
    parser.add_argument('--viz', action='store_true', help='Visualize the input and its relevance.')
    parser.add_argument('--logging', action="store_true", help="Whether to log the training process")
    parser.add_argument('--log-path', type=str, default='./logs/', help='Path to save the log file.')



    
    args = parser.parse_args()
    print(args)
    
    return args

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def data_loader(root, batch_size=256, workers=1, pin_memory=True):
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
        shuffle=True,
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

    return train_loader, val_loader

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
        datasets = train_val_dataset(val_dataset, val_split=0.25)
        dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                       for x in ['train', 'val']}
        trainloader = dataloaders['train']
        testloader = dataloaders['val']
    else:
        trainloader, testloader = data_loader(root, batch_size, num_workers, True)

    return trainloader, testloader, classes

#  Load the CIFAR-10 dataset
def load_CIFAR(batch_size=32, root='./data', large_image=True):

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

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes


#  Load the MNIST dataset
def load_MNIST(batch_size=32, root='./data', channel_first=True, train_all=False):
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
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

    return train_loader, test_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Save the torch (DNN) model
def save_model(model, model_name):
    torch.save(model.state_dict(), model_name + '.pth')
    print("Model saved as", model_name + '.pth')


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
    
# TODO: Adding more models
def get_model(model_name='vgg16'):
    # Hardcoded model names for now
    offer_moder_name = ['lenet', 'custom', 
                        'vgg16', 
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
    module_name = []
    module = []
    # Check if model_name is in the list
    if model_name in offer_moder_name:
        if model_name == 'lenet':
            model = LeNet()
        elif model_name == 'custom':
            model = Net()
        else:
            # Dynamically get the model function from torchvision.models
            model_func = getattr(models, model_name)
        
            # Dynamically get the weights attribute for the model
            # "IMAGENET1K_V2", "IMAGENET1K_V1"
            model = model_func(weights="IMAGENET1K_V1")        
            print(f"{model_name} model loaded with weights.")
    else:
        raise(f"{model_name} not in the list of available models.")

    # Alternatively, to get all submodule names (including nested ones)
    for name, layer in model.named_modules():
        module_name.append(name)
        module.append(layer)
        
    return model, module_name, module

def get_class_data(dataloader, classes, target_class):
    class_index = classes.index(target_class)

    filtered_data = []
    filtered_labels = []
    for inputs, labels in dataloader:
        for i, l in zip(inputs, labels):
            if l == class_index:
                filtered_data.append(i)
                filtered_labels.append(l)
    
    if filtered_data:
        return torch.stack(filtered_data), torch.tensor(filtered_labels)
    else:
        return None, None

def prepare_data_cifa(data_path='../data', cifar10=True):
    
    if cifar10:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

# See if a neuron is activated
def is_neuron_activated(activation_value, threshold=0):
    return activation_value > threshold

def is_conv2d_neuron_activated(activation_map, threshold=0):
    """
    Check if any neuron in the Conv2D activation map is activated.
    
    Parameters:
    - activation_map (torch.Tensor): The activation map of shape (output_height, output_width).
    - threshold (float): The threshold to determine activation (default: 0 for ReLU).
    
    Returns:
    - activated (bool): True if the neuron is activated, False otherwise.
    """
    return (activation_map > threshold).any().item()

# Generate Adversarial Examples
def fgsm_attack(model, inputs, labels, epsilon):
    """
    Generate adversarial examples using the FGSM method.
    
    Parameters:
    - model (torch.nn.Module): The trained model.
    - inputs (torch.Tensor): Original input images.
    - labels (torch.Tensor): True labels for the inputs.
    - epsilon (float): The perturbation strength.
    
    Returns:
    - adv_examples (torch.Tensor): Generated adversarial examples.
    """
    inputs.requires_grad = True
    
    # Forward pass
    outputs = model(inputs)
    loss = F.nll_loss(outputs, labels)
    
    # Backward pass (calculate gradients)
    model.zero_grad()
    loss.backward()
    
    # Create adversarial perturbations
    perturbation = epsilon * inputs.grad.sign()
    
    # Create adversarial examples by adding perturbations to original inputs
    adv_examples = inputs + perturbation
    
    # Clip the values to stay within valid image range (0-1)
    adv_examples = torch.clamp(adv_examples, 0, 1)
    
    return adv_examples

def test_model():
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
    offer_moder_name = ['vgg16', 'convnext_base', 'efficientnet_v2_s']
    for model_name in offer_moder_name:
         model, module_name, module = get_model(model_name=model_name)
         print(model_name, len(module_name))
         print(model_name, module_name)

## TODO: bugs here, scatter plot is not working, for the shape of X
def plot_cluster_info(n_clusters, silhouette_avg, X, clusterer, cluster_labels):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    # The silhouette coefficient can range from -0.1, 1
    ax1.set_xlim([-0.1, 1])
    y_lower = 10
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    # [5000, 1] for CIFAR-10 here
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )
    
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    
    plt.savefig('./images/silhouette_n_{}.pdf'.format(n_clusters), dpi=1500)

def visualize_idc_scores(idc_scores, filename='./logs/idc_scores.pdf'):
    methods = list(idc_scores.keys())
    scores = [idc_scores[method] for method in methods]
    
    plt.figure(figsize=(12, 6))
    plt.bar(methods, scores, color='skyblue')
    plt.xlabel('Attribution Method')
    plt.ylabel('IDC Score (%)')
    plt.title('Comparison of IDC Scores for Different Attribution Methods')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=1200)
    # plt.show()


if __name__ == '__main__':
    # unit test - model
    args = parse_args()
    # train_loader, test_loader, classes = load_CIFAR(batch_size=32, root=args.data_path)
    # train_loader, test_loader, classes = load_CIFAR(batch_size=32, root=args.data_path)
    trainloader, testloader, classes = load_ImageNet()
    # print("Classes: ", classes)
    # test_model()
    
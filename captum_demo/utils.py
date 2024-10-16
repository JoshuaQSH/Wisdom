import numpy as np
import argparse
import joblib

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

from model_hub import LeNet, Net

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'vgg16', 'custom'], help='Model to use for training.')
    parser.add_argument('--model-path', type=str, default='None', help='Path to the trained model.')
    parser.add_argument('--saved-model', type=str, default='lenet_cifar10.pt', help='Saved model name.')
    parser.add_argument('--data-path', type=str, default='/data/shenghao/dataset/', help='Path to the data directory.')
    parser.add_argument('--importance-file', type=str, default='./saved_files/plane_lenet_importance.json', help='The file to save the importance scores.')
    parser.add_argument('--viz', action='store_true', help='Visualize the input and its relevance.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training.')
    parser.add_argument('--is-cifar10', action='store_true', help='Use CIFAR-10 dataset.')
    parser.add_argument('--test-image', type=str, default='plane', choices=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 
                        help='Test image name.')
    parser.add_argument('--layer-index', type=int, default=1, help='Get the layer index for the model, should start with 1')
    parser.add_argument('--capture-all', action='store_true', help='Capture all the layers.')
    parser.add_argument('--cluster-scores', action='store_true', help='Cluserting the importance scores rather than using actiation values.')
    parser.add_argument('--top-m-neurons', type=int, default=5, help='Number of top neurons to select.')
    
    args = parser.parse_args()
    print(args)
    
    return args

# Save the k-means model
def save_kmeans_model(kmeans, filename='kmeans_model.pkl'):
    joblib.dump(kmeans, filename)

# Load the k-means model
def load_kmeans_model(filename='kmeans_model.pkl'):
    return joblib.load(filename)

def get_model(model_name='vgg16'):
    module_name = []
    module = []
    if model_name == 'vgg16':
        model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == 'lenet':
        model = LeNet()
    else:
        model = Net()
    
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
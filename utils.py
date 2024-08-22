import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import os
import numpy as np


def load_CIFAR(batch_size=32, one_hot=True):
    # Default
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_MNIST(batch_size=32, one_hot=True, channel_first=True, train_all=False):
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    if channel_first:
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))  # If you want 3 channels
    transform = transforms.Compose(transform_list)

    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    
    if train_all:
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def save_model(model, model_name):
    torch.save(model.state_dict(), model_name + '.pth')
    print("Model saved as", model_name + '.pth')

def load_model(model_class, model_name):
    model = model_class()
    model.load_state_dict(torch.load(model_name + '.pth'))
    model.eval()  # Set the model to evaluation mode
    print("Model structure loaded from", model_name)
    return model

def get_layer_outputs(model, x, layers_to_skip=[]):
    outputs = []
    hooks = []

    def hook_fn(module, input, output):
        outputs.append(output)

    # Register hooks to the layers you want to inspect
    for i, layer in enumerate(model.children()):
        if i not in layers_to_skip:
            hooks.append(layer.register_forward_hook(hook_fn))

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return outputs

def calc_major_func_regions(model, train_inputs, skip=None):
    if skip is None:
        skip = []

    outputs = get_layer_outputs(model, train_inputs, layers_to_skip=skip)
    major_regions = []

    for output in outputs:
        output = output.mean(dim=tuple(range(1, output.ndim - 1)))
        major_regions.append((output.min(dim=0).values, output.max(dim=0).values))

    return major_regions

def get_layer_outputs_by_layer_name(model, test_input, skip=None):
    if skip is None:
        skip = []

    outputs = {}
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            outputs[name] = output
        return hook

    for name, layer in model.named_children():
        if name not in skip and 'input' not in name:
            hooks.append(layer.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        _ = model(test_input)

    for hook in hooks:
        hook.remove()

    return outputs

# TODO: Test not pass
def get_layer_inputs(model, test_input, skip=None):
    if skip is None:
        skip = []

    inputs = []
    previous_output = test_input

    for i, layer in enumerate(model.features):
        if i not in skip:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                weights = layer.weight.data
                biases = layer.bias.data if layer.bias is not None else None
                layer_inputs = []
                for input_index in range(len(test_input)):
                    if biases is not None:
                        input_for_layer = torch.add(torch.matmul(previous_output[input_index].view(-1), weights.view(weights.size(0), -1).t()), biases)
                    else:
                        input_for_layer = torch.matmul(previous_output[input_index].view(-1), weights.view(weights.size(0), -1).t())
                    layer_inputs.append(input_for_layer)

                inputs.append(layer_inputs)
            previous_output = layer(test_input)

    return inputs

def filter_correct_classifications(model, X, Y):
    X_corr = []
    Y_corr = []
    X_misc = []
    Y_misc = []
    
    model.eval()
    with torch.no_grad():
        outputs = model(X)
    
    # preds = torch.argmax(preds, dim=1)
    # Y_true = torch.argmax(Y, dim=1)

    _, preds = outputs.max(1)

    for idx, (x, y, pred) in enumerate(zip(X, Y, preds)):
        if pred == Y[idx]:
            X_corr.append(x)
            Y_corr.append(y)
        else:
            X_misc.append(x)
            Y_misc.append(y)

    return X_corr, Y_corr, X_misc, Y_misc

def filter_val_set(desired_class, X, Y):
    X_class = []
    Y_class = []

    for idx, (x, y) in enumerate(zip(X, Y)):
        if y == desired_class:
            X_class.append(x)
            Y_class.append(y)
            
    print(f"Validation set filtered for desired class: {desired_class}")
    return X_class, Y_class

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (torch.sqrt(torch.mean(x ** 2)) + 1e-5)

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_layer_outs(model, inputs, skip=[]):
    outputs = []
    hooks = []

    def hook_fn(module, input, output):
        outputs.append(output)

    # Register hooks to the layers except those specified in skip
    for index, layer in enumerate(model.features):  # Adjust if your model structure differs
        if index not in skip:
            hooks.append(layer.register_forward_hook(hook_fn))

    # Forward pass to capture the outputs
    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return outputs

def get_trainable_layers(model, parent_name=''):
    trainable_layers = []
    non_trainable_layers = []
    for name, layer in model.named_children():
        current_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            trainable_layers.append(current_name)
        else:
            non_trainable_layers.append(current_name)
            if isinstance(layer, nn.Sequential):
                trainable, non_trainable = get_trainable_layers(layer, current_name)
                trainable_layers.extend(trainable)
                non_trainable_layers.extend(non_trainable)
    
    return trainable_layers, non_trainable_layers
    
    # for idx, layer in enumerate(model.children()):
    #     try:
    #         for idx_, layer_module in enumerate(layer):
    #             layer_name = layer_module.__class__.__name__.lower()
    #             if 'input' not in layer_name and 'softmax' not in layer_name and \
    #                     'pred' not in layer_name and 'dropout' not in layer_name:
    #                 for param in layer.parameters():
    #                     if param.requires_grad:
    #                         trainable_layers[idx][idx_] = 1
    #                         break
    #     except:
    #         pass
    # return trainable_layers

def test_get_layers_out():
    model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    inputs = torch.randn(10, 3, 224, 224)
    layer_outputs = get_layer_outs(model, inputs)
    for i, output in enumerate(layer_outputs):
        print(f"Output of layer {i}: {output.shape}")

def test_get_trainable_layers():
    model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 512), 
                      nn.ReLU(), 
                      nn.Dropout(0.5),
                      nn.Linear(512, 10))
    trainable_layers = get_trainable_layers(model)
    print("------ get trainable layer PASS ------")

def test_filter():
    # Load a pre-trained VGG16 model
    model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 512), 
                      nn.ReLU(), 
                      nn.Dropout(0.5),
                      nn.Linear(512, 10))
    model.eval()

    train_loader, test_loader = load_CIFAR()

    X, Y = next(iter(train_loader))
    # Y_one_hot = F.one_hot(Y, num_classes=10).float()  # Convert labels to one-hot encoding

    # Filter correct and incorrect classifications
    X_corr, Y_corr, X_misc, Y_misc = filter_correct_classifications(model, X, Y)
    print(f"Correctly classified samples: {len(X_corr)}")
    print(f"Misclassified samples: {len(X_misc)}")
    print("------ Filter correct PASS ------")

    # Filter validation set for a desired class
    desired_class = 0
    X_class, Y_class = filter_val_set(desired_class, X, Y)
    print(f"Filtered samples for class {desired_class}: {len(X_class)}")
    print("------ Filter val PASS ------")

    # Normalize a tensor
    x_normalized = normalize(X[0])
    # print(f"Normalized tensor: {x_normalized}")
    print("------ Normalized PASS ------")


def test_get_layers():
    # Load a pre-trained VGG16 model
    vgg16 = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)

    # Example input image
    url = "https://cdn.mos.cms.futurecdn.net/ASHH5bDmsp6wnK6mEfZdcU-650-80.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # img = Image.open(requests.get(url, stream=True).raw)


    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)  # Create a mini-batch as expected by the model

    # Get layer outputs
    layer_outputs = get_layer_outputs(vgg16, batch_t, layers_to_skip=[])

    # Print the shapes of the outputs from each layer
    for i, output in enumerate(layer_outputs):
        print(f"Output of layer {i}: {output.shape}")
    
    print("------ get layer outputs PASS ------")

# TODO: Test not pass
def test_get_inputs():
    # Load a pre-trained VGG16 model
    vgg16 = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)

    # Example input image
    url = "https://cdn.mos.cms.futurecdn.net/ASHH5bDmsp6wnK6mEfZdcU-650-80.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)  # Create a mini-batch as expected by the model

    # Calculate major functional regions
    major_regions = calc_major_func_regions(vgg16, batch_t)
    print("Major Functional Regions:")
    for i, region in enumerate(major_regions):
        print(f"Layer {i}: Min {region[0].shape}, Max {region[1].shape}")

    # Get layer outputs by layer name
    layer_outputs = get_layer_outputs_by_layer_name(vgg16, batch_t)
    print("\nLayer Outputs by Layer Name:")
    for name, output in layer_outputs.items():
        print(f"{name}: {output.shape}")

    print("------ get layer by name PASS ------")

    # Get layer inputs
    # TODO: Issues here, test not pass
    # layer_inputs = get_layer_inputs(vgg16, batch_t)
    # print("\nLayer Inputs:")
    # for i, input_layer in enumerate(layer_inputs):
    #     print(f"Layer {i} Inputs: {len(input_layer)}")



def test_module():
    origin_module = ["load_CIFAR", 
        "load_MNIST", 
        "load_driving_data", 
        "data_generator",
        "preprocess_image",
        "deprocess_image",
        "load_dave_model",
        "load_model",
        "get_layer_outs_old",
        "get_layer_outs_new",
        "calc_major_func_regions",
        "get_layer_outputs_by_layer_name", 
        "get_layer_inputs", 
        "get_python_version", 
        "save_quantization", 
        "load_quantization", 
        "save_data", 
        "load_data",
        "save_layerwise_relevances",
        "load_layerwise_relevances",
        "save_perturbed_test",
        "load_perturbed_test",
        "save_perturbed_test_groups",
        "load_perturbed_test_groups",
        "create_experiment_dir",
        "save_classifications",
        "load_classifications",
        "save_totalR",
        "load_totalR",
        "save_layer_outs",
        "load_layer_outs",
        "filter_correct_classifications",
        "filter_val_set",
        "normalize",
        "get_trainable_layers",
        "weight_analysis",
        "percent_str",
        "generate_adversarial",
        "find_relevant_pixels",
        "save_relevant_pixels",
        "load_relevant_pixels",
        "create_dir"]

    print("Origin Module: ", len(origin_module))

    # Example usage:
    cifar_train_loader, cifar_test_loader = load_CIFAR()
    print("------ CIFAR10 Loading PASS ------")
    mnist_train_loader, mnist_test_loader = load_MNIST()
    print("------ MNIST Loading PASS ------")

    # test_get_inputs()
    # test_get_layers()
    # test_filter()
    # test_get_trainable_layers()
    test_get_layers_out()

### Testing for the modules
if __name__ == "__main__":
    test_module()

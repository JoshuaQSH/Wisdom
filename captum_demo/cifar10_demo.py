import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import LayerConductance, NeuronConductance
from captum.metrics import infidelity_perturb_func_decorator, infidelity

from sklearn.cluster import KMeans
import numpy as np


USE_PRETRAINED_MODEL = True

def prepare_data():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

# Define a new module that combines fc1 and relu3
class Fc1ReluModule(nn.Module):
    def __init__(self, net):
        super(Fc1ReluModule, self).__init__()
        # Extract the relevant layers from the original model
        self.fc1 = net.fc1
        self.relu3 = net.relu3

    def forward(self, x):
        # Apply fc1 and relu3 in sequence
        x = self.fc1(x)
        x = self.relu3(x)
        return x

def train(trainloader, testloader, classes, device='cuda:0'):
    net = Net()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    if USE_PRETRAINED_MODEL:
        print("Using existing trained model")
        net.load_state_dict(torch.load('models/cifar_torchvision.pt'))
    else:
        for epoch in tqdm(range(5)):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
        net = net.cpu()
        torch.save(net.state_dict(), 'models/cifar_torchvision.pt')
                
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('images/cifar10.png')
    # plt.show()
    
def attribute_image_features(algorithm, input, labels, ind, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=labels[ind],
                                              **kwargs
                                             )
    
    return tensor_attributions

def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float()
    breakpoint()
    return noise, inputs - noise
    

def test_one(testloader, classes, net):
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))
    ind = 3
    input = images[ind].unsqueeze(0)
    input.requires_grad = True
    
    net.eval()
    saliency = Saliency(net)
    grads = saliency.attribute(input, target=labels[ind].item())
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    ig = IntegratedGradients(net)
    attr_ig, delta = attribute_image_features(ig, input, labels, ind, baselines=input * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))
    
    ig = IntegratedGradients(net)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, labels, ind, baselines=input * 0, nt_type='smoothgrad_sq',
                                        nt_samples=100, stdevs=0.2)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    dl = DeepLift(net)
    attr_dl = attribute_image_features(dl, input, labels, ind, baselines=input * 0)
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    print('Original Image')
    print('Predicted:', classes[predicted[ind]], 
        ' Probability:', torch.max(F.softmax(outputs, 1)).item())

    original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

    plt_fig, plt_axis = viz.visualize_image_attr(None, original_image, 
                        method="original_image", title="Original Image")
    plt.savefig('images/origin.png')


    plt_fig, plt_axis = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                            show_colorbar=True, title="Overlayed Gradient Magnitudes")
    plt.savefig('images/gradientMagnitudes.png')

    plt_fig, plt_axis = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",
                            show_colorbar=True, title="Overlayed Integrated Gradients")
    plt.savefig('images/integratedGradient.png')

    plt_fig, plt_axis = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value", 
                                outlier_perc=10, show_colorbar=True, 
                                title="Overlayed Integrated Gradients \n with SmoothGrad Squared")
    plt.savefig('images/integratedSmoothGrad.png')

    plt_fig, plt_axis = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True, 
                            title="Overlayed DeepLift")
    plt.savefig('images/deepLift.png')

def get_neuron_conductance(net, images, labels, classes, neuron_index):
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    
    # target_layer = next(net.children())
    target_layer = net.fc1
    neuron_cond = NeuronConductance(net, target_layer)
    attribution = neuron_cond.attribute(images, neuron_selector=neuron_index, target=labels)
    
    return attribution

def get_layer_conductance(net, images, labels, classes):
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    
    # target_layer = next(net.children())
    target_layer = net.fc1
    neuron_cond = LayerConductance(net, target_layer)
    attribution = neuron_cond.attribute(images, target=labels)
    
    return attribution

def cluster_important_neurons(importance_scores, n_clusters=5):
    # Reshape for KMeans (needs 2D input)
    scores = importance_scores.cpu().detach().numpy().reshape(-1, 1)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(scores)

    return clusters, kmeans.cluster_centers_

def compute_idc_coverage(testloader, model, clusters, centroids, classes, layer_number=4):
    model.eval()
    covered_combinations = set()

    for inputs, labels in testloader:
        for i in range(inputs.size(0)):  # Loop over batch
            
            # Get importance scores for the specific input
            importance_scores = get_layer_conductance(net, inputs, labels, classes)
            layer_scores = importance_scores[i]
            
            # Map the layer's importance scores to clusters
            layer_clusters = assign_clusters(layer_scores, centroids)
            
            # Add the clusters combination (tuple) to the covered_combinations set
            covered_combinations.add(tuple(layer_clusters))

    # Total possible combinations
    total_combinations = np.prod([len(centroids)] * len(clusters))
    
    # Calculate IDC coverage
    coverage = len(covered_combinations) / total_combinations
    return coverage

def assign_clusters(scores, centroids):
    """Assigns each score to its closest centroid."""
    clusters = []
    for score in scores:
        distances = np.linalg.norm(centroids - score.detach().numpy(), axis=1)  # Calculate distances from centroids
        clusters.append(np.argmin(distances))  # Assign to closest cluster
    return clusters

def get_layer_outs(model, inputs, skip=[]):
        outputs = []
        hooks = []

        def hook_fn(module, input, output):
            outputs.append(output)
        
        # Register hooks to the layers except those specified in skip
        for index, layer in enumerate(model.children()):
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

def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def infidelity_metric(net, perturb_fn, inputs, attribution, layer_index=6):
    
    from captum.attr import IntegratedGradients
    
    train_layer_outs = get_layer_outs(net, inputs)
    # activation = {}
    # net.fc1.register_forward_hook(get_activation('fc1', activation))
    # net.relu3.register_forward_hook(get_activation('relu3', activation))
    new_inputs = train_layer_outs[layer_index - 1].view(-1, 16 * 5 * 5)
    
    # new_inputs = train_layer_outs[layer_index+1]
    model_new = Fc1ReluModule(net)
    ig = IntegratedGradients(model_new)
    attribution_new = ig.attribute(new_inputs, target=0)
    infid = infidelity(model_new, perturb_fn, new_inputs, attribution_new)
    
    return infid

if __name__ == '__main__':
    trainloader, testloader, classes = prepare_data()
    net = Net()
    train(trainloader, testloader, classes)
    net.load_state_dict(torch.load('models/cifar_torchvision.pt'))
    
    # A demo test example
    # test_one(testloader, classes, net)
    
    # Further example, get each neruone's attribution
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    attribution = get_layer_conductance(net, images, labels, classes)
    # attribution = get_neuron_conductance(net, images, labels, classes, 2)

    print(attribution.shape)
    clusters, centroids = cluster_important_neurons(attribution)
    # print("Clusters:", clusters)
    print("Cluster Centroids:", centroids)
    
    # Compute IDC coverage
    # idc_coverage = compute_idc_coverage(testloader, net, clusters, centroids, classes)
    # print(f"IDC Coverage: {idc_coverage * 100:.2f}%")
    
    # Infidelity metric
    infid = infidelity_metric(net, perturb_fn, images, attribution)
    print(f"Infidelity: {infid:.2f}")
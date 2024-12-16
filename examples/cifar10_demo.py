import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import LayerConductance, NeuronConductance
from captum.metrics import infidelity_perturb_func_decorator, infidelity

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import json
import os

from matplotlib.colors import LinearSegmentedColormap

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
    
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 3 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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

def save_importance_scores(importance_scores, mean_importance, filename, class_label):
    scores = importance_scores.cpu().detach().numpy().tolist()
    mean_scores = mean_importance.cpu().detach().numpy().tolist()
    data = {
        "class_label": class_label,
        "importance_scores": scores,
        "mean_importance": mean_scores
    }

    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Importance scores saved to {filename}")

def load_importance_scores(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return torch.tensor(data["importance_scores"]), torch.tensor(data["mean_importance"]), data["class_label"]

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

def train(net, trainloader, device='cuda:0', model_path='models/cifar_torchvision.pt', pretrained=False, epochs=5):
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    if pretrained:
        print("Using existing trained model")
        net.load_state_dict(torch.load(model_path))
    else:
        for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
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
        torch.save(net.state_dict(), model_path)
                
def imshow(img, img_path='images/cifar10.png'):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.savefig('images/cifar10.png')
    plt.savefig(img_path)

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
    return noise, inputs - noise

def test_one(images, labels, classes, net, device, get_top=100):
    # dataiter = iter(testloader)
    # images, labels = next(dataiter)

    print("Getting.. ", classes[labels[0]])
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    net = net.cpu()
    # images = images.to(device)
    if get_top:
        images = images[:get_top]
        labels = labels[:get_top]
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

def get_layer_conductance(net, images, labels, classes, layer_name='fc1', top_m_images=100):
    # print images
    # imshow(torchvision.utils.make_grid(images))
    net = net.cpu()
    
    if top_m_images:
        images = images[:top_m_images]
        labels = labels[:top_m_images]
    
    net_layer = getattr(net, layer_name)
    print("GroundTruth: {}, Model Layer: {}".format(classes[labels[0]], net_layer[19]))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    
    # target_layer = next(net.children())
    # target_layer = net.fc1
    target_layer = net_layer[19]
    neuron_cond = LayerConductance(net, target_layer)
    attribution = neuron_cond.attribute(images, target=labels)
    
    return attribution, torch.mean(attribution, dim=0)

# Select the top M important neurons based on their importance scores.
def select_top_neurons(importance_scores, top_m_neurons=5):
    _, indices = torch.topk(importance_scores, top_m_neurons)
    return indices

# TODO: We choose the the cluster number based on the silhouette score, could be customized in the future
def find_optimal_clusters(importance_scores, max_k=20):
    scores = []
    importance_scores_np = importance_scores.cpu().detach().numpy().reshape(-1, 1)
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(importance_scores_np)
        score = silhouette_score(importance_scores_np, labels)
        scores.append((k, score))
    # Select k with the highest silhouette score
    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"Optimal number of clusters: {best_k}")
    
    return best_k

# Assign Test inputs to Clusters (activation values)
# TODO: Implement this function
def assign_clusters_to_act_importance_scores(importance_scores, kmeans_model, top_m_neurons=None):
    if top_m_neurons:
        # Get indices of top m neurons
        _, indices = select_top_neurons(importance_scores, top_m_neurons)
    else:
        # Cluster based on all the neurons
        pass
    pass

# Assign Test inputs to Clusters
def assign_clusters_to_importance_scores(importance_scores, kmeans_model):
    importance_scores_np = importance_scores.cpu().detach().numpy().reshape(-1, 1)
    cluster_labels = kmeans_model.predict(importance_scores_np)
    return cluster_labels

# Track Combinations of Clusters Activated
# mode = ['scores', 'activations']
def compute_idc_coverage(model, inputs_images, labels, classes, kmeans_model, top_m_neurons=None, mode='scores'):
    model.eval()
    covered_combinations = set()
    
    if mode == 'scores':
        _, layer_importance_scores = get_layer_conductance(model, inputs_images, labels, classes, layer_name='fc1')
        if top_m_neurons:
            # Get indices of top m neurons
            _, indices = select_top_neurons(layer_importance_scores, top_m_neurons)
            # Select importance scores for top m neurons
            selected_scores = layer_importance_scores[indices]
            # Assign clusters
            cluster_labels = assign_clusters_to_importance_scores(selected_scores, kmeans_model)
        else:
            # Assign clusters to all neurons
            cluster_labels = assign_clusters_to_importance_scores(layer_importance_scores, kmeans_model)
            indices = torch.arange(len(layer_importance_scores))
        
        # Record the combination (as a tuple) of cluster labels
        combination = tuple(cluster_labels.tolist())
        covered_combinations.add(combination)
        
        # Compute total possible combinations
        n_clusters = kmeans_model.n_clusters
        n_neurons = top_m_neurons if top_m_neurons else len(layer_importance_scores)
        total_possible_combinations = n_clusters ** n_neurons
        
        idc_value = len(covered_combinations) / total_possible_combinations
    
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return idc_value, covered_combinations, total_possible_combinations

def cluster_importance_scores(importance_scores, n_clusters): 
    importance_scores_np = importance_scores.cpu().detach().numpy().reshape(-1, 1) 
    kmeans = KMeans(n_clusters=n_clusters, random_state=0) 
    cluster_labels = kmeans.fit_predict(importance_scores_np) 
    return cluster_labels, kmeans

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

def get_model(model_name='vgg16'):
    if model_name == 'vgg16':
        model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == 'lenet':
        model = LeNet()
    else:
        model = Net()
    
    return model

if __name__ == '__main__':
    
    # hyperparameters
    model_path = os.getenv("HOME") + '/torch-deepimportance/captum_demo/models/lenet_cifar10.pt'
    # model_path = os.getenv("HOME") + '/torch-deepimportance/captum_demo/models/cifar_torchvision.pt'

    data_path = '/data/shenghao/dataset/'
    importance_file = 'plane_lenet_importance.json'
    # importance_file = 'plane_importance.json'

    is_cifar10 = True
    model_name = ['vgg16', 'custom', 'lenet']
    test_image_name_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    test_image_name = test_image_name_list[0]
    epochs = 5
    
    trainloader, testloader, classes = prepare_data_cifa(data_path=data_path, cifar10=is_cifar10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #  Choose the model to use
    net = get_model(model_name=model_name[2])
    # Train the model    
    train(net, trainloader, device=device, model_path=model_path, pretrained=True, epochs=epochs)

    # net.load_state_dict(torch.load('models/cifar_torchvision.pt'))
    net.load_state_dict(torch.load(model_path))
    
    # Further example, get each neruone's attribution
    images, labels = get_class_data(trainloader, classes, test_image_name)
    
    # A demo test example
    test_one(images, labels, classes, net, device, 100)
    
    if os.path.exists(importance_file):
        attribution, mean_attribution, labels = load_importance_scores(importance_file)
        test_images, test_labels = get_class_data(testloader, classes, test_image_name)
    else:
        attribution, mean_attribution = get_layer_conductance(net, images, labels, classes, layer_name='features')
        save_importance_scores(attribution, mean_attribution, importance_file, test_image_name)
    
    # attribution = get_neuron_conductance(net, images, labels, classes, 2)
    
    optimal_k = find_optimal_clusters(mean_attribution)
    cluster_labels, kmeans_model = cluster_importance_scores(mean_attribution, optimal_k)
    
    # clusters, centroids = cluster_important_neurons_old(mean_attribution)
    # print("Clusters:", clusters)
    # print("Cluster Centroids:", centroids)
    
    ### Compute IDC coverage
    idc_value, covered_combinations, total_combinations = compute_idc_coverage(
        model=net,
        inputs_images=test_images,
        labels=test_labels,
        classes=classes,
        kmeans_model=kmeans_model,
        top_m_neurons=2  # Adjust as needed
    )
    print(f"IDC Coverage: {idc_value * 100:.6f}%")
    
    # Infidelity metric
    # infid = infidelity_metric(net, perturb_fn, images, attribution)
    # print(f"Infidelity: {infid:.2f}")
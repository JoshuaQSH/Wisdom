import os
import json

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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

from utils import prepare_data_cifa, get_class_data, parse_args, get_model, load_kmeans_model, save_kmeans_model
from model_hub import LeNet, Net
from matplotlib.colors import LinearSegmentedColormap

def load_importance_scores(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return torch.tensor(data["importance_scores"]), torch.tensor(data["mean_importance"]), data["class_label"]

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

def get_layer_conductance(model, images, labels, classes, layer_name='fc1', top_m_images=-1):
    model = model.cpu()
    
    if top_m_images != -1:
        images = images[:top_m_images]
        labels = labels[:top_m_images]
    net_layer = getattr(model, layer_name)
    print("GroundTruth: {}, Model Layer: {}".format(classes[labels[0]], net_layer))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    neuron_cond = LayerConductance(model, net_layer)
    attribution = neuron_cond.attribute(images, target=labels)
    
    return attribution, torch.mean(attribution, dim=0)

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
            # TODO: Conv by default will select all the neurons
            print("Selecting all the neurons (Conv2D layer).")
            return None

def visualize_activation(activation_values, selected_activations, layer_name, threshold=0, mode='fc'):
    saved_file = f'./images/mean_activation_values_{layer_name}.pdf'
    if mode == 'fc':
        mean_activation = activation_values.mean(dim=0)
        mean_selected_activation = selected_activations.mean(dim=0)
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(activation_values[1])), mean_activation.numpy(), marker='o')
        plt.plot(range(len(selected_activations[1])), mean_selected_activation.numpy(), marker='p')
        plt.xlabel('Neuron Index')
        plt.ylabel('Mean Activation Value')
        plt.title('Mean Activation Values of Neurons')
        plt.grid(True)
        plt.legend(['All Neurons', 'Selected Neurons'])
        plt.savefig(saved_file, dpi=1500)
        print("Mean Activation Values plotted, saved to {}".format(saved_file))
        # plt.show()
    elif mode == 'conv':
        mean_activation = torch.mean(selected_activations, dim=[2, 3]).mean(dim=0)
        plt.plot(range(len(mean_activation)), mean_activation.numpy(), marker='o')
        plt.xlabel('Neuron Index')
        plt.ylabel('Mean Activation Value')
        plt.title('Mean Activation Values of Neurons')
        plt.grid(True)
        plt.savefig(saved_file, dpi=1500)
        print("Mean Activation Values plotted, saved to {}".format(saved_file))
        # plt.show()
    else:
        raise ValueError(f"Invalid mode: {mode}")

def get_activation_values_for_neurons(model, inputs, labels, important_neuron_indices, layer_name='fc1', visualize=False):
    
    activation_values = []
    print("Getting the Class: {}".format(labels[0]))
    
    # Define a forward hook to capture the activations
    def hook_fn(module, input, output):
        activation_values.append(output.detach())
    
    # Register the hook on the specified layer
    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    
    if handle is None:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    
    # Remove the hook
    handle.remove()
    
    # Concatenate all activation values into one tensor
    activation_values = torch.cat(activation_values, dim=0)
    if important_neuron_indices is None:
        selected_activations = activation_values
    elif type(important_neuron_indices) == torch.Tensor and important_neuron_indices.shape[0] > 0:
        # Select only the activations for the important neurons
        selected_activations = activation_values[:, important_neuron_indices]
    else:
        raise ValueError(f"Invalid important_neuron_indices: {important_neuron_indices}")
    
    if visualize:
        visualize_activation(activation_values, selected_activations, layer_name, threshold=0, mode=layer_name[:-1])
        
    return activation_values, selected_activations

# TODO: We choose the the cluster number based on the silhouette score, could be customized in the future
# score could be importance scores or activation values, using silhouette score to find the optimal cluster number
def find_optimal_clusters(scores, max_k=100):
    scores_silhouette = []
    scores_np = scores.cpu().detach().numpy().reshape(-1, 1)
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(scores_np)
        scores_t = silhouette_score(scores_np, labels)
        scores_silhouette.append((k, scores_t))
    
    # Select k with the highest silhouette score
    best_k = max(scores_silhouette, key=lambda x: x[1])[0]
    print(f"Optimal number of clusters: {best_k}")
    
    return best_k

# Option - 1: This is used for clustering the importance scores
def cluster_importance_scores(importance_scores, n_clusters): 
    importance_scores_np = importance_scores.cpu().detach().numpy().reshape(-1, 1) 
    kmeans = KMeans(n_clusters=n_clusters, random_state=0) 
    cluster_labels = kmeans.fit_predict(importance_scores_np)
    save_kmeans_model(kmeans, 'kmeans_model.pkl')
    return cluster_labels, kmeans

# Option - 2: This is used for clustering the activation values
def cluster_activation_values(activation_values, n_clusters):
    activation_values_np = activation_values.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(activation_values_np)
    save_kmeans_model(kmeans, 'kmeans_model.pkl')
    return cluster_labels, kmeans

# Assign Test inputs to Clusters
def assign_clusters_to_importance_scores(importance_scores, kmeans_model):
    importance_scores_np = importance_scores.cpu().detach().numpy().reshape(-1, 1)
    cluster_labels = kmeans_model.predict(importance_scores_np)
    return cluster_labels

def assign_clusters_to_activation_values(activation_values, kmeans_model):
    activation_values_np = activation_values.cpu().numpy()
    cluster_labels = kmeans_model.predict(activation_values_np)
    return cluster_labels


def compute_idc_test(model, inputs_images, labels, kmeans, classes, layer_name, top_m_neurons=-1, n_clusters = 5):
    kmeans_model_path = 'kmeans_model.pkl'
    _, layer_importance_scores = get_layer_conductance(model, inputs_images, labels, classes, layer_name=layer_name)
    indices = select_top_neurons(layer_importance_scores, top_m_neurons)
    activation_values, selected_activations = get_activation_values_for_neurons(model, 
                                                                                inputs_images,
                                                                                labels, 
                                                                                indices, 
                                                                                layer_name,
                                                                                False)
    activation_values_np = activation_values.cpu().numpy()
    cluster_labels = kmeans.predict(activation_values_np)
    unique_clusters = np.unique(cluster_labels)
    coverage_rate = len(unique_clusters) / n_clusters
    print(f"IDC Score: {coverage_rate}")

# Track Combinations of Clusters Activated
# mode = ['scores', 'activations']
def compute_idc_coverage(model, inputs_images, labels, classes, kmeans_model, layer_name, top_m_neurons=None, mode='activations'):
    model.eval()
    covered_combinations = set()
    
    _, layer_importance_scores = get_layer_conductance(model, inputs_images, labels, classes, layer_name=layer_name)
    if top_m_neurons:
        # Get indices of top m neurons
        indices = select_top_neurons(layer_importance_scores, top_m_neurons)
        # Assign clusters
        if mode == 'scores':
            # Select importance scores for top m neurons
            selected_scores = layer_importance_scores[indices]
            # Assign clusters
            cluster_labels = assign_clusters_to_importance_scores(selected_scores, kmeans_model)
        elif mode == 'activations':
            activation_values, selected_activations = get_activation_values_for_neurons(model, 
                                                                                        inputs_images,
                                                                                        labels, 
                                                                                        indices, 
                                                                                        layer_name,
                                                                                        False)
            cluster_labels = assign_clusters_to_activation_values(selected_scores, kmeans_model)
    
    elif top_m_neurons is None:
        if mode == 'scores':
            # Assign clusters to all neurons
            cluster_labels = assign_clusters_to_importance_scores(layer_importance_scores, kmeans_model)
            indices = torch.arange(len(layer_importance_scores))            
        elif mode == 'activations':
            activation_values, selected_activations = get_activation_values_for_neurons(model, 
                                                                                        inputs_images,
                                                                                        labels, 
                                                                                        None, 
                                                                                        layer_name,
                                                                                        False)
            cluster_labels = assign_clusters_to_activation_values(selected_activations, kmeans_model)
    else:
        raise ValueError(f"Invalid mode: {mode}")
           
    # Record the combination (as a tuple) of cluster labels
    combination = tuple(cluster_labels.tolist())
    covered_combinations.add(combination)
        
    # Compute total possible combinations
    n_clusters = kmeans_model.n_clusters
    n_neurons = top_m_neurons if top_m_neurons else len(layer_importance_scores)
    total_possible_combinations = n_clusters ** n_neurons
        
    idc_value = len(covered_combinations) / total_possible_combinations
    
    return idc_value, covered_combinations, total_possible_combinations

if __name__ == '__main__':
    args = parse_args()
    
    if args.model_path != 'None':
        model_path = args.model_path
    else:
        model_path = os.getenv("HOME") + '/torch-deepimportance/captum_demo/models/'
    model_path += args.saved_model
    trainloader, testloader, classes = prepare_data_cifa(data_path=args.data_path, cifar10=args.is_cifar10)
    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    
    model, module_name, module = get_model(model_name=args.model)    
    model.load_state_dict(torch.load(model_path))
    
    #  Get the specific class data
    images, labels = get_class_data(trainloader, classes, args.test_image)
    
    # Get the importance scores - LRP
    if os.path.exists(args.importance_file):
        attribution, mean_attribution, labels = load_importance_scores(args.importance_file)
        test_images, test_labels = get_class_data(testloader, classes, args.test_image)
    else:
        if args.capture_all:
            for name in module_name[1:]:
                attribution, mean_attribution = get_layer_conductance(model, images, labels, classes, layer_name=name, top_m_images=-1)
                filename = args.importance_file.replace('.json', f'_{name}.json')
                save_importance_scores(attribution, mean_attribution, filename, args.test_image)
                print("{} Saved".format(filename))
        else:
            attribution, mean_attribution = get_layer_conductance(model, images, labels, classes, layer_name=module_name[args.layer_index], top_m_images=-1)
            save_importance_scores(attribution, mean_attribution, args.importance_file, args.test_image)

    # Obtain the important neuron indices 
    important_neuron_indices = select_top_neurons(mean_attribution, args.top_m_neurons)
    activation_values, selected_activations = get_activation_values_for_neurons(model, 
                                                                                images, 
                                                                                labels, 
                                                                                important_neuron_indices, 
                                                                                module_name[args.layer_index],
                                                                                args.viz)
        
    # Clustering based on the importance scores or activation values
    # optimal_k_importance = find_optimal_clusters(mean_attribution)
    # optimal_k = find_optimal_clusters(activation_values)
    optimal_k = 5
    
    if args.cluster_scores:
        ### Option - 1: Cluster the importance scores
        try:
            kmeans_model = load_kmeans_model('kmeans_model.pkl')
        except FileNotFoundError:
            cluster_labels, kmeans_model = cluster_importance_scores(mean_attribution, optimal_k)
        print("Importance scores clustered.")
    else:
        ### Option - 2: Cluster the activation values
        try:
            kmeans_model = load_kmeans_model('kmeans_model.pkl')
        except FileNotFoundError:
            cluster_labels, kmeans_model = cluster_activation_values(selected_activations, optimal_k)
        print("Activation values clustered.")

    ### Compute IDC coverage
    compute_idc_test(model, test_images, test_labels, kmeans_model, classes, module_name[args.layer_index], top_m_neurons=args.top_m_neurons, n_clusters=optimal_k)
    # idc_value, covered_combinations, total_combinations = compute_idc_coverage(
    #     model=model,
    #     inputs_images=test_images,
    #     labels=test_labels,
    #     classes=classes,
    #     kmeans_model=kmeans_model,
    #     layer_name=module_name[args.layer_index],
    #     top_m_neurons=2,
    #     mode='activations'
    # )
    # print(f"IDC Coverage: {idc_value * 100:.6f}%")
    
    ### Infidelity metric
    # infid = infidelity_metric(net, perturb_fn, images, attribution)
    # print(f"Infidelity: {infid:.2f}")
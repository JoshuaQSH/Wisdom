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

# attribution_method = ['lrp', 'ig', 'saliency', 'deeplift']
def get_layer_conductance(model, images, labels, classes, layer_name='fc1', top_m_images=-1, attribution_method='lrp'):
    model = model.cpu()
    
    if top_m_images != -1:
        images = images[:top_m_images]
        labels = labels[:top_m_images]
        
    net_layer = getattr(model, layer_name)
    print("GroundTruth: {}, Model Layer: {}".format(classes[labels[0]], net_layer))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    if attribution_method == 'lrp':
        neuron_cond = LayerConductance(model, net_layer)
        attribution = neuron_cond.attribute(images, target=labels)
    else:
        raise ValueError(f"Invalid attribution method: {attribution}")
    
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
    print("Getting the Class: {}".format(labels))
    
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
def cluster_importance_scores(importance_scores, n_clusters, layer_name='fc1'): 
    if layer_name[:-1] == 'fc':
        importance_scores_np = importance_scores.cpu().detach().numpy().reshape(-1, 1) 
        kmeans = KMeans(n_clusters=n_clusters, random_state=0) 
        cluster_labels = kmeans.fit_predict(importance_scores_np)
    elif layer_name[:-1] == 'conv':
        importance_scores = torch.mean(importance_scores, dim=[2, 3])
        importance_scores_np = importance_scores.cpu().detach().numpy().reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(importance_scores_np)
    else:
        raise ValueError(f"Invalid layer name: {layer_name}")
        return None, None
    
    print("The cluster labels: {}, etc.".format(cluster_labels[:10]))
    save_kmeans_model(kmeans, 'kmeans_impo_{}.pkl'.format(layer_name))    
    
    return cluster_labels, kmeans

# Option - 2: This is used for clustering the activation values
def cluster_activation_values(activation_values, n_clusters, layer_name='fc1'):
    
    if layer_name[:-1] == 'fc':
        n_neurons = activation_values.shape[1]
        activation_values_np = activation_values.cpu().detach().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans_comb = [KMeans(n_clusters=n_clusters).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)) for i in range(n_neurons)]
        cluster_labels = kmeans.fit_predict(activation_values_np)
        
    elif layer_name[:-1] == 'conv':
        activation_values = torch.mean(activation_values, dim=[2, 3])
        n_neurons = activation_values.shape[1]
        activation_values_np = activation_values.cpu().detach().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans_comb = [KMeans(n_clusters=2).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)) for i in range(n_neurons)]
        cluster_labels = kmeans.fit_predict(activation_values_np)
    else:
        raise ValueError(f"Invalid layer name: {layer_name}")
        return None, None, None
        
    print("The cluster labels: {}, etc., with number of clusters: {}".format(cluster_labels[:10], n_clusters))
    save_kmeans_model(kmeans, 'kmeans_acti_{}.pkl'.format(layer_name))
    
    return cluster_labels, kmeans, kmeans_comb

# Assign Test inputs to Clusters
def assign_clusters_to_importance_scores(importance_scores, kmeans_model):
    importance_scores_np = importance_scores.cpu().detach().numpy().reshape(-1, 1)
    cluster_labels = kmeans_model.predict(importance_scores_np)
    return cluster_labels

def assign_clusters_to_activation_values(activation_values, kmeans_model):
    activation_values_np = activation_values.cpu().numpy()
    cluster_labels = kmeans_model.predict(activation_values_np)
    return cluster_labels

# A for loop to assign clusters to all the neurons
def assign_clusters(activations, kmeans_models):
    n_samples, n_neurons = activations.shape
    cluster_assignments = []

    # Loop over each test sample
    for i in range(n_samples):
        # Get the activations for this sample
        sample_activations = activations[i].cpu().numpy()
        sample_clusters = []
        
        # Loop over each neuron and assign it to a cluster
        for neuron_idx in range(n_neurons):
            cluster = kmeans_models[neuron_idx].predict(sample_activations[neuron_idx].reshape(-1, 1))
            sample_clusters.append(cluster[0])  # Append the cluster ID
        
        # Convert to tuple for uniqueness
        cluster_assignments.append(tuple(sample_clusters))  
    return cluster_assignments


def compute_idc_test(model, inputs_images, labels, kmeans, classes, layer_name, top_m_neurons=-1, n_clusters = 5, attribution_method='lrp'):
    
    _, layer_importance_scores = get_layer_conductance(model, inputs_images, labels, classes, 
                                                       layer_name=layer_name, attribution_method=attribution_method)
    indices = select_top_neurons(layer_importance_scores, top_m_neurons)
    activation_values, selected_activations = get_activation_values_for_neurons(model, 
                                                                                inputs_images,
                                                                                classes[int(labels[0])], 
                                                                                indices, 
                                                                                layer_name,
                                                                                False)
    
    if layer_name[:-1] == 'conv':
        activation_values = torch.mean(activation_values, dim=[2, 3])
    else:
        activation_values = selected_activations
        
    activation_values_np = activation_values.cpu().numpy()
    cluster_labels = assign_clusters(kmeans_models=kmeans, activations=activation_values)
    # cluster_labels = kmeans.predict(activation_values_np)
    
    unique_clusters = set(cluster_labels)    
    # unique_clusters = np.unique(cluster_labels)
    
    total_combination = pow(n_clusters, activation_values_np.shape[1])
    if activation_values.shape[0] > total_combination:
        max_coverage = 1
    else:
        max_coverage = activation_values.shape[0] / total_combination
    
    # n_cluster -> total_combination
    coverage_rate = len(unique_clusters) / total_combination
    print(f"Total INCC combinations: {total_combination}")
    print(f"Max Coverage (the best we can achieve): {max_coverage * 100:.6f}%")
    print(f"IDC Coverage: {coverage_rate * 100:.6f}%")

def compute_incc_centroids(important_neurons_clusters):
    centroids = {}
    for neuron, clusters in important_neurons_clusters.items():
        centroids[neuron] = [cluster.cluster_centers_ for cluster in clusters]
    return centroids

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
    if args.use_silhouette:
        optimal_k = find_optimal_clusters(activation_values)
    else:
        optimal_k = args.n_clusters
    
    if args.cluster_scores:
        ### Option - 1: Cluster the importance scores
        try:
            kmeans_model = load_kmeans_model('kmeans_model_{}.pkl'.format(module_name[args.layer_index]))
            print('kmeans_model_{}.pkl Loaded!'.format(module_name[args.layer_index]))
        except FileNotFoundError:
            cluster_labels, kmeans_model = cluster_importance_scores(mean_attribution, optimal_k)
        print("Importance scores clustered.")
    else:
        ### Option - 2: Cluster the activation values
        cluster_labels, kmeans_model, kmeans_comb = cluster_activation_values(selected_activations, optimal_k, module_name[args.layer_index])
        # try:
        #     kmeans_model = load_kmeans_model('kmeans_model_{}.pkl'.format(module_name[args.layer_index]))
        # except FileNotFoundError:
        #     cluster_labels, kmeans_model = cluster_activation_values(selected_activations, optimal_k, module_name[args.layer_index])
        print("Activation values clustered.")

    ### Compute IDC coverage
    compute_idc_test(model, 
                     test_images, 
                     test_labels, 
                     kmeans_comb, # kmeans_model
                     classes, 
                     module_name[args.layer_index], 
                     top_m_neurons=args.top_m_neurons, 
                     n_clusters=optimal_k,
                     attribution_method=args.attr)
    
    ### Infidelity metric
    # infid = infidelity_metric(net, perturb_fn, images, attribution)
    # print(f"Infidelity: {infid:.2f}")
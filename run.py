import os
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

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
from captum.attr import NeuronConductance
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerActivation, InternalInfluence, LayerGradientXActivation, LayerGradCam, LayerDeepLift, LayerDeepLiftShap, LayerGradientShap, LayerIntegratedGradients, LayerFeatureAblation, LayerLRP
from captum.metrics import infidelity_perturb_func_decorator, infidelity

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from utils import load_CIFAR, load_MNIST, load_ImageNet, get_class_data, parse_args, get_model, load_kmeans_model, save_kmeans_model, visualize_idc_scores

# Add src directory to sys.path
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))


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

def get_layer_conductance(model, images, labels, classes, layer_name='fc1', top_m_images=-1, attribution_method='lrp'):
    model = model.cpu()
    
    if top_m_images != -1:
        images = images[:top_m_images]
        labels = labels[:top_m_images]
        
    net_layer = getattr(model, layer_name)
    print("GroundTruth: {}, Model Layer: {}".format(classes[labels[0]], net_layer))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # ['LayerConductance', 'LayerActivation', 'InternalInfluence', 
    # 'LayerGradientXActivation', 'LayerGradCam', 'LayerDeepLift', 
    # 'LayerDeepLiftShap', 'LayerGradientShap', 'LayerIntegratedGradients', 
    # 'LayerFeatureAblation', 'LayerLRP']
    if attribution_method == 'lc':
        print("Running with LayerConductance")
        neuron_cond = LayerConductance(model, net_layer)
        attribution = neuron_cond.attribute(images, target=labels)
    elif attribution_method == 'la':
        print("Running with LayerActivation")
        neuron_cond = LayerActivation(model, net_layer)
        attribution = neuron_cond.attribute(images)
    elif attribution_method == 'ii':
        print("Running with InternalInfluence")
        neuron_cond = InternalInfluence(model, net_layer)
        attribution = neuron_cond.attribute(images, target=labels)
    elif attribution_method == 'lgxa':
        print("Running with LayerGradientXActivation")
        neuron_cond = LayerGradientXActivation(model, net_layer)
        attribution = neuron_cond.attribute(images, target=labels)
    elif attribution_method == 'lgc':
        print("Running with LayerGradCam")
        neuron_cond = LayerGradCam(model, net_layer)
        attribution = neuron_cond.attribute(images, target=labels)
    elif attribution_method == 'ldl':
        print("Running with LayerDeepLift")
        neuron_cond = LayerDeepLift(model, net_layer)
        attribution = neuron_cond.attribute(images, baselines=torch.zeros_like(images), target=labels)
    elif attribution_method == 'ldls':
        print("Running with LayerDeepLiftShap")
        neuron_cond = LayerDeepLiftShap(model, net_layer)
        attribution = neuron_cond.attribute(images,  baselines=torch.zeros_like(images), target=labels)
    elif attribution_method == 'lgs':
        print("Running with LayerGradientShap")
        neuron_cond = LayerGradientShap(model, net_layer)
        attribution = neuron_cond.attribute(images, baselines=torch.zeros_like(images), target=labels)
    elif attribution_method == 'lig':
        print("Running with LayerIntegratedGradients")
        neuron_cond = LayerIntegratedGradients(model, net_layer)
        attribution = neuron_cond.attribute(images, target=labels)
    elif attribution_method == 'lfa':
        print("Running with LayerFeatureAblation")
        neuron_cond = LayerFeatureAblation(model, net_layer)
        attribution = neuron_cond.attribute(images, target=labels)
    elif attribution_method == 'lrp':
        print("Running with LayerLRP")
        neuron_cond = LayerLRP(model, net_layer)
        attribution = neuron_cond.attribute(images, target=labels)
    else:
        raise ValueError(f"Invalid attribution method: {attribution}")
    
    return attribution, torch.mean(attribution, dim=0)

def evaluate_attribution_methods(model, inputs_images, labels, kmeans, classes, layer_name, attribution_methods, top_m_neurons=-1, n_clusters=5):
    idc_scores = {}
    for method in attribution_methods:
        print(f"Evaluating method: {method}")
        # Compute IDC for the given attribution method
        _, idc_score = compute_idc_test(model, inputs_images, labels, kmeans, classes, layer_name,
                                        top_m_neurons=top_m_neurons, n_clusters=n_clusters,
                                        attribution_method=method)
        
        print(f"IDC score for {method}: {idc_score}")
        # Store the IDC score for this method
        idc_scores[method] = idc_score
    
    visualize_idc_scores(idc_scores)
    return idc_scores

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
            mean_attribution = torch.mean(importance_scores, dim=[1, 2])
            if mean_attribution.shape[0] < top_m_neurons:
                print("Selecting all the neurons (Conv2D layer).")
                return None
            else:
                _, indices = torch.topk(mean_attribution, top_m_neurons)
                return indices

# Select the top m neurons from all layers
def select_top_neurons_all(importance_scores_dict, top_m_neurons=5):
    # Create a list to store tuples of (layer_name, importance_value, index_in_layer)
    flattened_importance = []

    for layer_name, importance_scores in importance_scores_dict.items():
        if top_m_neurons == -1:
            print(f"Selecting all neurons from layer {layer_name}.")
            if importance_scores.dim() == 1:  # FC layer
                indices = torch.arange(importance_scores.shape[0])
                flattened_importance.extend([(layer_name, score.item(), idx.item()) for idx, score in enumerate(importance_scores)])
            else:  # Conv2D layer
                mean_attribution = torch.mean(importance_scores, dim=[1, 2])
                indices = torch.arange(mean_attribution.shape[0])
                flattened_importance.extend([(layer_name, score.item(), idx.item()) for idx, score in enumerate(mean_attribution)])
        else:
            if importance_scores.dim() == 1:  # FC layer
                _, indices = torch.topk(importance_scores, importance_scores.shape[0] if top_m_neurons == -1 else top_m_neurons)
                flattened_importance.extend([(layer_name, importance_scores[i].item(), i) for i in indices.tolist()])
            else:  # Conv2D layer
                mean_attribution = torch.mean(importance_scores, dim=[1, 2])
                _, indices = torch.topk(mean_attribution, mean_attribution.shape[0] if top_m_neurons == -1 else min(top_m_neurons, mean_attribution.shape[0]))
                flattened_importance.extend([(layer_name, mean_attribution[i].item(), i) for i in indices.tolist()])

        # Sort flattened importance scores across all layers
        flattened_importance = sorted(flattened_importance, key=lambda x: x[1], reverse=True)
        
        # Select the top_m_neurons from the flattened list
        if top_m_neurons == -1:
            selected = flattened_importance
        else:
            selected = flattened_importance[:top_m_neurons]

        # Create a dictionary to store the selected indices per layer
        selected_indices = {}
        for layer_name, _, index in selected:
            if layer_name not in selected_indices:
                selected_indices[layer_name] = []
            selected_indices[layer_name].append(index)

        # Convert lists to tensors for each layer
        for layer_name in selected_indices:
            selected_indices[layer_name] = torch.tensor(selected_indices[layer_name])

        return selected_indices

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

# TODO: Implement the function to get the relevance scores for all layers
def get_relevance_scores_for_all_layers(model, images, labels, attribution_method='lrp'):
    model.eval()
    layer_relevance_scores = {}

    for name, layer in model.named_modules():
        # Skip if it's not a trainable layer (e.g., ReLU, pooling, etc.)
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            print(f"Processing layer: {name}")
            
            # Choose the attribution method
            if attribution_method == 'lrp':
                neuron_cond = LayerLRP(model, layer)
                relevance = neuron_cond.attribute(images, target=labels)
            # Additional methods can be added here (e.g., LayerLRP, LayerIntegratedGradients)
            else:
                raise ValueError(f"Invalid attribution method: {attribution_method}")

            # Calculate mean relevance score for each neuron (optional step for aggregation)
            mean_relevance = torch.mean(relevance, dim=0)
            layer_relevance_scores[name] = mean_relevance

    return layer_relevance_scores

def get_activation_values_for_model(model, inputs, labels, important_neuron_indices):
    model.eval()
    activation_dict = {}
    # print("Getting the Class: {}".format(labels))
    
    # Define a hook to capture activations for each layer
    def hook_fn(module, input, output, layer_name):
        activation_dict[layer_name] = output.detach()
    
    # Register hooks on all trainable layers (e.g., Linear and Conv2d)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handle = module.register_forward_hook(lambda module, input, output, layer_name=name: hook_fn(module, input, output, layer_name))
            handles.append(handle)

    # Perform forward pass to get activations
    with torch.no_grad():
        model(inputs)
    
    # Remove all hooks
    for handle in handles:
        handle.remove()

    # Select activations for important neurons if specified
    selected_activation_dict = {}
    for layer_name, activation_values in activation_dict.items():
        if important_neuron_indices is not None and layer_name in important_neuron_indices:
            indices = important_neuron_indices[layer_name]
            selected_activations = activation_values[:, indices]
            selected_activation_dict[layer_name] = selected_activations
        # else:
        #     selected_activations = activation_values
    return activation_dict, selected_activation_dict

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


def find_optimal_clusters(scores, min_k=2, max_k=10):
    scores_np = scores.cpu().detach().numpy().reshape(-1, 1)
    silhouette_list = []
    for n_clusters  in range(min_k, max_k):
        
        # Initialize the clusterer with n_clusters value and a random generator
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(scores_np)
        
        silhouette_avg = silhouette_score(scores_np, cluster_labels)
        silhouette_list.append(silhouette_avg)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        sample_silhouette_values = silhouette_samples(scores_np, cluster_labels)
        
        #  Plot the cluster info
        # plot_cluster_info(n_clusters, silhouette_avg, scores_np, clusterer, cluster_labels)
        
        ## Get the samples for each cluster
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
    best_k = silhouette_list.index(max(silhouette_list)) + min_k
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
    save_kmeans_model(kmeans, './saved_files/kmeans_impo_{}.pkl'.format(layer_name))    
    
    return cluster_labels, kmeans

# Option - 2: This is used for clustering the activation values [MAIN]
def cluster_activation_values(activation_values, n_clusters, layer_name='fc1', use_silhouette=False):
    
    kmeans_comb = []
    
    if layer_name[:-1] == 'fc':
        n_neurons = activation_values.shape[1]
        # kmeans = KMeans(n_clusters=optimal_k, random_state=0)
        for i in range(n_neurons):
            if use_silhouette:
                optimal_k = find_optimal_clusters(activation_values, 2, 10)
            else:
                optimal_k = n_clusters
                
            kmeans_comb.append(KMeans(n_clusters=optimal_k).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)))
        
        # kmeans_comb = [KMeans(n_clusters=optimal_k).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)) for i in range(n_neurons)]
        # cluster_labels = kmeans.fit_predict(activation_values_np)
        
    elif layer_name[:-1] == 'conv':
        activation_values = torch.mean(activation_values, dim=[2, 3])
        n_neurons = activation_values.shape[1]
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        for i in range(n_neurons):
            if use_silhouette:
                optimal_k = find_optimal_clusters(activation_values, 2, 10)
            else:
                optimal_k = n_clusters
                
            kmeans_comb.append(KMeans(n_clusters=optimal_k).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)))
        
        # kmeans_comb = [KMeans(n_clusters=optimal_k).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)) for i in range(n_neurons)]
        # cluster_labels = kmeans.fit_predict(activation_values_np)
    else:
        raise ValueError(f"Invalid layer name: {layer_name}")
        
    save_kmeans_model(kmeans_comb, './saved_files/kmeans_acti_{}.pkl'.format(layer_name))
    print("KMeans model saved! Name: kmeans_acti_{}.pkl".format(layer_name))
    
    return kmeans_comb

# Used for the whole model clustering
def cluster_activation_values_all(activation_dict, n_clusters, use_silhouette=False):
    kmeans_models = {}
    
    # Iterate through all layers and cluster their activations
    all_activations = []
    for layer_name, activation_values in activation_dict.items():
        if len(activation_values.shape) > 2:  # For Conv2D layers, reduce spatial dimensions
            activation_values = torch.mean(activation_values, dim=[2, 3])  # Shape: [batch_size, num_filters]
        
        all_activations.append(activation_values)

    # Concatenate all activation values across layers into one big tensor
    all_activations_tensor = torch.cat(all_activations, dim=1)  # Shape: [batch_size, total_neurons]

    # Perform clustering across all neurons
    total_neurons = all_activations_tensor.shape[1]
    kmeans_comb = []
    for i in range(total_neurons):
        if use_silhouette:
            optimal_k = find_optimal_clusters(all_activations_tensor[:, i].cpu().numpy().reshape(-1, 1), 2, 10)
        else:
            optimal_k = n_clusters
        
        kmeans_model = KMeans(n_clusters=optimal_k, random_state=42).fit(all_activations_tensor[:, i].cpu().numpy().reshape(-1, 1))
        kmeans_comb.append(kmeans_model)

    # Save KMeans models
    # save_kmeans_model(kmeans_comb, './saved_files/kmeans_acti_all_layers.pkl')
    print("KMeans models saved for all layers! Name: kmeans_acti_all_layers.pkl")

    return kmeans_comb


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

def compute_idc_test_whole(model, inputs_images, labels, kmeans, classes, top_m_neurons=-1, n_clusters = 5, attribution_method='lrp'):
    layer_importance_scores = get_relevance_scores_for_all_layers(model, 
                                                                  inputs_images, 
                                                                  labels, 
                                                                  attribution_method=attribution_method)
    indices = select_top_neurons_all(layer_importance_scores, top_m_neurons)
    activation_, selected_activations = get_activation_values_for_model(model, inputs_images, labels, indices)
    activation_values = []
    for layer_name, importance_scores in selected_activations.items():
        if layer_name[:-1] == 'conv':
            activation_values.append(torch.mean(importance_scores, dim=[2, 3]))
        else:
            activation_values.append(importance_scores)
    
    # TODO: assign_clusters here, some bugs HERE
    all_activations_tensor = torch.cat(activation_values, dim=1)  
    cluster_labels = assign_clusters(all_activations_tensor, kmeans)
    
    unique_clusters = set(cluster_labels)
    total_combination = pow(n_clusters, all_activations_tensor.shape[1])
    if all_activations_tensor.shape[0] > total_combination:
        max_coverage = 1
    else:
        max_coverage = all_activations_tensor.shape[0] / total_combination
    
    coverage_rate = len(unique_clusters) / total_combination
    print(f"Total INCC combinations: {total_combination}")
    print(f"Max Coverage (the best we can achieve): {max_coverage * 100:.6f}%")
    print(f"IDC Coverage: {coverage_rate * 100:.6f}%")
    
    return unique_clusters, coverage_rate

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
        activation_values = torch.mean(selected_activations, dim=[2, 3])
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
    
    return unique_clusters, coverage_rate

def compute_incc_centroids(important_neurons_clusters):
    centroids = {}
    for neuron, clusters in important_neurons_clusters.items():
        centroids[neuron] = [cluster.cluster_centers_ for cluster in clusters]
    return centroids

if __name__ == '__main__':
    args = parse_args()
    
    ### Model settings
    if args.model_path != 'None':
        model_path = args.model_path
    else:
        model_path = os.getenv("HOME") + '/torch-deepimportance/models_info/'
    model_path += args.saved_model
    
    ### Dataset settings
    if args.dataset == 'cifar10':
        trainloader, testloader, classes = load_CIFAR(batch_size=args.batch_size, root=args.data_path, large_image=args.large_image)
    elif args.dataset == 'mnist':
        trainloader, testloader, classes = load_MNIST(batch_size=args.batch_size, root=args.data_path)
    elif args.dataset == 'imagenet':
        # batch_size=32, root='/data/shenghao/dataset/ImageNet', num_workers=2, use_val=False
        trainloader, testloader, classes = load_ImageNet(batch_size=args.batch_size, 
                                                         root=args.data_path + '/ImageNet', 
                                                         num_workers=2, 
                                                         use_val=False)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    ### Device settings    
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
        print("Obtaining the importance scores from the file.")
        attribution, mean_attribution, labels = load_importance_scores(args.importance_file)
    else:
        if args.capture_all and not args.end2end:
            for name in module_name[1:]:
                attribution, mean_attribution = get_layer_conductance(model, images, labels, classes, 
                                                                      layer_name=name, 
                                                                      top_m_images=-1, 
                                                                      attribution_method=args.attr)
                filename = args.importance_file.replace('.json', f'_{name}.json')
                save_importance_scores(attribution, mean_attribution, filename, args.test_image)
                print("{} Saved".format(filename))
       
        elif args.end2end:
            print("Relevance scores for all layers obtained.")
            attribution = get_relevance_scores_for_all_layers(model, images, labels, attribution_method=args.attr)

        else:
            attribution, mean_attribution = get_layer_conductance(model, images, labels, classes, 
                                                                  layer_name=module_name[args.layer_index], 
                                                                  top_m_images=-1, attribution_method=args.attr)
            save_importance_scores(attribution, mean_attribution, args.importance_file, args.test_image)

    # Get the test data
    test_images, test_labels = get_class_data(testloader, classes, args.test_image)


    if args.end2end:
        important_neuron_indices = select_top_neurons_all(attribution, args.top_m_neurons)
        activation_values, selected_activations = get_activation_values_for_model(model, images, labels, important_neuron_indices)
    else:
        # Obtain the important neuron indices
        important_neuron_indices = select_top_neurons(mean_attribution, args.top_m_neurons)
        activation_values, selected_activations = get_activation_values_for_neurons(model, 
                                                                                    images, 
                                                                                    labels, 
                                                                                    important_neuron_indices, 
                                                                                    module_name[args.layer_index],
                                                                                    args.viz)
    
    if args.cluster_scores:
        ### Option - 1: Cluster the importance scores
        try:
            kmeans_model = load_kmeans_model('./saved_files/kmeans_model_{}.pkl'.format(module_name[args.layer_index]))
            print('kmeans_model_{}.pkl Loaded!'.format(module_name[args.layer_index]))
        except FileNotFoundError:
            cluster_labels, kmeans_model = cluster_importance_scores(mean_attribution, args.n_clusters)
        print("Importance scores clustered.")
    else:
        ### Option - 2: Cluster the activation values
        # cluster_labels, kmeans_comb = cluster_activation_values(selected_activations, optimal_k, module_name[args.layer_index])
        # try:
        #     kmeans_model = load_kmeans_model('./saved_files/kmeans_model_{}.pkl'.format(module_name[args.layer_index]))
        # except FileNotFoundError:
        #     cluster_labels, kmeans_comb = cluster_activation_values(selected_activations, args.n_clusters, module_name[args.layer_index], args.use_silhouette)
        if args.end2end:
            kmeans_comb = cluster_activation_values_all(selected_activations, args.n_clusters, args.use_silhouette)
        else:    
            kmeans_comb = cluster_activation_values(selected_activations, args.n_clusters, module_name[args.layer_index], args.use_silhouette)
        print("Activation values clustered.")

    ### Evaluate the attribution methods
    if args.vis_attributions:
        
        # List of available attribution methods
        attribution_methods = [
            'lc', 'la', 'ii', 'lgxa', 'lgc', 'ldl', 
            'ldls', 'lgs', 'lig', 'lfa', 'lrp'
        ]
        evaluate_attribution_methods(model, 
                                    test_images, 
                                    test_labels, 
                                    kmeans_comb, 
                                    classes, 
                                    module_name[args.layer_index], 
                                    attribution_methods, 
                                    top_m_neurons=args.top_m_neurons, 
                                    n_clusters=args.n_clusters)
    
    else:
        ### Compute IDC coverage
        if args.end2end:
            unique_cluster, coverage_rate = compute_idc_test_whole(model, 
                            test_images, 
                            test_labels, 
                            kmeans_comb, # kmeans_model
                            classes,
                            top_m_neurons=args.top_m_neurons, 
                            n_clusters=args.n_clusters,
                            attribution_method=args.attr)
        else:
            unique_cluster, coverage_rate = compute_idc_test(model, 
                            test_images, 
                            test_labels, 
                            kmeans_comb, # kmeans_model
                            classes, 
                            module_name[args.layer_index], 
                            top_m_neurons=args.top_m_neurons, 
                            n_clusters=args.n_clusters,
                            attribution_method=args.attr)

    ### Infidelity metric
    # infid = infidelity_metric(net, perturb_fn, images, attribution)
    # print(f"Infidelity: {infid:.2f}")
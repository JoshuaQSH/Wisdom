
import torch
import numpy as np
from sklearn.cluster import KMeans

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import LayerConductance, NeuronConductance


'''
TODO:
- Compute whole (partial) layer conductance
    - Input: inputs, labels, model, layer_name, attribution_method
    - Output: layer conductance
- INCC
- IDC Test
'''

class IDC:
    def __init__(self, 
                 model, 
                 dataloader,
                 kmeans, 
                 classes, 
                 target_class='plane',
                 layer_name='fc', 
                 train_all=False,
                 test_all_layer=False,
                 top_m_neurons=-1, 
                 n_clusters = 5, 
                 attribution_method='lrp'):
        
        self.model = model
        self.train_all = train_all
        self.dataloader = dataloader
        self.target_class = target_class
        self.kmeans = kmeans
        self.classes = classes
        self.layer_name = layer_name
        self.test_all_layer = test_all_layer
        self.top_m_neurons = top_m_neurons
        self.n_clusters = n_clusters
        self.attribution_method = attribution_method
        self.top_m_images = -1
        
        self.input_images = None
        self.input_labels = None
        
        if not self.train_all:
            self.input_images, self.input_labels = self.get_class_data(self.dataloader, self.classes, self.target_class)
    
    # Get the data for the target class
    def get_class_data(self, dataloader, classes, target_class):
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
    
    # Compute the difference between activations and relevance scores
    def compute_incc_centroids(self, important_neurons_clusters):
        centroids = {}
        for neuron, clusters in important_neurons_clusters.items():
            centroids[neuron] = [cluster.cluster_centers_ for cluster in clusters]
        return centroids
    
    # attribution_method = ['lrp', 'ig', 'saliency', 'deeplift']
    def get_layer_conductance(self, model):
        model = model.cpu()
        
        if self.top_m_images != -1:
            images = images[:self.top_m_images]
            labels = labels[:self.top_m_images]
            
        net_layer = getattr(model, self.layer_name)
        print("GroundTruth: {}, Model Layer: {}".format(self.classes[labels[0]], net_layer))
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        if self.attribution_method == 'lrp':
            neuron_cond = LayerConductance(model, net_layer)
            attribution = neuron_cond.attribute(images, target=labels)
        else:
            raise ValueError(f"Invalid attribution method: {attribution}")
        
        return attribution, torch.mean(attribution, dim=0)
    
    # def get_layer_conductance_all(self, model, images, labels):
    #     model = model.cpu()
    #     print("GroundTruth: {}".format(self.classes[labels[0]]))
    #     if self.attribution_method == 'lrp':
    #         neuron_cond = LayerConductance(model, net_layer)
    #         attribution = neuron_cond.attribute(images, target=labels)
    #     else:
    #         raise ValueError(f"Invalid attribution method: {attribution}")


import torch
from sklearn.cluster import KMeans

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


### Track Combinations of Clusters Activated
### mode = ['scores', 'activations']
# def compute_idc_coverage(model, inputs_images, labels, classes, kmeans_model, layer_name, top_m_neurons=None, mode='activations'):
#     model.eval()
#     covered_combinations = set()
    
#     _, layer_importance_scores = get_layer_conductance(model, inputs_images, labels, classes, layer_name=layer_name)
#     if top_m_neurons:
#         # Get indices of top m neurons
#         indices = select_top_neurons(layer_importance_scores, top_m_neurons)
#         # Assign clusters
#         if mode == 'scores':
#             # Select importance scores for top m neurons
#             selected_scores = layer_importance_scores[indices]
#             # Assign clusters
#             cluster_labels = assign_clusters_to_importance_scores(selected_scores, kmeans_model)
#         elif mode == 'activations':
#             activation_values, selected_activations = get_activation_values_for_neurons(model, 
#                                                                                         inputs_images,
#                                                                                         labels, 
#                                                                                         indices, 
#                                                                                         layer_name,
#                                                                                         False)
#             cluster_labels = assign_clusters_to_activation_values(selected_scores, kmeans_model)
    
#     elif top_m_neurons is None:
#         if mode == 'scores':
#             # Assign clusters to all neurons
#             cluster_labels = assign_clusters_to_importance_scores(layer_importance_scores, kmeans_model)
#             indices = torch.arange(len(layer_importance_scores))            
#         elif mode == 'activations':
#             activation_values, selected_activations = get_activation_values_for_neurons(model, 
#                                                                                         inputs_images,
#                                                                                         labels, 
#                                                                                         None, 
#                                                                                         layer_name,
#                                                                                         False)
#             cluster_labels = assign_clusters_to_activation_values(selected_activations, kmeans_model)
#     else:
#         raise ValueError(f"Invalid mode: {mode}")
           
#     # Record the combination (as a tuple) of cluster labels
#     combination = tuple(cluster_labels.tolist())
#     covered_combinations.add(combination)
        
#     # Compute total possible combinations
#     n_clusters = kmeans_model.n_clusters
#     n_neurons = top_m_neurons if top_m_neurons else len(layer_importance_scores)
#     total_possible_combinations = n_clusters ** n_neurons
        
#     idc_value = len(covered_combinations) / total_possible_combinations
    
#     return idc_value, covered_combinations, total_possible_combinations
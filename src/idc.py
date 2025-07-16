# src/idc.py
"""
Should be moved to the coverage.py in the future for better maintainability.

This module implements the IDC (Importance Distribution Clustering) method for analyzing the
importance of neurons in neural networks. It provides functionality for clustering neuron activations

@shenghao_qiu
"""
import json
import os

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .utils import get_layer_by_name, save_cluster_groups, load_cluster_groups

class IDC:
    def __init__(self, model, 
                 top_m_neurons, 
                 n_clusters,
                 use_silhouette, 
                 test_all_classes, 
                 clustering_method_name, 
                 clustering_params = None, 
                 cache_path=None):
        self.model = model
        self.top_m_neurons = top_m_neurons
        self.use_silhouette = use_silhouette
        self.test_all_classes = test_all_classes
        self.total_combination = 1
        self.cache_path = cache_path
        self.n_clusters = n_clusters
        
        # Initialize clustering method
        self.clustering_method_name = clustering_method_name
        self.clustering_params = clustering_params or {}
    
    def save_to_json(self, coverage_rate, max_coverage, model_name, testing_layer, file_path='coverage_rate.json'):
        """
        Save the coverage rate to a JSON file.
        """

        data = {
            'Total Combination': self.total_combination,
            'Max Coverage': max_coverage,
            'Coverage Rate': coverage_rate,
            'Model Name': model_name,
            'Testing Layer': testing_layer
        }
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
        # print(f"Coverage rate saved to {file_path}")

    ## Top-k Neurons Selection [layer-wise]
    def select_top_neurons(self, importance_scores):
        """
        Select the top-k neurons based on importance scores.
        """
        if self.top_m_neurons == -1:
            # print("Selecting all neurons.")
            if importance_scores.dim() == 1:
                _, indices = torch.sort(importance_scores, descending=True)
                return indices
            else:
                return None
        else:
            if importance_scores.dim() == 1:
                # print("Selecting top {} neurons (FC layer).".format(self.top_m_neurons))
                _, indices = torch.topk(importance_scores, self.top_m_neurons)
                return indices
            else:
                mean_attribution = torch.mean(importance_scores, dim=[1, 2])
                if mean_attribution.shape[0] < self.top_m_neurons:
                    print("Selecting all the neurons (Conv2D layer).")
                    return None
                else:
                    _, indices = torch.topk(mean_attribution, self.top_m_neurons)
                    return indices

    ## Top-k Neurons Selection [model-wise]
    def select_top_neurons_all(self, importance_scores_dict, filter_neuron=None):
        """
        Select the top-k neurons based on importance scores across all layers.
        """
        flattened_importance = []

        # Flatten and collect importance scores across all layers
        for layer_name, importance_scores in importance_scores_dict.items():
            # Fully connected layer
            if importance_scores.dim() == 1:  
                for idx, score in enumerate(importance_scores):
                    flattened_importance.append((layer_name, score.item(), idx))
            # Convolutional layer
            else:  
                # Compute mean importance per channel
                mean_attribution = torch.mean(importance_scores, dim=[1, 2])
                for idx, score in enumerate(mean_attribution):
                    flattened_importance.append((layer_name, score.item(), idx))
                
        # Filter out the last layer
        if filter_neuron is not None:
            # Filter out the specific layer (e.g., 'fc3')
            filtered_importance = [item for item in flattened_importance if item[0] != filter_neuron]
        else:
            filtered_importance = flattened_importance
        
        # Sort by importance score in descending order
        flattened_importance = sorted(filtered_importance, key=lambda x: x[1], reverse=True)
        
        # Select top-m neurons across all layers
        if self.top_m_neurons == -1:
            selected = flattened_importance
        else:
            selected = flattened_importance[:self.top_m_neurons]
                
        # Group selected neurons by layer
        selected_indices = {}
        for layer_name, _, index in selected:
            if layer_name not in selected_indices:
                selected_indices[layer_name] = []
            selected_indices[layer_name].append(index)
        
        # Convert lists to tensors
        for layer_name in selected_indices:
            selected_indices[layer_name] = torch.tensor(selected_indices[layer_name])
        
        # Neurons Index without and with scores
        return selected_indices, selected
    
    ## Get the activation values [layer-wise]
    def get_activations_neurons_sample(self, inputs, important_neuron_indices, layer_name='fc1', dataloader=None):
        """
        Get the activation values for a specific layer.
        @param inputs: Input data.
        @param important_neuron_indices: Indices of important neurons.
        @param layer_name: Name of the layer to get activation values from.
        @param dataloader: DataLoader for the dataset.
        """
        activation_values = []
        
        def hook_fn(module, input, output):
            activation_values.append(output.detach())
        
        handle = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
        
        if handle is None:
            raise ValueError(f"Layer {layer_name} not found in the model.")

        self.model.eval()

        if self.test_all_classes:
            with torch.no_grad():
                for inputs, labels in dataloader:
                    outputs = self.model(inputs)        
        else:
            with torch.no_grad():
                outputs = self.model(inputs)
        
        handle.remove()
        
        activation_values = torch.cat(activation_values, dim=0)
        if important_neuron_indices is None:
            selected_activations = activation_values
        elif type(important_neuron_indices) == torch.Tensor and important_neuron_indices.shape[0] > 0:
            selected_activations = activation_values[:, important_neuron_indices]
        else:
            raise ValueError(f"Invalid important_neuron_indices: {important_neuron_indices}")
            
        return activation_values, selected_activations
    
    ## Get the activation values [layer-wise]
    def get_activations_neurons_dataloader(self, dataloader, layer_name, important_neuron_indices):
        """
        Get the activation values for a specific layer using a dataloader.
        @param dataloader: DataLoader containing input data.
        @param layer_name: Name of the layer to get activation values from.
        @param important_neuron_indices: Indices of important neurons.
        @return: activation_dict and selected_activation_dict
        """
        activation_values = []
        def hook_fn(module, input, output):
            activation_values.append(output.detach())
        
        handle = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
        
        if handle is None:
            raise ValueError(f"Layer {layer_name} not found in the model.")

        self.model.eval()

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(next(self.model.parameters()).device)
                outputs = self.model(inputs)        
        
        handle.remove()
        
        # Concatenate all activation values from batches
        all_activations = torch.cat(activation_values, dim=0)
        
        # Create activation_dict with the specific layer
        activation_dict = {layer_name: all_activations}
        
        # Create selected_activation_dict based on important_neuron_indices
        selected_activation_dict = {}
        if important_neuron_indices is not None:
            if isinstance(important_neuron_indices, torch.Tensor) and important_neuron_indices.shape[0] > 0:
                selected_activations = all_activations[:, important_neuron_indices]
                selected_activation_dict[layer_name] = selected_activations
            elif isinstance(important_neuron_indices, dict) and layer_name in important_neuron_indices:
                indices = important_neuron_indices[layer_name]
                selected_activations = all_activations[:, indices]
                selected_activation_dict[layer_name] = selected_activations
            else:
                raise ValueError(f"Invalid important_neuron_indices: {important_neuron_indices}")
            
        return activation_dict, selected_activation_dict

    ## Get the activation [model-wise]
    def get_activations_model_sample(self, inputs, important_neuron_indices):
        """
        Get the activation values for the entire model.
        @param inputs: Input data.
        @param important_neuron_indices: Indices of important neurons.
        """
        self.model.eval()
        activation_dict = {}
        
        def hook_fn(module, input, output, layer_name):
            activation_dict[layer_name] = output.detach()
        
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                handle = module.register_forward_hook(lambda module, input, output, layer_name=name: hook_fn(module, input, output, layer_name))
                handles.append(handle)

        with torch.no_grad():
            self.model(inputs)
        
        for handle in handles:
            handle.remove()

        selected_activation_dict = {}
        for layer_name, activation_values in activation_dict.items():
            if important_neuron_indices is not None and layer_name in important_neuron_indices:
                indices = important_neuron_indices[layer_name]
                selected_activations = activation_values[:, indices]
                selected_activation_dict[layer_name] = selected_activations

        return activation_dict, selected_activation_dict
    
    ## Get the activation values [model-wise]
    def get_activations_model_dataloader(self, dataloader, important_neuron_indices):
        """
        Get the activation values for the entire model using a dataloader.
        @param dataloader: DataLoader containing input data.
        @param important_neuron_indices: Indices of important neurons.
        @return: activation_dict and selected_activation_dict
        """
        self.model.eval()
        activation_dict = {}
        
        def hook_fn(module, input, output, layer_name):
            if layer_name not in activation_dict:
                activation_dict[layer_name] = []
            # activation_dict[layer_name].append(output.detach())
            act_cpu = output.detach().to('cpu', torch.float16)
            activation_dict.setdefault(layer_name, []).append(act_cpu)
        
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                handle = module.register_forward_hook(lambda module, input, output, layer_name=name: hook_fn(module, input, output, layer_name))
                handles.append(handle)

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(next(self.model.parameters()).device)
                self.model(inputs)
                torch.cuda.empty_cache() 
        
        for handle in handles:
            handle.remove()

        # Concatenate all batch activations
        for layer_name in activation_dict:
            activation_dict[layer_name] = torch.cat(activation_dict[layer_name], dim=0)

        selected_activation_dict = {}
        for layer_name, activation_values in activation_dict.items():
            if important_neuron_indices is not None and layer_name in important_neuron_indices:
                indices = important_neuron_indices[layer_name]
                selected_activations = activation_values[:, indices]
                selected_activation_dict[layer_name] = selected_activations

        return activation_dict, selected_activation_dict

    ## Find the optimal number of clusters
    def find_optimal_clusters(self, scores, min_k=2, max_k=10):
        """
        Find the optimal number of clusters using silhouette score.
        @param scores: Importance scores.
        @param min_k: Minimum number of clusters.
        @param max_k: Maximum number of clusters.
        """
        # scores_np = scores.cpu().detach().numpy().reshape(-1, 1)
        silhouette_list = []
        for n_clusters in range(min_k, max_k):
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(scores)
            silhouette_avg = silhouette_score(scores, cluster_labels)
            silhouette_list.append(silhouette_avg)
            print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
            
        best_k = silhouette_list.index(max(silhouette_list)) + min_k
        print("Best number of clusters: ", best_k)
        return best_k
    
    ## Cluster the importance scores [layer-wise]
    def cluster_activation_values(self, activation_values, layer_name):
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"[INFO] Loading cached clusters from {self.cache_path}")
            return load_cluster_groups(self.cache_path)
        
        cluster_groups = []
        activation_values_single = next(iter(activation_values.values()))
        if isinstance(layer_name, torch.nn.Linear):
            n_neurons = activation_values_single.shape[1]
            for i in range(n_neurons):
                if self.use_silhouette:
                    optimal_k = self.find_optimal_clusters(activation_values_single.cpu().numpy().reshape(-1, 1), 2, 10)
                else:
                    optimal_k = self.n_clusters
                cluster_groups.append(KMeans(n_clusters=optimal_k).fit(activation_values_single[:, i].cpu().numpy().reshape(-1, 1)))

        elif isinstance(layer_name, torch.nn.Conv2d):
            activation_values_single = torch.mean(activation_values_single, dim=[2, 3])
            n_neurons = activation_values_single.shape[1]
            for i in range(n_neurons):
                if self.use_silhouette:
                    self.n_clusters = self.find_optimal_clusters(activation_values_single.cpu().numpy().reshape(-1, 1), 2, 10)

                cluster_groups.append(KMeans(n_clusters=self.n_clusters).fit(activation_values_single[:, i].cpu().numpy().reshape(-1, 1)))
        else:
            raise ValueError(f"Invalid layer name: {layer_name}")
        
        if self.cache_path:
            save_cluster_groups(cluster_groups, self.cache_path)
            print(f"[INFO] Saved clusters to {self.cache_path}")
        
        return cluster_groups

    ## Cluster the importance scores [model-wise]
    def cluster_activation_values_all(self, activation_dict): 
        from src.clustering import make

        if self.cache_path and os.path.exists(self.cache_path):
            print(f"[INFO] Loading cached clusters from {self.cache_path}")
            return load_cluster_groups(self.cache_path)
        
        all_activations = []
        for layer_name, activation_values in activation_dict.items():
            
            if len(activation_values.shape) > 2:
                activation_values = torch.mean(activation_values, dim=[2, 3])
            # print("[INSIDE] Layer: ", layer_name, " Activations: ", activation_values.mean())
            all_activations.append(activation_values)
        
        all_activations_tensor = torch.cat(all_activations, dim=1)
        # print("All activations MEAN: ", all_activations_tensor.mean())
        total_neurons = all_activations_tensor.shape[1]
        cluster_groups = []
        for i in range(total_neurons):
            if self.use_silhouette:
                self.n_clusters = self.find_optimal_clusters(all_activations_tensor[:, i].cpu().numpy().reshape(-1, 1), 2, 10)
            # cluster_ = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10).fit(all_activations_tensor[:, i].cpu().numpy().reshape(-1, 1))
            cluster_ = make(self.clustering_method_name, **self.clustering_params).fit(all_activations_tensor[:, i].cpu().numpy().reshape(-1, 1))
            cluster_groups.append(cluster_)
        
        if self.cache_path:
            save_cluster_groups(cluster_groups, self.cache_path)
            print(f"[INFO] Saved clusters to {self.cache_path}")

        return cluster_groups
    
    # A for loop to assign clusters to all the neurons
    def assign_clusters(self, activations, cluster_groups):
        n_samples, n_neurons = activations.shape
        cluster_comb = []
        update_total_combination = True
                
        # Loop over each test sample
        for i in range(n_samples):
            # Get the activations for this sample
            sample_activations = activations[i].cpu().numpy().astype(np.float32)
            sample_clusters = []
            
            # Loop over each neuron and assign it to a cluster
            for neuron_idx in range(n_neurons):
                try:
                    cluster = cluster_groups[neuron_idx].predict(sample_activations[neuron_idx].reshape(-1, 1))
                except ValueError as e:
                    if "Buffer dtype mismatch" in str(e):
                        # print("Type mismatch detected. Converting to float64 and retrying...")
                        cluster = cluster_groups[neuron_idx].predict(sample_activations[neuron_idx].astype(np.float64).reshape(-1, 1))
                    else:
                        raise e
                
                sample_clusters.append(cluster[0])  # Append the cluster ID
                if update_total_combination:
                    if hasattr(cluster_groups[neuron_idx], "n_clusters"):
                        self.total_combination *= cluster_groups[neuron_idx].n_clusters
                    else:
                        self.total_combination *= len(cluster_groups[neuron_idx].cluster_centers_)
            
            # Convert to tuple for uniqueness
            cluster_comb.append(tuple(sample_clusters))
            update_total_combination = False
            
        return cluster_comb

    ## Compute the IDC test [model-wise]
    def compute_idc_test_whole(self, inputs_images, indices, cluster_groups):

        activation_, selected_activations = self.get_activations_model_sample(inputs_images, indices)
        activation_values = []
        for layer_name, importance_scores in selected_activations.items():
            layer = get_layer_by_name(self.model, layer_name)
            if isinstance(layer, torch.nn.Conv2d):
                activation_values.append(torch.mean(selected_activations[layer_name], dim=[2, 3]))
            else:
                activation_values.append(selected_activations[layer_name])
        
        all_activations_tensor = torch.cat(activation_values, dim=1)
        cluster_comb = self.assign_clusters(all_activations_tensor, cluster_groups)
        unique_clusters = set(cluster_comb)
        # total_combination = pow(self.n_clusters, all_activaptions_tensor.shape[1])
        total_combination = self.total_combination
        if all_activations_tensor.shape[0] > total_combination:
            max_coverage = 1
        else:
            max_coverage = all_activations_tensor.shape[0] / total_combination
        
        coverage_rate = len(unique_clusters) / total_combination
        # model_name = self.model.__class__.__name__
        # self.save_to_json(coverage_rate, max_coverage, model_name, "Whole model")
        
        return coverage_rate, total_combination, max_coverage
    

    ## Compute the IDC test [layer-wise]
    def compute_idc_test(self, inputs_images, indices, cluster_groups, net_layer, layer_name):
        
        self.test_all_classes = False
        activation_values, selected_activations = self.get_activations_neurons_sample(inputs_images, indices, layer_name)
        
        if isinstance(net_layer, torch.nn.Conv2d):
            activation_values = torch.mean(selected_activations, dim=[2, 3])
        else:
            activation_values = selected_activations
        
        activation_values_np = activation_values.cpu().numpy()
        cluster_comb = self.assign_clusters(activation_values, cluster_groups)
        
        unique_clusters = set(cluster_comb)
        # total_combination = pow(self.n_clusters, activation_values_np.shape[1])
        total_combination = self.total_combination
        if activation_values.shape[0] > total_combination:
            max_coverage = 1
        else:
            max_coverage = activation_values.shape[0] / total_combination
        
        coverage_rate = len(unique_clusters) / total_combination
        model_name = self.model.__class__.__name__
        self.save_to_json(coverage_rate, max_coverage, model_name, layer_name)
        
        return coverage_rate, total_combination, max_coverage
    
    ## Compute the IDC test [model-wise]
    def compute_idc_test_whole_dataloader(self, dataloader, indices, cluster_groups):

        activation_, selected_activations = self.get_activations_model_dataloader(dataloader, indices)
        activation_values = []
        for layer_name, importance_scores in selected_activations.items():
            layer = get_layer_by_name(self.model, layer_name)
            if isinstance(layer, torch.nn.Conv2d):
                activation_values.append(torch.mean(selected_activations[layer_name], dim=[2, 3]))
            else:
                activation_values.append(selected_activations[layer_name])
        
        all_activations_tensor = torch.cat(activation_values, dim=1)
        cluster_comb = self.assign_clusters(all_activations_tensor, cluster_groups)
        unique_clusters = set(cluster_comb)
        # total_combination = pow(self.n_clusters, all_activaptions_tensor.shape[1])
        total_combination = self.total_combination
        if all_activations_tensor.shape[0] > total_combination:
            max_coverage = 1
        else:
            max_coverage = all_activations_tensor.shape[0] / total_combination
        
        coverage_rate = len(unique_clusters) / total_combination
        model_name = self.model.__class__.__name__
        self.save_to_json(coverage_rate, max_coverage, model_name, "Whole model")
        
        return coverage_rate, total_combination, max_coverage
    

    ## Compute the IDC test [layer-wise]
    def compute_idc_test_dataloader(self, dataloader, indices, cluster_groups, net_layer, layer_name):
        
        self.test_all_classes = False

        activation_values, selected_activations = self.get_activations_neurons_dataloader(dataloader, layer_name, indices)
        selected_activations_single = next(iter(selected_activations.values()))
        
        if isinstance(net_layer, torch.nn.Conv2d):
            activation_values = torch.mean(selected_activations_single, dim=[2, 3])
        else:
            activation_values = selected_activations_single
        
        activation_values_np = activation_values.cpu().numpy()
        cluster_comb = self.assign_clusters(activation_values, cluster_groups)
        
        unique_clusters = set(cluster_comb)
        # total_combination = pow(self.n_clusters, activation_values_np.shape[1])
        total_combination = self.total_combination
        if activation_values.shape[0] > total_combination:
            max_coverage = 1
        else:
            max_coverage = activation_values.shape[0] / total_combination
        
        coverage_rate = len(unique_clusters) / total_combination
        model_name = self.model.__class__.__name__
        self.save_to_json(coverage_rate, max_coverage, model_name, layer_name)
        
        return coverage_rate, total_combination, max_coverage
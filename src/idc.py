# src/idc.py
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from attribution import get_relevance_scores, get_relevance_scores_for_all_layers
from utils import load_kmeans_model, save_kmeans_model, get_layer_by_name
from visualization import visualize_activation, visualize_idc_scores
import json
import numpy as np

class IDC:
    def __init__(self, model, classes, top_m_neurons, n_clusters, use_silhouette, test_all_classes):
        self.model = model
        self.classes = classes
        self.top_m_neurons = top_m_neurons
        self.n_clusters = n_clusters
        self.use_silhouette = use_silhouette
        self.test_all_classes = test_all_classes
        self.total_combination = 1
    
    def save_to_json(self, coverage_rate, model_name, testing_class, testing_layer, file_path='coverage_rate.json'):
        data = {
            'coverage_rate': coverage_rate,
            'model_name': model_name,
            'testing_class': testing_class,
            'testing_layer': testing_layer
        }
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
        print(f"Coverage rate saved to {file_path}")

    ## Top-k Neurons Selection [layer-wise]
    def select_top_neurons(self, importance_scores):
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
    def get_activation_values_for_neurons(self, inputs, important_neuron_indices, layer_name='fc1', dataloader=None):
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

    ## Get the activation values [model-wise]
    def get_activation_values_for_model(self, inputs, labels, important_neuron_indices):
        self.model.eval()
        activation_dict = {}
        print("Getting the Class: {}".format(labels))
        
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

    ## Find the optimal number of clusters
    def find_optimal_clusters(self, scores, min_k=2, max_k=10):
        scores_np = scores.cpu().detach().numpy().reshape(-1, 1)
        silhouette_list = []
        for n_clusters in range(min_k, max_k):
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(scores_np)
            silhouette_avg = silhouette_score(scores_np, cluster_labels)
            silhouette_list.append(silhouette_avg)
            print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
            
        best_k = silhouette_list.index(max(silhouette_list)) + min_k
        print("Best number of clusters: ", best_k)
        return best_k

    ## Cluster the importance scores [layer-wise]
    def cluster_importance_scores(self, importance_scores, n_clusters, layer_name='fc1'):
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
        
        print("The cluster labels: {}, etc.".format(cluster_labels[:10]))
        save_kmeans_model(kmeans, './saved_files/kmeans_impo_{}.pkl'.format(layer_name))    
        
        return cluster_labels, kmeans
    
    ## Cluster the importance scores [layer-wise]
    def cluster_activation_values(self, activation_values, layer_name):
        kmeans_comb = []
        
        if isinstance(layer_name, torch.nn.Linear):
            n_neurons = activation_values.shape[1]
            for i in range(n_neurons):
                if self.use_silhouette:
                    optimal_k = self.find_optimal_clusters(activation_values, 2, 10)
                else:
                    optimal_k = self.n_clusters
                kmeans_comb.append(KMeans(n_clusters=optimal_k).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)))

        elif isinstance(layer_name, torch.nn.Conv2d):
            activation_values = torch.mean(activation_values, dim=[2, 3])
            n_neurons = activation_values.shape[1]
            for i in range(n_neurons):
                if self.use_silhouette:
                    self.n_clusters = self.find_optimal_clusters(activation_values, 2, 10)

                kmeans_comb.append(KMeans(n_clusters=self.n_clusters).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)))
        else:
            raise ValueError(f"Invalid layer name: {layer_name}")
        
        # save_kmeans_model(kmeans_comb, './saved_files/kmeans_acti_{}.pkl'.format(layer_name))
        # print("KMeans model saved! Name: kmeans_acti_{}.pkl".format(layer_name))
        
        return kmeans_comb

    ## Cluster the importance scores [layer-wise]
    def cluster_activation_values_(self, activation_values, layer_name='fc1'):
        kmeans_comb = []
        
        if layer_name[:-1] == 'fc':
            n_neurons = activation_values.shape[1]
            for i in range(n_neurons):
                if self.use_silhouette:
                    optimal_k = self.find_optimal_clusters(activation_values, 2, 10)
                else:
                    optimal_k = self.n_clusters
                kmeans_comb.append(KMeans(n_clusters=optimal_k).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)))

        elif layer_name[:-1] == 'conv':
            activation_values = torch.mean(activation_values, dim=[2, 3])
            n_neurons = activation_values.shape[1]
            for i in range(n_neurons):
                if self.use_silhouette:
                    self.n_clusters = self.find_optimal_clusters(activation_values, 2, 10)

                kmeans_comb.append(KMeans(n_clusters=self.n_clusters).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)))
        else:
            raise ValueError(f"Invalid layer name: {layer_name}")
        
        # save_kmeans_model(kmeans_comb, './saved_files/kmeans_acti_{}.pkl'.format(layer_name))
        # print("KMeans model saved! Name: kmeans_acti_{}.pkl".format(layer_name))
        
        return kmeans_comb

    ## Cluster the importance scores [model-wise]
    def cluster_activation_values_all(self, activation_dict):
        kmeans_models = {}
        
        all_activations = []
        for layer_name, activation_values in activation_dict.items():
            if len(activation_values.shape) > 2:
                activation_values = torch.mean(activation_values, dim=[2, 3])
            all_activations.append(activation_values)

        all_activations_tensor = torch.cat(all_activations, dim=1)

        total_neurons = all_activations_tensor.shape[1]
        kmeans_comb = []
        for i in range(total_neurons):
            if self.use_silhouette:
                self.n_clusters = self.find_optimal_clusters(all_activations_tensor[:, i].reshape(-1, 1), 2, 10)
            
            kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42).fit(all_activations_tensor[:, i].reshape(-1, 1))
            kmeans_comb.append(kmeans_model)

        # print("KMeans models saved for all layers! Name: kmeans_acti_all_layers.pkl")
        return kmeans_comb
    
    # A for loop to assign clusters to all the neurons
    def assign_clusters(self, activations, kmeans_models):
        n_samples, n_neurons = activations.shape
        cluster_assignments = []
        update_total_combination = True
        
        # Loop over each test sample
        for i in range(n_samples):
            # Get the activations for this sample
            sample_activations = activations[i].cpu().numpy().astype(np.float32)
            sample_clusters = []
            
            # Loop over each neuron and assign it to a cluster
            for neuron_idx in range(n_neurons):
                try:
                    cluster = kmeans_models[neuron_idx].predict(sample_activations[neuron_idx].reshape(-1, 1))
                except ValueError as e:
                    if "Buffer dtype mismatch" in str(e):
                        print("Type mismatch detected. Converting to float64 and retrying...")
                        cluster = kmeans_models[neuron_idx].predict(sample_activations[neuron_idx].astype(np.float64).reshape(-1, 1))
                    else:
                        raise e                
                
                sample_clusters.append(cluster[0])  # Append the cluster ID
                if update_total_combination:
                    self.total_combination *= kmeans_models[neuron_idx].n_clusters
            
            # Convert to tuple for uniqueness
            cluster_assignments.append(tuple(sample_clusters))
            update_total_combination = False
            
        return cluster_assignments

    ## Compute the IDC test [model-wise]
    def compute_idc_test_whole(self, inputs_images, labels, indices, kmeans, attribution_method='lrp'):

        activation_, selected_activations = self.get_activation_values_for_model(inputs_images, self.classes[labels[0]], indices)
        activation_values = []
        for layer_name, importance_scores in selected_activations.items():
            layer = get_layer_by_name(self.model, layer_name)
            # TODO: VGG16 Conv2D layer seems to have issue
            if isinstance(layer, torch.nn.Conv2d):
                activation_values.append(torch.mean(selected_activations[layer_name], dim=[2, 3]))
            else:
                activation_values.append(selected_activations[layer_name])
        # TODO: assign_clusters here, some bugs HERE
        all_activations_tensor = torch.cat(activation_values, dim=1)
        cluster_labels = self.assign_clusters(all_activations_tensor, kmeans)
        unique_clusters = set(cluster_labels)
        # total_combination = pow(self.n_clusters, all_activations_tensor.shape[1])
        total_combination = self.total_combination
        if all_activations_tensor.shape[0] > total_combination:
            max_coverage = 1
        else:
            max_coverage = all_activations_tensor.shape[0] / total_combination
        
        coverage_rate = len(unique_clusters) / total_combination
        print("Attribution Method: ", attribution_method)
        print(f"Total INCC combinations: {total_combination}")
        print(f"Max Coverage (the best we can achieve): {max_coverage * 100:.6f}%")
        print(f"IDC Coverage: {coverage_rate * 100:.6f}%")
        
        model_name = self.model.__class__.__name__
        testing_class = self.classes[labels[0]]
        self.save_to_json(coverage_rate, model_name, testing_class, "Whole model")
        
        return unique_clusters, coverage_rate

    ## Compute the IDC test [layer-wise]
    def compute_idc_test(self, inputs_images, labels, indices, kmeans, net_layer, layer_name, attribution_method='lrp'):
        
        print("The first label for IDC is: {}".format(self.classes[labels[0]]))
        self.test_all_classes = False
        activation_values, selected_activations = self.get_activation_values_for_neurons(inputs_images, indices, layer_name)
        if isinstance(net_layer, torch.nn.Conv2d):
            activation_values = torch.mean(selected_activations, dim=[2, 3])
        else:
            activation_values = selected_activations
        
        activation_values_np = activation_values.cpu().numpy()
        cluster_labels = self.assign_clusters(kmeans_models=kmeans, activations=activation_values)
        
        unique_clusters = set(cluster_labels)
        # total_combination = pow(self.n_clusters, activation_values_np.shape[1])
        total_combination = self.total_combination
        if activation_values.shape[0] > total_combination:
            max_coverage = 1
        else:
            max_coverage = activation_values.shape[0] / total_combination
        
        coverage_rate = len(unique_clusters) / total_combination
        print("Attribution Method: ", attribution_method)
        print(f"Total INCC combinations: ", total_combination)
        print(f"Max Coverage (the best we can achieve): {max_coverage * 100:.6f}%")
        print(f"IDC Coverage: {coverage_rate * 100:.6f}%")
        
        model_name = self.model.__class__.__name__
        testing_class = self.classes[labels[0]]
        self.save_to_json(coverage_rate, model_name, testing_class, layer_name)
        
        return unique_clusters, coverage_rate
    
    def evaluate_attribution_methods(self, inputs_images, labels, kmeans, layer_name, attribution_methods):
        idc_scores = {}
        for method in attribution_methods:
            print(f"Evaluating method: {method}")
            # Compute IDC for the given attribution method
            _, idc_score = self.compute_idc_test(inputs_images, labels, kmeans, layer_name, attribution_method=method)
            
            print(f"IDC score for {method}: {idc_score}")
            # Store the IDC score for this method
            idc_scores[method] = idc_score
        
        visualize_idc_scores(idc_scores)
        return idc_scores
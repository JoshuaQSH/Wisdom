import random
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class PruningMethods:
    def __init__(self, model):
        """
        Initialize the pruning methods class with a model.
        
        Args:
            model (torch.nn.Module): The model to be pruned.
        """
        self.model = model

    def random_prune(self, num_neurons, sparse_prune=False):
        """
        Randomly prune neurons or filters across all eligible layers.

        Args:
            num_neurons (int): Number of neurons/filters to prune.
            sparse_prune (bool): Whether to use sparse pruning.
        """
        eligible_layers = [
            (name, layer) for name, layer in self.model.named_modules()
            if isinstance(layer, (nn.Linear, nn.Conv2d))
        ]
        
        all_neuron_indices = []
        for name, layer in eligible_layers:
            num_units = layer.out_features if isinstance(layer, nn.Linear) else layer.out_channels
            all_neuron_indices.extend([(name, idx) for idx in range(num_units)])
        
        selected = random.sample(all_neuron_indices, min(num_neurons, len(all_neuron_indices)))

        for name, idx in selected:
            layer = dict(self.model.named_modules())[name]
            if sparse_prune:
                amount = 1.0 / (layer.out_features if isinstance(layer, nn.Linear) else layer.out_channels)
                prune.random_unstructured(layer, name='weight', amount=amount)
            else:
                with torch.no_grad():
                    if isinstance(layer, nn.Linear):
                        layer.weight[idx, :] = 0
                        if layer.bias is not None:
                            layer.bias[idx] = 0
                    elif isinstance(layer, nn.Conv2d):
                        layer.weight[idx, :, :, :] = 0
                        if layer.bias is not None:
                            layer.bias[idx] = 0

    def prune_specific_layer(self, layer_name, neurons_to_prune):
        """
        Prune specific neurons or filters in a given layer.

        Args:
            layer_name (str): Name of the layer to prune.
            neurons_to_prune (list): List of neuron/filter indices to prune.
        """
        layer = dict(self.model.named_modules()).get(layer_name)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")
        
        with torch.no_grad():
            if isinstance(layer, nn.Linear):
                layer.weight[neurons_to_prune, :] = 0
                if layer.bias is not None:
                    layer.bias[neurons_to_prune] = 0
            elif isinstance(layer, nn.Conv2d):
                for idx in neurons_to_prune:
                    layer.weight[idx, :, :, :] = 0
                    if layer.bias is not None:
                        layer.bias[idx] = 0
            else:
                raise ValueError(f"Pruning is only implemented for Linear and Conv2D layers. Given: {type(layer)}")

    def prune_across_layers(self, layers_to_prune):
        """
        Prune neurons or filters across multiple layers.

        Args:
            layers_to_prune (list of tuples): List of tuples where each tuple contains:
                - layer_name (str): Name of the layer to prune.
                - relative_importance (float): Relative importance score of the neuron/filter.
                - neuron_index (int): Index of the neuron/filter to prune.
                e.g., [('features.10', 0.46760982275009155, 15), ('features.20', 0.44575604796409607, 243), ...]
        """
        with torch.no_grad():
            for layer_name, neuron_index in layers_to_prune:
                layer = dict(self.model.named_modules()).get(layer_name)
                if layer is None:
                    raise ValueError(f"Layer '{layer_name}' not found in the model.")
                
                if isinstance(layer, nn.Linear):
                    layer.weight[neuron_index, :] = 0
                    if layer.bias is not None:
                        layer.bias[neuron_index] = 0
                elif isinstance(layer, nn.Conv2d):
                    layer.weight[neuron_index, :, :, :] = 0
                    if layer.bias is not None:
                        layer.bias[neuron_index] = 0
                else:
                    raise ValueError(f"Pruning is only implemented for Linear and Conv2D layers. Given: {type(layer)}")
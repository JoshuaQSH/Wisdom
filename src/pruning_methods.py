import random
import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.utils import test_model_dataloder

def random_prune_whole_model(model, num_neurons=5, sparse_prune=False):
    # Collect all eligible layers
    eligible_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            eligible_layers.append((name, layer))
    
    # Flat list of all neuron indices: [(layer, neuron_idx), ...]
    all_neuron_indices = []
    for name, layer in eligible_layers:
        num_units = layer.out_features if isinstance(layer, nn.Linear) else layer.out_channels
        all_neuron_indices.extend([(name, idx) for idx in range(num_units)])

    # Randomly select neurons/filters to prune
    selected = random.sample(all_neuron_indices, min(num_neurons, len(all_neuron_indices)))

    for name, idx in selected:
        layer = dict(model.named_modules())[name]

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
                else:
                    raise ValueError(f"Unsupported layer type: {type(layer)}")
    return model

def ramdon_prune(model, layer_name='fc1', neurons_to_prune=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], num_neurons=5, sparse_prune=False):
    layer = getattr(model, layer_name)
    
    ## Use the torch pruning for the random prune
    if sparse_prune:
        amount = num_neurons / layer.weight.shape[0]
        prune.random_unstructured(layer, name="weight", amount=amount)
    
    # Take each of the neuron and channel (conv2D) as basic unit
    else:
        with torch.no_grad():
            
            full_tensor = torch.arange(layer.weight.shape[0])
            
            # Exclude the neurons to prune
            for v in neurons_to_prune:
                full_tensor = full_tensor[full_tensor!=v]
            
            possible_choices = random.sample(set(full_tensor), num_neurons)
            possible_choices = [t.item() for t in possible_choices]
            
            if isinstance(layer, nn.Linear):
                # Set weights and biases of the selected neurons to zero
                layer.weight[possible_choices, :] = 0
                if layer.bias is not None:
                    layer.bias[possible_choices] = 0
            
            elif isinstance(layer, nn.Conv2d):
                # Set the weights of the selected filters to zero
                for f in possible_choices:
                    layer.weight[f] = 0
                    if layer.bias is not None:
                        layer.bias[f] = 0
            else:
                raise ValueError(f"Pruning is only implemented for Linear and Conv2D layers. Given: {type(layer)}")

    # print(
    #         "Sparsity in weight: {:.2f}%".format(
    #             100. * float(torch.sum(layer.weight == 0))
    #             / float(layer.weight.nelement())
    #         )
    #     )

def prune_neurons(model, layer='fc1', neurons_to_prune=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    
    # layer = getattr(model, layer_name)
    with torch.no_grad():
        if isinstance(layer, nn.Linear):
            # Set weights and biases of the selected neurons to zero
            layer.weight[neurons_to_prune, :] = 0
            if layer.bias is not None:
                layer.bias[neurons_to_prune] = 0
        elif isinstance(layer, nn.Conv2d):
            # Set the weights of the selected filters to zero
            for f in neurons_to_prune:
                layer.weight[f] = 0
                if layer.bias is not None:
                    layer.bias[f] = 0
        else:
            raise ValueError(f"Pruning is only implemented for Linear and Conv2D layers. Given: {type(layer)}")

def prune_layers(model, layers_to_prune):
    """
    Args:
        model (torch.nn.Module): The model to prune.
        layers_to_prune (list of tuples): A list where each tuple contains:
            - layer_name (str): The name of the layer to prune.
            - importance_score (float): The importance score (not used in pruning).
            - neuron_index (int): The index of the neuron to prune.
    """
    with torch.no_grad():
        for lname, _, idx in layers_to_prune:
            layer = dict(model.named_modules()).get(lname)
            if layer is None:
                raise ValueError(f"Layer '{lname}' not found in the model.")
            
            # Prune the neuron based on its type
            if isinstance(layer, nn.Linear):
                # Set weights and biases of the selected neuron to zero
                layer.weight[idx, :] = 0
                if layer.bias is not None:
                    layer.bias[idx] = 0
            elif isinstance(layer, nn.Conv2d):
                # Set the weights of the selected filter to zero
                layer.weight[idx] = 0
                if layer.bias is not None:
                    layer.bias[idx] = 0
            else:
                raise ValueError(f"Pruning is only implemented for Linear and Conv2D layers. Given: {type(layer)}")
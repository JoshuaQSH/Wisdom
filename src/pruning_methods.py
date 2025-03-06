import random

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

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

def prune_neurons_(model, layer_name='fc1', neurons_to_prune=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    breakpoint()
    layer = getattr(model, layer_name)
    
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
    # print(
    #     "Sparsity in weight: {:.2f}%".format(
    #         100. * float(torch.sum(layer.weight == 0))
    #         / float(layer.weight.nelement())
    #     )
    # )
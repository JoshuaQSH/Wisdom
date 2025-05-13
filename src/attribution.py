# src/attribution.py
import torch
from captum.attr import (
    LayerConductance, LayerActivation, InternalInfluence, 
    LayerGradientXActivation, LayerGradCam, LayerDeepLift, 
    LayerDeepLiftShap, LayerGradientShap, LayerIntegratedGradients, 
    LayerFeatureAblation, LayerLRP
)

attribution_classes = {
        'lc': LayerConductance,
        'la': LayerActivation,
        'ii': InternalInfluence,
        'lgxa': LayerGradientXActivation,
        'lgc': LayerGradCam,
        'ldl': LayerDeepLift,
        'ldls': LayerDeepLiftShap,
        'lgs': LayerGradientShap,
        'lig': LayerIntegratedGradients,
        'lfa': LayerFeatureAblation,
        'lrp': LayerLRP
    }

def traverse_sequential_rule(layer):
    rules = []
    for sub_layer in layer:
        rules.append(LayerLRP(sub_layer))
    return rules

def get_relevance_scores(model, images, labels, classes, net_layer, layer_name='fc1', top_m_images=-1, attribution_method='lrp'):
    model.eval()
    model = model.cpu()

    if top_m_images != -1:
        images = images[:top_m_images]
        labels = labels[:top_m_images]
    
    # print("Running with {}".format(attribution_method))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    if attribution_method not in attribution_classes:
        raise ValueError(f"Invalid attribustion method: {attribution_method}")

    neuron_cond_class = attribution_classes[attribution_method]
    neuron_cond = neuron_cond_class(model, net_layer)

    if attribution_method in ['ldl', 'ldls', 'lgs']:
        attribution = neuron_cond.attribute(images, baselines=torch.zeros_like(images), target=labels)
    elif attribution_method == 'la':
        attribution = neuron_cond.attribute(images)
    else:
        attribution = neuron_cond.attribute(images, target=labels)
    
    return attribution, torch.mean(attribution, dim=0)

def get_relevance_scores_for_all_classes(model, dataloader, net_layer, layer_name='fc1', attribution_method='lrp'):
    
    model.eval()
    model = model.cpu()
    
    print("Layer's Name: {}, Module: {}".format(layer_name, net_layer))
    print("Running with {}".format(attribution_method))
    
    if attribution_method not in attribution_classes:
        raise ValueError(f"Invalid attribution method: {attribution_method}")

    total_attribution = None
    num_samples = 0

    for images, labels in dataloader:
        neuron_cond_class = attribution_classes[attribution_method]
        neuron_cond = neuron_cond_class(model, net_layer)
        if attribution_method in ['ldl', 'ldls', 'lgs']:
            attribution = neuron_cond.attribute(images, baselines=torch.zeros_like(images), target=labels)
        elif attribution_method == 'la':
            attribution = neuron_cond.attribute(images)
        else:
            attribution = neuron_cond.attribute(images, target=labels)
        
        # Accumulate attributions and count samples
        if total_attribution is None:
            total_attribution = torch.sum(attribution, dim=0)  # Sum attributions across batch
        else:
            total_attribution += torch.sum(attribution, dim=0)
        
        num_samples += images.size(0)

    # Compute mean attribution
    mean_attribution = total_attribution / num_samples
    return mean_attribution

def get_relevance_scores_for_all_layers(model, images, labels, device, attribution_method='lrp'):
    model = model.to(device)
    model.eval()
    layer_relevance_scores = {}
    num_samples = labels.size(0)

    if attribution_method not in attribution_classes:
        raise ValueError(f"Invalid attribution method: {attribution_method}")

    candidate_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            candidate_layers.append((name, layer))
            if isinstance(layer, torch.nn.Linear):
                layer_relevance_scores[name] = torch.zeros(layer.out_features)
            else:
                layer_relevance_scores[name] = torch.zeros(layer.out_channels)
    
    # Begin attribution
    # print(f"Processing layer: {name}, Attribution: {attribution_method}")
    images, labels = images.to(device), labels.to(device)
    
    for name, layer in candidate_layers:
        neuron_cond_class = attribution_classes[attribution_method]
        neuron_cond = neuron_cond_class(model, layer)
        # Compute relevance using attribution method
        if attribution_method in ['ldl', 'ldls', 'lgs']:
            relevance = neuron_cond.attribute(images, baselines=torch.zeros_like(images), target=labels)
        elif attribution_method == 'la':
            relevance = neuron_cond.attribute(images)
        else:
            relevance = neuron_cond.attribute(images, target=labels)

        # Sum relevance across batch and spatial dims if Conv2d
        # TODO: option 1: sum over all channels, option 2: mean over all channels
        if relevance.dim() == 4:
            batch_sum = relevance.sum(dim=(0, 2, 3)).detach().cpu()
            # batch_sum = torch.mean(relevance, dim=[2, 3]).sum(dim=0).detach().cpu()
        else:
            batch_sum = relevance.sum(dim=0).detach().cpu()
        
        layer_relevance_scores[name] += batch_sum
    num_samples += images.size(0)
    
    # Normalize by total samples
    for name in layer_relevance_scores:
        layer_relevance_scores[name] /= num_samples
        
    return layer_relevance_scores

# Get relevance scores for all layers with the dataloader
def get_relevance_scores_dataloader(model, dataloader, device, attribution_method='lrp'):
    """
    Computes per-neuron relevance scores for all Linear and Conv2D layers (except final classifier).
    Relevance is computed using the specified attribution method and averaged over all training data.
    
    Args:
        model (nn.Module): Trained PyTorch model (e.g., LeNet).
        dataloader (DataLoader): DataLoader for the training dataset.
        attribution_method (str): One of 'lrp', 'ldl', 'ldls', etc.

    Returns:
        dict[str, torch.Tensor]: {layer_name: mean relevance tensor}
    """
    model = model.to(device)
    model.eval()
    
    layer_relevance_scores = {}
    num_samples = 0

    if attribution_method not in attribution_classes:
        raise ValueError(f"Invalid attribution method: {attribution_method}")

    # Identify eligible layers and initialize accumulators
    candidate_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            candidate_layers.append((name, layer))
            if isinstance(layer, torch.nn.Linear):
                layer_relevance_scores[name] = torch.zeros(layer.out_features)
            else:
                layer_relevance_scores[name] = torch.zeros(layer.out_channels)

    # Loop over training data
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        for name, layer in candidate_layers:
            neuron_cond_class = attribution_classes[attribution_method]
            neuron_cond = neuron_cond_class(model, layer)

            # Compute relevance using attribution method
            if attribution_method in ['ldl', 'ldls', 'lgs']:
                relevance = neuron_cond.attribute(images, baselines=torch.zeros_like(images), target=labels)
            elif attribution_method == 'la':
                relevance = neuron_cond.attribute(images)
            else:
                relevance = neuron_cond.attribute(images, target=labels)

            # Sum relevance across batch and spatial dims if Conv2d
            # TODO: option 1: sum over all channels, option 2: mean over all channels
            if relevance.dim() == 4:
                batch_sum = relevance.sum(dim=(0, 2, 3)).detach().cpu()
                # batch_sum = torch.mean(relevance, dim=[2, 3]).sum(dim=0).detach().cpu()
            else:
                batch_sum = relevance.sum(dim=0).detach().cpu()

            layer_relevance_scores[name] += batch_sum
        num_samples += images.size(0)

    # Normalize by total samples
    for name in layer_relevance_scores:
        layer_relevance_scores[name] /= num_samples

    return layer_relevance_scores
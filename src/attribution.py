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

# TODO: Implement the function to get the relevance scores for all layers [Require testing], should ingore the last layer
def get_relevance_scores_for_all_layers(model, images, labels, attribution_method='lrp'):
    model.eval()
    layer_relevance_scores = {}

    if attribution_method not in attribution_classes:
        raise ValueError(f"Invalid attribution method: {attribution_method}")

    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            # print(f"Processing layer: {name}, Attribution: {attribution_method}")
            
            neuron_cond_class = attribution_classes[attribution_method]
            neuron_cond = neuron_cond_class(model, layer)

            if attribution_method in ['ldl', 'ldls', 'lgs']:
                relevance = neuron_cond.attribute(images, baselines=torch.zeros_like(images), target=labels)
            elif attribution_method == 'la':
                relevance = neuron_cond.attribute(images)
            else:
                relevance = neuron_cond.attribute(images, target=labels)

            mean_relevance = torch.mean(relevance, dim=0)
            layer_relevance_scores[name] = mean_relevance
    return layer_relevance_scores
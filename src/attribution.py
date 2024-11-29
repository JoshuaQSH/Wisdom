# src/attribution.py
import torch
from captum.attr import (
    LayerConductance, LayerActivation, InternalInfluence, 
    LayerGradientXActivation, LayerGradCam, LayerDeepLift, 
    LayerDeepLiftShap, LayerGradientShap, LayerIntegratedGradients, 
    LayerFeatureAblation, LayerLRP
)

def get_layer_conductance(model, images, labels, classes, layer_name='fc1', top_m_images=-1, attribution_method='lrp'):
    model = model.cpu()
    
    if top_m_images != -1:
        images = images[:top_m_images]
        labels = labels[:top_m_images]
        
    net_layer = getattr(model, layer_name)
    print("GroundTruth: {}, Model Layer: {}".format(classes[labels[0]], net_layer))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
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

# TODO: Implement the function to get the relevance scores for all layers [Require testing]
def get_relevance_scores_for_all_layers(model, images, labels, attribution_method='lrp'):
    model.eval()
    layer_relevance_scores = {}

    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            print(f"Processing layer: {name}")
            
            if attribution_method == 'lrp':
                neuron_cond = LayerLRP(model, layer)
                relevance = neuron_cond.attribute(images, target=labels)
            else:
                raise ValueError(f"Invalid attribution method: {attribution_method}")

            mean_relevance = torch.mean(relevance, dim=0)
            layer_relevance_scores[name] = mean_relevance

    return layer_relevance_scores
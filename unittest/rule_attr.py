import torch, torch.nn as nn
from captum.attr._utils.lrp_rules import (
    PropagationRule,                 # abstract base
    EpsilonRule,                     # simplest concrete rule we can reuse
)

from models_info.models_cv import *
from src.utils import load_ImageNet, get_model, eval_model_dataloder

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

SAVED_MODEL_PATH = ['./models_info/saved_models/convnext_base_IMAGENET_whole.pth', 
                    './models_info/saved_models/resnet18_IMAGENET_whole.pth',
                    './models_info/saved_models/mobilenet_v3_small_IMAGENET_whole.pth', 
                    './models_info/saved_models/resnet152_IMAGENET_whole.pth', 
                    './models_info/saved_models/efficientnet_v2_m_IMAGENET_whole.pth', 
                    './models_info/saved_models/efficientnet_v2_s_IMAGENET_whole.pth',
                    './models_info/saved_models/googlenet_IMAGENET_whole.pth',
                    './models_info/saved_models/mnasnet1_0_IMAGENET_whole.pth',
                    './models_info/saved_models/vgg16_IMAGENET_whole.pth',
                    './models_info/saved_models/inception_v3_IMAGENET_whole.pth',
                    './models_info/saved_models/resnext101_32x8d_IMAGENET_whole.pth',
                    './models_info/saved_models/vit_b_16_IMAGENET_whole.pth']

class SiLULRPRule(EpsilonRule):
    """Attach ε-rule behaviour to nn.SiLU (a.k.a. Swish)."""
    

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

        if relevance.dim() == 4:
            batch_sum = relevance.sum(dim=(0, 2, 3)).detach().cpu()
        else:
            batch_sum = relevance.sum(dim=0).detach().cpu()
        
        layer_relevance_scores[name] += batch_sum
    num_samples += images.size(0)
    
    # Normalize by total samples
    for name in layer_relevance_scores:
        layer_relevance_scores[name] /= num_samples
        
    return layer_relevance_scores

def eval_imagenet_all():
    # Load ImageNet dataset
    batch_size = 256
    trainloader, testloader, train_dataset, val_dataset, classes = load_ImageNet(batch_size=batch_size, root='/data/shenghao/dataset/ImageNet')
    # Load model
    for model_path in SAVED_MODEL_PATH:
        model, module_name, module = get_model(load_model_path=model_path)
        # Evaluate model
        acc, loss, f1 = eval_model_dataloder(model, testloader, device='cuda:0')
        print(f"Accuracy: {acc}, Loss: {loss}, F1 Score: {f1}")

def eval_imagenet_attr():
    # Load ImageNet dataset
    batch_size = 16
    trainloader, testloader, train_dataset, val_dataset, classes = load_ImageNet(batch_size=batch_size, root='/data/shenghao/dataset/ImageNet')
    model, module_name, module = get_model(load_model_path=SAVED_MODEL_PATH[3])
    image, labels = next(iter(testloader))
    layer_relevance_scores = get_relevance_scores_for_all_layers(model, image, labels , device='cpu', attribution_method='lc')
    return layer_relevance_scores

def weights_align():
    # Load ImageNet dataset
    batch_size = 16
    trainloader, testloader, train_dataset, val_dataset, classes = load_ImageNet(batch_size=batch_size, root='/data/shenghao/dataset/ImageNet')
    # model_pre, module_name, module = get_model(load_model_path=SAVED_MODEL_PATH[3])
    model_custom = ResNet152()
    # pretrained_dict = model_pre.state_dict()
    # custom_dict = model_custom.state_dict()
    image, labels = next(iter(testloader))
    print(image.shape)
    model_custom(image)
    
# layer_relevance_scores = eval_imagenet_attr()
weights_align()

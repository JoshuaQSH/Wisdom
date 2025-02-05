import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import torchvision
from torchvision import models
from torchvision import transforms

# Gradient-based attribution
from captum.attr import IntegratedGradients, InputXGradient, DeepLift, DeepLiftShap, GradientShap, GuidedBackprop, Deconvolution
# Perturbation-based attribution
from captum.attr import FeatureAblation, Occlusion, FeaturePermutation, ShapleyValueSampling
from captum.attr import LimeBase, Lime, KernelShap
from captum.attr import LRP
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule
from captum._utils.models.linear_model import SkLearnLinearModel

### 1- Loading the model and the dataset
def give_model_label(labels_path="/home/shenghao/torch-deepimportance/images/"):
    model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    # model = models.resnet18(weights='DEFAULT')
    model = model.eval()

    # Downloads the list of classes/labels for ImageNet dataset and reads them into the memory
    with open(labels_path + "imagenet_class_index.json") as json_data:
        idx_to_labels = json.load(json_data)
    
    return model, idx_to_labels

def test_one():
    # Defines transformers and normalizing functions for the image.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    model, idx_to_labels = give_model_label()
    # img = Image.open('./images/swan-3299528_1280.jpg')
    img = Image.open('/data/shenghao/dataset/ImageNet/train/n01532829/n01532829_82.JPEG')

    transformed_img = transform(img)

    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)

    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
    
    return model, pred_label_idx, predicted_label, transformed_img, input

def test_batch():
    # Defines transformers and normalizing functions for the image.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    model, idx_to_labels = give_model_label()
    images = []
    img = Image.open('/data/shenghao/dataset/ImageNet/train/n01532829/n01532829_{}.JPEG'.format(82))
    transformed_img = transform(img)
    images.append(transformed_img)
    img = Image.open('/data/shenghao/dataset/ImageNet/train/n01532829/n01532829_{}.JPEG'.format(13101))
    images.append(transform(img))
    img = Image.open('/data/shenghao/dataset/ImageNet/train/n01532829/n01532829_{}.JPEG'.format(13106))
    images.append(transform(img))
    img = Image.open('/data/shenghao/dataset/ImageNet/train/n01532829/n01532829_{}.JPEG'.format(1312))
    images.append(transform(img))
    img = Image.open('/data/shenghao/dataset/ImageNet/train/n01532829/n01532829_{}.JPEG'.format(1318))
    images.append(transform(img))
    
    input = torch.stack(images)
    input = transform_normalize(input)

    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    
    
    pred_label_idx.squeeze_()
    predicted_labels = [idx_to_labels[str(idx.item())][1] for idx in pred_label_idx]
    print('Predicted:', predicted_labels, '(', prediction_score.squeeze().tolist(), ')')

    # pred_label_idx.squeeze_()
    # predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    # print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
    
    return pred_label_idx, input

### 2- Gradient-based attribution
def compute_attr(model, pred_label_idx, transformed_img, input, name='swan', is_gaussian=False):
    
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)
    
    # All the intended attribution methods
    # attributions_list = ['IntegratedGradients', 
    #                         'InputXGradient', 
    #                         'DeepLift', 
    #                         'DeepLiftShap', 
    #                         'GradientShap', 
    #                         'GuidedBackprop', 
    #                         'Deconvolution', 
    #                         'FeatureAblation', 
    #                         'Occlusion', 
    #                         'FeaturePermutation',
    #                         'ShapleyValueSampling',
    #                         'KernelShap',
    #                         'LRP']
    
    # attributions_list = [ 'ShapleyValueSampling',
    #                         'KernelShap',
    #                         'LRP']
    
    attributions_list = ['LRP']

    attributions_dict = {}
    # n_methods = len(attributions_list)
    # n_cols = 4
    # n_rows = (n_methods + n_cols - 1) // n_cols
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    # axes = axes.flatten()
        
    for i, method in enumerate(attributions_list):
        attribution_method = globals()[method](model)
        if method == 'IntegratedGradients':
            attributions_dict[method] = attribution_method.attribute(input, target=pred_label_idx, n_steps=200)
        elif method == 'InputXGradient':
            attributions_dict[method] = attribution_method.attribute(input, target=pred_label_idx)
        elif method == 'DeepLift':
            attributions_dict[method] = attribution_method.attribute(input, target=pred_label_idx)
        elif method == 'DeepLiftShap':
            torch.manual_seed(0)
            np.random.seed(0)
            rand_img_dist = torch.cat([input * 0, input * 1])
            attributions_dict[method] = attribution_method.attribute(input, baselines=rand_img_dist, target=pred_label_idx)
        elif method == 'GradientShap':
            torch.manual_seed(0)
            np.random.seed(0)
            rand_img_dist = torch.cat([input * 0, input * 1])
            attributions_dict[method] = attribution_method.attribute(input,
                                            n_samples=50,
                                            stdevs=0.0001,
                                            baselines=rand_img_dist,
                                            target=pred_label_idx)
        elif method == 'GuidedBackprop':
            attributions_dict[method] = attribution_method.attribute(input, target=pred_label_idx)
        elif method == 'Deconvolution':
            attributions_dict[method] = attribution_method.attribute(input, target=pred_label_idx)
        elif method == 'FeatureAblation':
            attributions_dict[method] = attribution_method.attribute(input, target=pred_label_idx)
        elif method == 'Occlusion':
            attributions_dict[method] = attribution_method.attribute(input,
                                        strides = (3, 50, 50),
                                        target=pred_label_idx,
                                        sliding_window_shapes=(3,60, 60),
                                        baselines=0)
        elif method == 'FeaturePermutation':
            pred_label_idx, input_batch = test_batch()
            attributions_dict[method] = attribution_method.attribute(input_batch, target=pred_label_idx)
        elif method == 'ShapleyValueSampling':
            attributions_dict[method] = attribution_method.attribute(input, target=pred_label_idx, feature_mask=None)
        elif method == 'KernelShap':
            attributions_dict[method] = attribution_method.attribute(input, target=pred_label_idx)
        elif method == 'LRP':
            # layers = list(model._modules["features"]) + list(model._modules["classifier"])
            layers = list(model._modules["features"])
            num_layers = len(layers)
            for idx_layer in range(1, num_layers):
                if idx_layer <= 16:
                    setattr(layers[idx_layer], "rule", GammaRule())
                elif 17 <= idx_layer <= 30:
                    setattr(layers[idx_layer], "rule", EpsilonRule())
                elif idx_layer >= 31:
                    setattr(layers[idx_layer], "rule", EpsilonRule(epsilon=0))
            attribution_method = globals()[method](model)
            attributions_dict[method] = attribution_method.attribute(input, target=pred_label_idx)
            
        plt_fig, plt_axis = viz.visualize_image_attr_multiple(np.transpose(attributions_dict[method].squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        show_colorbar=True,
                                        outlier_perc=2)
        print("Done with", method)
        plt.savefig('/home/shenghao/torch-deepimportance/images/{}_{}.pdf'.format(name, method),  format='pdf', dpi=1200)

        
        if is_gaussian:
            # `NoiseTunnel` to smooth the attributions using SmoothGrad and visualize the results.
            noise_tunnel = NoiseTunnel(attributions_dict[method])
            attributions_nt = noise_tunnel.attribute(input, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
            plt_fig, plt_axis = viz.visualize_image_attr_multiple(np.transpose(attributions_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                ["original_image", "heat_map"],
                                                ["all", "positive"],
                                                cmap=default_cmap,
                                                show_colorbar=True)
            plt.savefig('/home/shenghao/torch-deepimportance/images/{}_noise.pdf'.format(name), format='pdf', dpi=1200)
    
    # for j in range(i + 1, len(axes)):
    #     fig.delaxes(axes[j])
    # plt.savefig('/home/shenghao/torch-deepimportance/images/{}_all.pdf'.format(name),  format='pdf', dpi=1200)

            

def others(model, pred_label_idx, transformed_img, input, name='swan', is_gaussian=False):
    
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)
    
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)

    plt_fig, plt_axis = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                method='heat_map',
                                cmap=default_cmap,
                                show_colorbar=True,
                                sign='positive',
                                outlier_perc=1)
    plt.savefig('/home/shenghao/torch-deepimportance/images/{}_ig.png'.format(name))
    

    # Finally, let us use `GradientShap`, 
    # a linear explanation model which uses a distribution of reference samples (in this case two images) 
    # to explain predictions of the model. It computes the expectation of 
    # gradients for an input which was chosen randomly between the input and a baseline. 
    # The baseline is also chosen randomly from given baseline distribution.
    torch.manual_seed(0)
    np.random.seed(0)

    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([input * 0, input * 1])

    attributions_gs = gradient_shap.attribute(input,
                                            n_samples=50,
                                            stdevs=0.0001,
                                            baselines=rand_img_dist,
                                            target=pred_label_idx)
    plt_fig, plt_axis = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "absolute_value"],
                                        cmap=default_cmap,
                                        show_colorbar=True)
    plt.savefig('/home/shenghao/torch-deepimportance/images/{}_demo_3.png'.format(name))


    # Occlusion-based attribution
    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(input,
                                        strides = (3, 8, 8),
                                        target=pred_label_idx,
                                        sliding_window_shapes=(3,15,15),
                                        baselines=0)

    plt_fig, plt_axis = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        show_colorbar=True,
                                        outlier_perc=2)
    plt.savefig('/home/shenghao/torch-deepimportance/images/{}_demo_4.png'.format(name))

    occlusion = Occlusion(model)
    # larger window sizes can be used to get a more global view of the importance of different parts of the image
    attributions_occ = occlusion.attribute(input,
                                        strides = (3, 50, 50),
                                        target=pred_label_idx,
                                        sliding_window_shapes=(3,60, 60),
                                        baselines=0)
    plt_fig, plt_axis = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        show_colorbar=True,
                                        outlier_perc=2)
    plt.savefig('/home/shenghao/torch-deepimportance/images/{}_demo_4-2.png'.format(name))


### 4- LRP-based attribution
# Now let's try a different approach called Layer-Wise Relevance Propagation (LRP). 
# It uses a backward propagation mechanism applied sequentially to 
# all layers of the model, to see which neurons contributed to the output. 
# The output score of LRP represents the relevance, decomposed into values for each layer. 
# The decomposition is defined by rules that may vary for each layer. 
# Initially, we apply a direct implementation of LRP attribution. The default Epsilon-Rule is used for each layer. 
# Note: We use the VGG16 model instead here since the default rules for LRP are not fine-tuned for ResNet currently.
def lrp_vgg(pred_label_idx, transformed_img, input, name='swan'):
    model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    lrp = LRP(model)

    attributions_lrp = lrp.attribute(input, target=pred_label_idx)

    # Let us visualize the attribution, focusing on the areas with positive attribution (those that are critical for the classifier's decision):
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_lrp.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        show_colorbar=True,
                                        outlier_perc=2)
    plt.savefig('/home/shenghao/torch-deepimportance/images/{}_demo_5.png'.format(name))

    # Now let's play around with changing the propagation rules for the various layers. 
    # This is a crucial step to get expressive heatmaps. 
    # Captum currently has the following propagation rules implemented: LRP-Epsilon, LRP-0, LRP-Gamma, LRP-Alpha-Beta, and the Identity-Rule. 
    # In the next steps, we list all the layers of VGG16 and assign a rule to each one. 
    # Note: Reference for recommmendations on how to set the rules can be found in 
    # *[Towards best practice in explaining neural network decisions with LRP](https://arxiv.org/abs/1910.09840)*.
    layers = list(model._modules["features"]) + list(model._modules["classifier"])
    num_layers = len(layers)

    for idx_layer in range(1, num_layers):
        if idx_layer <= 16:
            setattr(layers[idx_layer], "rule", GammaRule())
        elif 17 <= idx_layer <= 30:
            setattr(layers[idx_layer], "rule", EpsilonRule())
        elif idx_layer >= 31:
            setattr(layers[idx_layer], "rule", EpsilonRule(epsilon=0))

    lrp = LRP(model)
    attributions_lrp = lrp.attribute(input, target=pred_label_idx)
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_lrp.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        show_colorbar=True,
                                        outlier_perc=2)
    plt.savefig('/home/shenghao/torch-deepimportance/images/{}_demo_5-2.png'.format(name))

if __name__ == "__main__":
    model, pred_label_idx, predicted_label, transformed_img, input = test_one()
    compute_attr(model, pred_label_idx, transformed_img, input, name=predicted_label)
    # lrp_vgg(pred_label_idx, transformed_img, input, name='bird')
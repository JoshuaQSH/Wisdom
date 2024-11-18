import os
import json
import urllib
import urllib.request
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

import torch
import torchvision
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune

from torchvision import models
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerActivation, InternalInfluence, LayerGradientXActivation, LayerGradCam, LayerDeepLift, LayerDeepLiftShap, LayerGradientShap, LayerIntegratedGradients, LayerFeatureAblation, LayerLRP
from captum.metrics import infidelity_perturb_func_decorator, infidelity

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from utils import load_CIFAR, load_MNIST, load_ImageNet, get_class_data, parse_args, get_model, load_kmeans_model, save_kmeans_model, Logger

def ramdon_prune_fc(model, layer_name='fc1', num_neurons=5):
    net_layer = getattr(model, layer_name)
    amount = num_neurons / net_layer.weight.shape[0]
    prune.random_unstructured(net_layer, name="weight", amount=amount)
    print(
        "Sparsity in weight: {:.2f}%".format(
            100. * float(torch.sum(net_layer.weight == 0))
            / float(net_layer.weight.nelement())
        )
    )
    
def prune_neurons_fc(model, layer_name='fc1', neurons_to_prune=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    layer = getattr(model, layer_name)
    
    with torch.no_grad():
        if isinstance(layer, nn.Linear):
            # Set weights and biases of the selected neurons to zero
            layer.weight[neurons_to_prune, :] = 0
            if layer.bias is not None:
                layer.bias[neurons_to_prune] = 0
        else:
            raise ValueError(f"Pruning is only implemented for Linear layers. Given: {type(layer)}")
    print(
        "Sparsity in weight: {:.2f}%".format(
            100. * float(torch.sum(layer.weight == 0))
            / float(layer.weight.nelement())
        )
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
    
    # ['LayerConductance', 'LayerActivation', 'InternalInfluence', 
    # 'LayerGradientXActivation', 'LayerGradCam', 'LayerDeepLift', 
    # 'LayerDeepLiftShap', 'LayerGradientShap', 'LayerIntegratedGradients', 
    # 'LayerFeatureAblation', 'LayerLRP']
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

def select_top_neurons(importance_scores, top_m_neurons=5):
    if top_m_neurons == -1:
        print("Selecting all neurons.")
        if importance_scores.dim() == 1:
            _, indices = torch.sort(importance_scores, descending=True)
            return indices
        else:
            return None
    else:
        if importance_scores.dim() == 1:
            print("Selecting top {} neurons (FC layer).".format(top_m_neurons))
            _, indices = torch.topk(importance_scores, top_m_neurons)
            return indices
        else:
            # TODO: Conv by default will select all the neurons
            print("Selecting all the neurons (Conv2D layer).")
            return None

def prepare_data(args):
    ### Dataset settings
    if args.dataset == 'cifar10':
        trainloader, testloader, classes = load_CIFAR(batch_size=args.batch_size, root=args.data_path, large_image=args.large_image)
    elif args.dataset == 'mnist':
        trainloader, testloader, classes = load_MNIST(batch_size=args.batch_size, root=args.data_path)
    elif args.dataset == 'imagenet':
        # batch_size=32, root='/data/shenghao/dataset/ImageNet', num_workers=2, use_val=False
        trainloader, testloader, classes = load_ImageNet(batch_size=args.batch_size, 
                                                         root=args.data_path + '/ImageNet', 
                                                         num_workers=2, 
                                                         use_val=False)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    return trainloader, testloader, classes

def get_activation_values_for_neurons(model, inputs, labels, important_neuron_indices, layer_name='fc1', visualize=False):
    
    activation_values = []
    print("Getting the Class: {}".format(labels))
    
    # Define a forward hook to capture the activations
    def hook_fn(module, input, output):
        activation_values.append(output.detach())
    
    # Register the hook on the specified layer
    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    
    if handle is None:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    
    # Remove the hook
    handle.remove()
        
    # Concatenate all activation values into one tensor
    activation_values = torch.cat(activation_values, dim=0)
    if important_neuron_indices is None:
        selected_activations = activation_values
    elif type(important_neuron_indices) == torch.Tensor and important_neuron_indices.shape[0] > 0:
        # Select only the activations for the important neurons
        selected_activations = activation_values[:, important_neuron_indices]
    else:
        raise ValueError(f"Invalid important_neuron_indices: {important_neuron_indices}")
            
    return activation_values, selected_activations

def find_optimal_clusters(scores, min_k=2, max_k=10):
    scores_np = scores.cpu().detach().numpy().reshape(-1, 1)
    silhouette_list = []
    for n_clusters  in range(min_k, max_k):
        
        # Initialize the clusterer with n_clusters value and a random generator
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(scores_np)
        
        silhouette_avg = silhouette_score(scores_np, cluster_labels)
        silhouette_list.append(silhouette_avg)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        sample_silhouette_values = silhouette_samples(scores_np, cluster_labels)
        
        #  Plot the cluster info
        # plot_cluster_info(n_clusters, silhouette_avg, scores_np, clusterer, cluster_labels)
        
        ## Get the samples for each cluster
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
    best_k = silhouette_list.index(max(silhouette_list)) + min_k
    return best_k

def cluster_activation_values(activation_values, n_clusters, layer_name='fc1', use_silhouette=False):
    
    kmeans_comb = []
    
    if layer_name[:-1] == 'fc':
        n_neurons = activation_values.shape[1]
        # kmeans = KMeans(n_clusters=optimal_k, random_state=0)
        for i in range(n_neurons):
            if use_silhouette:
                optimal_k = find_optimal_clusters(activation_values, 2, 10)
            else:
                optimal_k = n_clusters
                
            kmeans_comb.append(KMeans(n_clusters=optimal_k).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)))
        
        # kmeans_comb = [KMeans(n_clusters=optimal_k).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)) for i in range(n_neurons)]
        # cluster_labels = kmeans.fit_predict(activation_values_np)
        
    elif layer_name[:-1] == 'conv':
        activation_values = torch.mean(activation_values, dim=[2, 3])
        n_neurons = activation_values.shape[1]
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        for i in range(n_neurons):
            if use_silhouette:
                optimal_k = find_optimal_clusters(activation_values, 2, 10)
            else:
                optimal_k = n_clusters
                
            kmeans_comb.append(KMeans(n_clusters=optimal_k).fit(activation_values[:, i].cpu().numpy().reshape(-1, 1)))
    else:
        raise ValueError(f"Invalid layer name: {layer_name}")
            
    return kmeans_comb

def offer_kmeans_model(args, model, images, labels, classes, module_name):
    
    # Get the importance scores - LRP
    if args.capture_all:
        for name in module_name[1:]:
            attribution, mean_attribution = get_layer_conductance(model, images, labels, classes, 
                                                                layer_name=name, 
                                                                top_m_images=-1, 
                                                                attribution_method=args.attr)
    else:
        ### Main entrance here
        attribution, mean_attribution = get_layer_conductance(model, images, labels, classes, 
                                                                  layer_name=module_name[args.layer_index], 
                                                                  top_m_images=-1, attribution_method=args.attr)
        
    # Obtain the important neuron indices 
    important_neuron_indices = select_top_neurons(mean_attribution, args.top_m_neurons)
    activation_values, selected_activations = get_activation_values_for_neurons(model, 
                                                                                images, 
                                                                                labels, 
                                                                                important_neuron_indices, 
                                                                                module_name[args.layer_index],
                                                                                args.viz)
    

    kmeans_comb = cluster_activation_values(selected_activations, args.n_clusters, module_name[args.layer_index], args.use_silhouette)
    return kmeans_comb

def test_model(model, inputs, labels):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    total_loss += loss.item() * inputs.size(0)
    _, predicted = torch.max(outputs, 1)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)
    accuracy = correct / total * 100
    
    return accuracy, total_loss

def test_attributions(model,
                      attribution_method,
                      top_m_neurons,
                      train_inputs, 
                      train_labels, 
                      test_inputs,
                      test_labels,
                      classes,
                      layer_name,
                      random_prune=False):

    # TODO: Now using the torch random prune, but should be pruned by the specific neurons
    if random_prune:
        ramdon_prune_fc(model, layer_name=layer_name, num_neurons=top_m_neurons)
    else:
        _, layer_importance_scores = get_layer_conductance(model, 
                                                           train_inputs, 
                                                           train_labels, 
                                                           classes, 
                                                           layer_name=layer_name, 
                                                           attribution_method=attribution_method)
        indices = select_top_neurons(layer_importance_scores, top_m_neurons)
        prune_neurons_fc(model, layer_name=layer_name, neurons_to_prune = indices)

    a_accuracy, a_total_loss = test_model(model, test_inputs, test_labels)
    
    # Restore the model state
    model.load_state_dict(original_state)
    print("Model restored to original state.")
    
    return a_accuracy, a_total_loss

    
if __name__ == '__main__':
    args = parse_args()

    ### Logger settings
    if args.logging:
        start_time = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
        saved_log_name = args.log_path + 'AttriTest-{}-{}-{}.log'.format(args.model, args.dataset, timestamp)
        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[=== Model: {}, Dataset: {}, Layers_Index: {} ==]".format(args.model, args.dataset, args.layer_index))
    else:
        log = None
    
    ### Model settings
    if args.model_path != 'None':
        model_path = args.model_path
    else:
        model_path = os.getenv("HOME") + '/torch-deepimportance/models_info/'
    model_path += args.saved_model
    model, module_name, module = get_model(model_name=args.model)
    model.load_state_dict(torch.load(model_path))
    
    ### Device settings    
    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    
    ### Data settings
    trainloader, testloader, classes = prepare_data(args)    
    
    ### Test all the model layer and also the each of the attribution methods
    if args.capture_all:
        # attributions = ['lc', 'la', 'ii', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp']
        attributions = ['ldl', 'lgs', 'lig', 'lfa', 'lrp']
        # Save the original model state (Before pruning)
        original_state = copy.deepcopy(model.state_dict())

        if args.dataset == 'cifar10':
            classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            for single_class in classes:
                train_images, train_labels = get_class_data(trainloader, classes, single_class)
                test_images, test_labels = get_class_data(testloader, classes, single_class)
                b_accuracy, b_total_loss = test_model(model, test_images, test_labels)
                if args.logging:
                    log.logger.info("Class: {} Before Accuracy: {:.2f}%, Loss: {:.2f}".format(single_class, b_accuracy, b_total_loss))
                else:
                    print("Class: {} Before Acc: {:.2f}%, Before Loss: {:.2f}".format(single_class, b_accuracy, b_total_loss))

                for attr in attributions:
                    a_accuracy, a_total_loss = test_attributions(model=model,
                        attribution_method=attr,
                        top_m_neurons=args.top_m_neurons,
                        train_inputs=train_images, 
                        train_labels=train_labels, 
                        test_inputs=test_images,
                        test_labels=test_labels,
                        classes=classes,
                        layer_name=module_name[args.layer_index],
                        random_prune=False)
                    
                    if args.logging:
                        log.logger.info("Attribution: {}, Accuracy: {:.2f}%, Loss: {:.2f}".format(attr, a_accuracy, a_total_loss))
                    else:
                        print("Attribution: {}, Accuracy: {:.2f}%, Loss: {:.2f}".format(attr, a_accuracy, a_total_loss))

                    # Restore the model state
                    model.load_state_dict(original_state)
                    print("Model restored to original state.")

                
        elif args.dataset == 'mnist':
            classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        elif args.dataset == 'imagenet':
            url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
            urllib.request.urlretrieve(url, "imagenet_labels.json")
            # Load the labels from the JSON file
            with open("imagenet_labels.json") as f:
                classes = json.load(f)
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")
    else:
        #  Get the specific class data
        train_images, train_labels = get_class_data(trainloader, classes, args.test_image)
        # kmeans_comb = offer_kmeans_model(args, model, images, labels, classes, module_name)
        # print("Kmeans Model done!")
        
        # Get the test data
        test_images, test_labels = get_class_data(testloader, classes, args.test_image)
        
        a_accuracy, a_total_loss = test_attributions(model=model,
                      attribution_method=args.attr,
                      top_m_neurons=args.top_m_neurons,
                      train_inputs=train_images, 
                      train_labels=train_labels, 
                      test_inputs=test_images,
                      test_labels=test_labels,
                      classes=classes,
                      layer_name=module_name[args.layer_index],
                      random_prune=False)

import os
import json
import urllib
import urllib.request
import time
import copy
import random
from pathlib import Path
import sys

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


# Add src directory to sys.path
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

from utils import load_CIFAR, load_MNIST, load_ImageNet, get_class_data, parse_args, get_model, load_kmeans_model, save_kmeans_model, Logger
from attribution import get_layer_conductance, get_relevance_scores_for_all_layers
from visualization import visualize_activation, plot_cluster_infos, visualize_idc_scores
from idc import IDC
from pruning_methods import ramdon_prune, prune_neurons

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
            mean_attribution = torch.mean(importance_scores, dim=[1, 2])
            if mean_attribution.shape[0] < top_m_neurons:
                print("Selecting all the neurons (Conv2D layer).")
                return None
            else:
                _, indices = torch.topk(mean_attribution, top_m_neurons)
                return indices

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
    if args.layer_by_layer:
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

    
    _, layer_importance_scores = get_layer_conductance(model, 
                                                           train_inputs, 
                                                           train_labels, 
                                                           classes, 
                                                           layer_name=layer_name, 
                                                           attribution_method=attribution_method)
    
    indices = select_top_neurons(layer_importance_scores, top_m_neurons)

    # TODO: Now using the torch random prune, but should be pruned by the specific neurons
    if random_prune:
        print("Random Pruning")
        ramdon_prune(model, layer_name=layer_name, neurons_to_prune=indices, num_neurons=top_m_neurons, sparse_prune=False)

    else:
        print("Important Pruning")
        prune_neurons(model, layer_name=layer_name, neurons_to_prune = indices)

    accuracy, total_loss = test_model(model, test_inputs, test_labels)
    
    return accuracy, total_loss, indices

    
if __name__ == '__main__':
    args = parse_args()

    ### Logger settings
    if args.logging:
        start_time = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
        if args.random_prune:
            saved_log_name = args.log_path + 'AttriTest-{}-{}-random-L{}-{}.log'.format(args.model, args.dataset, args.layer_index, timestamp)
        else:
            saved_log_name = args.log_path + 'AttriTest-{}-{}-L{}-{}.log'.format(args.model, args.dataset, args.layer_index, timestamp)
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
    if args.all_attr:
        # attributions = ['lc', 'la', 'ii', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp']
        attributions = ['lc', 'la', 'ii', 'ldl', 'lgs', 'lig', 'lfa', 'lrp']
        # attributions = ['lrp', 'ldl']
        
        ## Saved the index for comparing the Common Neurons across the attributions
        # {'lc': [1, 2, 3, 4, 5], 'la': [1, 2, 3, 4, 5], ...}
        attr_dict = {key: [] for key in attributions}
        
        # Save the original model state (Before pruning)
        original_state = copy.deepcopy(model.state_dict())

        # class -> attributions
        # 
        
        if args.dataset == 'cifar10':
            classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            ## Saved the index for comparing the Common Neurons across the classes
            # {'plane': [1, 2, 3, 4, 5], 'car': [1, 2, 3, 4, 5], ...}
            common_index_class = {key: [] for key in classes}
            
            for single_class in classes:
                train_images, train_labels = get_class_data(trainloader, classes, single_class)
                test_images, test_labels = get_class_data(testloader, classes, single_class)
                b_accuracy, b_total_loss = test_model(model, test_images, test_labels)
                if args.logging:
                    log.logger.info("Class: {} Before Accuracy: {:.2f}%, Loss: {:.2f}".format(single_class, b_accuracy, b_total_loss))
                else:
                    print("Class: {} Before Acc: {:.2f}%, Before Loss: {:.2f}".format(single_class, b_accuracy, b_total_loss))

                attr_index = []
                for attr in attributions:
                    a_accuracy, a_total_loss, index = test_attributions(model=model,
                        attribution_method=attr,
                        top_m_neurons=args.top_m_neurons,
                        train_inputs=train_images, 
                        train_labels=train_labels, 
                        test_inputs=test_images,
                        test_labels=test_labels,
                        classes=classes,
                        layer_name=module_name[args.layer_index],
                        random_prune=args.random_prune)
                    
                    if args.logging:
                        log.logger.info("Random Prune: {}, Attribution: {}, Accuracy: {:.2f}%".format(args.random_prune, attr, a_accuracy))
                        log.logger.info("The chosen index: {}".format(index))
                    else:
                        print("Random Prune: {}, Attribution: {}, Accuracy: {:.2f}%".format(args.random_prune, attr, a_accuracy))
                        print("The chosen index: {}".format(index))
                    
                    attr_index.append(index)
                    attr_dict[attr].append(index)

                    # Restore the model state
                    model.load_state_dict(original_state)
                    print("Model restored to original state.")
                
                # analyze the common index in each attribution method
                common_elements = set(attr_index[0].tolist()).intersection(*(set(tensor.tolist()) for tensor in attr_index[1:]))
                common_index_class[single_class] = list(common_elements)
            
            common_index_attr = {}
            for key, tensor_list in attr_dict.items():
                common_elements = set(tensor_list[0].tolist()).intersection(
                    *(set(tensor.tolist()) for tensor in tensor_list[1:])
                )
                common_index_attr[key] = list(common_elements)
            
            with open('log_attr_per_class_{}_{}.json'.format(module_name[args.layer_index], args.top_m_neurons), 'w') as log_file:
                json.dump(common_index_class, log_file, indent=4)
            with open('log_class_per_attr_{}_{}.json'.format(module_name[args.layer_index], args.top_m_neurons), 'w') as log_file:
                json.dump(common_index_attr, log_file, indent=4)
                                        
        elif args.dataset == 'mnist':
            classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            common_index = {}
            common_index = {key: [] for key in classes}
            
        elif args.dataset == 'imagenet':
            url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
            urllib.request.urlretrieve(url, "imagenet_labels.json")
            # Load the labels from the JSON file
            with open("imagenet_labels.json") as f:
                classes = json.load(f)
            
            common_index = {}
            common_index = {key: [] for key in classes}
            
            
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")    
    
    else:
        #  Get the specific class data
        train_images, train_labels = get_class_data(trainloader, classes, args.test_image)
        # kmeans_comb = offer_kmeans_model(args, model, images, labels, classes, module_name)
        # print("Kmeans Model done!")
        
        # Get the test data
        test_images, test_labels = get_class_data(testloader, classes, args.test_image)
        
        a_accuracy, a_total_loss, index = test_attributions(model=model,
                      attribution_method=args.attr,
                      top_m_neurons=args.top_m_neurons,
                      train_inputs=train_images, 
                      train_labels=train_labels, 
                      test_inputs=test_images,
                      test_labels=test_labels,
                      classes=classes,
                      layer_name=module_name[args.layer_index],
                      random_prune=False)

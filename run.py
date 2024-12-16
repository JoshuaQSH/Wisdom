import os
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import NeuronConductance
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerActivation, InternalInfluence, LayerGradientXActivation, LayerGradCam, LayerDeepLift, LayerDeepLiftShap, LayerGradientShap, LayerIntegratedGradients, LayerFeatureAblation, LayerLRP
from captum.metrics import infidelity_perturb_func_decorator, infidelity

# Add src directory to sys.path
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

from utils import load_CIFAR, load_MNIST, load_ImageNet, get_class_data, parse_args, get_model, get_model_cifar, get_trainable_modules_main, get_layer_by_name, test_random_class, Logger
from attribution import get_relevance_scores, get_relevance_scores_for_all_layers, get_relevance_scores_for_all_classes
from visualization import visualize_activation, plot_cluster_infos, visualize_idc_scores
from idc import IDC

def load_importance_scores(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return torch.tensor(data["importance_scores"]), torch.tensor(data["mean_importance"]), data["class_label"]

def save_importance_scores(importance_scores, mean_importance, filename, class_label):
    scores = importance_scores.cpu().detach().numpy().tolist()
    mean_scores = mean_importance.cpu().detach().numpy().tolist()
    data = {
        "class_label": class_label,
        "importance_scores": scores,
        "mean_importance": mean_scores
    }

    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Importance scores saved to {filename}")

if __name__ == '__main__':
    args = parse_args()
    
    ### Logger settings
    if args.logging:
        start_time = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
        saved_log_name = args.log_path + 'End2endIDC-{}-{}-{}-{}.log'.format(args.model, args.dataset, args.test_image, timestamp)
        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[=== Model: {}, Dataset: {}, Layers_Index: {}, Topk: {} ==]".format(args.model, args.dataset, args.layer_index, args.top_m_neurons))
    else:
        log = None
    
    ### Model settings
    if args.model_path != 'None':
        model_path = args.model_path
    # A hack here for debugging    
    elif args.dataset == 'cifar10' and args.model == 'lenet':
        args.model_path = os.getenv("HOME") + '/torch-deepimportance/models_info/saved_models/'
        model_path = os.getenv("HOME") + '/torch-deepimportance/models_info/saved_models/'
    else:
        model_path = os.getenv("HOME") + '/torch-deepimportance/models_info/saved_models/'
    model_path += args.saved_model
    
    ### Dataset settings
    if args.dataset == 'cifar10':
        trainloader, testloader, test_dataset, classes = load_CIFAR(batch_size=args.batch_size, root=args.data_path, large_image=args.large_image)
    elif args.dataset == 'mnist':
        trainloader, testloader, test_dataset, classes = load_MNIST(batch_size=args.batch_size, root=args.data_path)
    elif args.dataset == 'imagenet':
        # batch_size=32, root='/data/shenghao/dataset/ImageNet', num_workers=2, use_val=False
        trainloader, testloader, test_dataset, classes = load_ImageNet(batch_size=args.batch_size, 
                                                         root=args.data_path + '/ImageNet', 
                                                         num_workers=2, 
                                                         use_val=False)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    ### Device settings    
    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    
    if args.dataset == 'cifar10' and args.model != 'lenet':
        model, module_name, module = get_model_cifar(model_name=args.model, load_model_path=model_path)
    else:
        # We aussume that the SOTA models are pretrained with IMAGENET
        model, module_name, module = get_model(model_name=args.model)

    
    if args.dataset == 'mnist' and args.model == 'lenet':
        model.conv1 = torch.nn.Conv2d(1, 6, 5)
        model.fc1 = torch.nn.Linear(16 * 16, 120)
    # For one of the testing demo, we provide the pretrained LeNet5 with CIFAR10    
    if args.model_path != 'None' and args.model == 'lenet':
        model.load_state_dict(torch.load(model_path))

    trainable_module, trainable_module_name = get_trainable_modules_main(model)

    ### Task 1: Data and Model Preparation
    if args.all_class:
        images, labels = None, None
    else:
        images, labels = get_class_data(trainloader, classes, args.test_image)
    
    # Get the importance scores - TODO: use_saved_importance is just for debugging
    use_saved_importance = False
    if os.path.exists(args.importance_file) and use_saved_importance:
        print("Obtaining the importance scores from the file.")
        attribution, mean_attribution, labels = load_importance_scores(args.importance_file)
    else:
        if args.layer_by_layer and not args.end2end:
            # for name in module_name[1:]:
            for i, name in enumerate(trainable_module_name):
                attribution, mean_attribution = get_relevance_scores(model, images, labels, classes,
                                                                      net_layer=trainable_module[i],
                                                                      layer_name=name, 
                                                                      top_m_images=-1, 
                                                                      attribution_method=args.attr)
                filename = args.importance_file.replace('.json', f'_{name}.json')
                save_importance_scores(attribution, mean_attribution, filename, args.test_image)
                print("{} Saved".format(filename))
       
        elif args.end2end:
            print("Relevance scores for all layers obtained.")
            attribution = get_relevance_scores_for_all_layers(model, images, labels, attribution_method=args.attr)
        
        elif args.all_class:
            print("Relevance scores for all classes obtained.")
            # model, dataloader, net_layer, layer_name='fc1', attribution_method='lrp'
            mean_attribution = get_relevance_scores_for_all_classes(model, trainloader, 
                                                               net_layer=trainable_module[args.layer_index],
                                                               layer_name=trainable_module_name[args.layer_index], 
                                                               attribution_method=args.attr)

        else:
            print("Relevance scores for the selected layer (and class) obtained.")
            attribution, mean_attribution = get_relevance_scores(model, images, labels, classes,
                                                                  net_layer=trainable_module[args.layer_index],
                                                                  layer_name=trainable_module_name[args.layer_index], 
                                                                  top_m_images=-1, attribution_method=args.attr)
            # save_importance_scores(attribution, mean_attribution, args.importance_file, args.test_image)

    if args.all_class:
        # Sample the testset data for the IDC coverage
        subset_loader, test_images, test_labels = test_random_class(test_dataset, test_all=args.idc_test_all, num_samples=args.num_samples)
    else:
        test_images, test_labels = get_class_data(testloader, classes, args.test_image)    
    
    ### Task 2-1: prepare and test the IDC coverage
    idc = IDC(model, classes, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class)

    if args.end2end:
        important_neuron_indices = idc.select_top_neurons_all(attribution)
        activation_values, selected_activations = idc.get_activation_values_for_model(images, classes[labels[0]], important_neuron_indices)
    else:
        # Obtain the important neuron indices
        important_neuron_indices = idc.select_top_neurons(mean_attribution)
        if args.all_class:
            activation_values, selected_activations = idc.get_activation_values_for_neurons(images, 
                                                                                            important_neuron_indices, 
                                                                                            trainable_module_name[args.layer_index],
                                                                                            trainloader)
        else:
            activation_values, selected_activations = idc.get_activation_values_for_neurons(images,
                                                                                    important_neuron_indices, 
                                                                                    trainable_module_name[args.layer_index])
    
    ### Task 2-2: Cluster the activation values
    if args.end2end:
        kmeans_comb = idc.cluster_activation_values_all(selected_activations)
    else:    
        kmeans_comb = idc.cluster_activation_values(selected_activations, trainable_module[args.layer_index])
    print("Activation values clustered.")

    ### Evaluate the attribution methods
    if args.vis_attributions:
        # List of available attribution methods
        attribution_methods = [
            'lc', 'la', 'ii', 'lgxa', 'lgc', 'ldl', 
            'ldls', 'lgs', 'lig', 'lfa', 'lrp'
        ]
        idc.evaluate_attribution_methods(
                                    test_images, 
                                    test_labels, 
                                    kmeans_comb, 
                                    module_name[args.layer_index], 
                                    attribution_methods)
    
    else:
        ### Compute IDC coverage
        # End to end testing for the whole model
        if args.end2end:
            unique_cluster, coverage_rate = idc.compute_idc_test_whole(test_images, 
                            test_labels,
                            important_neuron_indices,
                            kmeans_comb,
                            args.attr)
            if log:
                log.logger.info("Class: {}".format(args.test_image))
                log.logger.info("Testing Samples: {}, IDC Coverage: {}, Attribution: {}".format(len(test_images), coverage_rate, args.attr))
            else:
                print("Number of Testing Samples: {}".format(len(test_images)))


        else:
            # Test the specific layer
            # inputs_images, labels, indices, kmeans, net_layer, layer_name, attribution_method='lrp'
            unique_cluster, coverage_rate = idc.compute_idc_test(inputs_images=test_images, 
                                                                 labels=test_labels,
                                                                 indices=important_neuron_indices, 
                                                                 kmeans=kmeans_comb,
                                                                 net_layer=trainable_module[args.layer_index],
                                                                 layer_name=trainable_module_name[args.layer_index],
                                                                 attribution_method=args.attr)
        
            if log:
                log.logger.info("Testing Samples: {}, Layner Name:{}, IDC Coverage: {}, Attribution: {}".format(len(test_images), 
                                                                                                                trainable_module_name[args.layer_index], 
                                                                                                                coverage_rate, args.attr))
            else:
                print("Testing Samples: {}, Layner Name:{}".format(len(test_images), trainable_module_name[args.layer_index]))
    
    ### Infidelity metric
    # infid = infidelity_metric(net, perturb_fn, images, attribution)
    # print(f"Infidelity: {infid:.2f}")
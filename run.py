import os
import json
import sys
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

from utils import load_CIFAR, load_MNIST, load_ImageNet, get_class_data, parse_args, get_model
from attribution import get_layer_conductance, get_relevance_scores_for_all_layers
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
    
    ### Model settings
    if args.model_path != 'None':
        model_path = args.model_path
    else:
        model_path = os.getenv("HOME") + '/torch-deepimportance/models_info/'
    model_path += args.saved_model
    
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
    
    ### Device settings    
    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    
    # We aussume that the SOAT models are pretrained with IMAGENET
    model, module_name, module = get_model(model_name=args.model)
    
    # For one of the testing demo, we provide the pretrained LeNet5 with CIFAR10
    if args.dataset == 'cifar10' and args.model == 'lenet':
        model.load_state_dict(torch.load(model_path))
    

    ### Task 1: Data and Model Preparation
    # Get the specific class data
    images, labels = get_class_data(trainloader, classes, args.test_image)
    
    breakpoint()
    
    # Get the importance scores - LRP
    if os.path.exists(args.importance_file):
        print("Obtaining the importance scores from the file.")
        attribution, mean_attribution, labels = load_importance_scores(args.importance_file)
    else:
        if args.layer_by_layer and not args.end2end:
            for name in module_name[1:]:
                attribution, mean_attribution = get_layer_conductance(model, images, labels, classes, 
                                                                      layer_name=name, 
                                                                      top_m_images=-1, 
                                                                      attribution_method=args.attr)
                filename = args.importance_file.replace('.json', f'_{name}.json')
                save_importance_scores(attribution, mean_attribution, filename, args.test_image)
                print("{} Saved".format(filename))
       
        elif args.end2end:
            print("Relevance scores for all layers obtained.")
            attribution = get_relevance_scores_for_all_layers(model, images, labels, attribution_method=args.attr)

        else:
            attribution, mean_attribution = get_layer_conductance(model, images, labels, classes, 
                                                                  layer_name=module_name[args.layer_index], 
                                                                  top_m_images=-1, attribution_method=args.attr)
            save_importance_scores(attribution, mean_attribution, args.importance_file, args.test_image)

    # Get the test data
    test_images, test_labels = get_class_data(testloader, classes, args.test_image)

    ### Task 2-1: prepare and test the IDC coverage
    idc = IDC(model, classes, args.top_m_neurons, args.n_clusters, args.use_silhouette)


    if args.end2end:
        important_neuron_indices = idc.select_top_neurons_all(attribution)
        activation_values, selected_activations = idc.get_activation_values_for_model(images, classes[labels[0]], important_neuron_indices)
    else:
        # Obtain the important neuron indices
        important_neuron_indices = idc.select_top_neurons(mean_attribution)
        activation_values, selected_activations = idc.get_activation_values_for_neurons(images, 
                                                                                    labels, 
                                                                                    important_neuron_indices, 
                                                                                    module_name[args.layer_index])
    

    ### Task 2-2: Cluster the activation values
    if args.end2end:
        kmeans_comb = idc.cluster_activation_values_all(selected_activations)
    else:    
        kmeans_comb = idc.cluster_activation_values(selected_activations, module_name[args.layer_index])
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
                            kmeans_comb,
                            args.attr)
        else:
            # Test the specific layer
            unique_cluster, coverage_rate = idc.compute_idc_test(test_images, 
                                                                 test_labels, 
                                                                 kmeans_comb, 
                                                                 module_name[args.layer_index],
                                                                 args.attr)
            
    ### Infidelity metric
    # infid = infidelity_metric(net, perturb_fn, images, attribution)
    # print(f"Infidelity: {infid:.2f}")
import os
import json
import sys
import time
from pathlib import Path

import torch
from captum.attr import visualization as viz
from captum.metrics import infidelity_perturb_func_decorator, infidelity

from src.utils import load_CIFAR, load_MNIST, load_ImageNet, get_class_data, parse_args, get_model, get_trainable_modules_main, test_model_dataloder, test_random_class, Logger
from src.attribution import get_relevance_scores, get_relevance_scores_for_all_layers, get_relevance_scores_for_all_classes
from src.idc import IDC

def load_importance_scores(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return torch.tensor(data["importance_scores"]), torch.tensor(data["mean_importance"]), data["class_label"]
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON in {filename}.")
        sys.exit(1)

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

def load_dataset(args):
    if args.dataset == 'cifar10':
        return load_CIFAR(batch_size=args.batch_size, root=args.data_path, large_image=args.large_image)
    elif args.dataset == 'mnist':
        return load_MNIST(batch_size=args.batch_size, root=args.data_path)
    elif args.dataset == 'imagenet':
        return load_ImageNet(batch_size=args.batch_size, 
                             root=os.path.join(args.data_path, 'ImageNet'), 
                             num_workers=2, 
                             use_val=False)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

if __name__ == '__main__':
    args = parse_args()
    
    ### Logger settings
    if args.logging:
        start_time = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
        saved_log_name = args.log_path + '-{}-{}-{}.log'.format(args.model, args.dataset, timestamp)
        
        # saved_log_name = args.log_path + 'End2endIDC-{}-{}-{}-{}.log'.format(args.model, args.dataset, args.test_image, timestamp)
        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[=== Model: {}, Dataset: {}, Layers_Index: {}, Topk: {} ==]".format(args.model, args.dataset, args.layer_index, args.top_m_neurons))
    else:
        log = None
    
    ### Model settings
    model_path = os.getenv("HOME") + args.saved_model
    
    ### Dataset settings
    trainloader, testloader, train_dataset, test_dataset, classes = load_dataset(args)

    ### Device settings
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    
    ### Model loading
    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)
    
    ### Model evaluation
    accuracy, avg_loss, f1 = test_model_dataloder(model, testloader, device)
    if log:
        log.logger.info("Model test Acc: {}, Loss: {}, F1 Score: {}".format(accuracy, avg_loss, f1))
        log.logger.info("Test Model Layer: {}".format(trainable_module[args.layer_index]))
    else:
        print("Model test Acc: {}, Loss: {}, F1 Score: {}".format(accuracy, avg_loss, f1))
        print("Test Model Layer: {}".format(trainable_module[args.layer_index]))
    
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
            print("Looping all the layers for Class: {}".format(args.test_image))
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
            print("Relevance scores for all layers.")
            attribution = get_relevance_scores_for_all_layers(model, images, labels, device, attribution_method=args.attr)
        
        elif args.all_class:
            print("Relevance scores for all classes.")
            # model, dataloader, net_layer, layer_name='fc1', attribution_method='lrp'
            mean_attribution = get_relevance_scores_for_all_classes(model, trainloader, 
                                                               net_layer=trainable_module[args.layer_index],
                                                               layer_name=trainable_module_name[args.layer_index], 
                                                               attribution_method=args.attr)

        else:
            print("Relevance scores for Layer: {} and Class {}".format(trainable_module_name[args.layer_index], args.test_image))
            attribution, mean_attribution = get_relevance_scores(model, images, labels, classes,
                                                                  net_layer=trainable_module[args.layer_index],
                                                                  layer_name=trainable_module_name[args.layer_index], 
                                                                  top_m_images=-1, attribution_method=args.attr)
            filename = args.importance_file.replace('.json', f'{trainable_module_name[args.layer_index]}_{args.test_image}.json')
            save_importance_scores(attribution, mean_attribution, filename, args.test_image)
            print("{} Saved".format(filename))

    if args.num_samples != 0:
        # Sample the testset data for the IDC coverage
        subset_loader, test_images, test_labels = test_random_class(test_dataset, test_all=args.idc_test_all, num_samples=args.num_samples)
    else:
        test_images, test_labels = get_class_data(testloader, classes, args.test_image)    
    
    ### Task 2-1: prepare and test the IDC coverage
    idc = IDC(model, classes, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class, "KMeans")

    if args.end2end:
        # TODO: the last attr should be the filtered layer's name, a hack here
        important_neuron_indices, inorderd_neuron_indices = idc.select_top_neurons_all(attribution, 'fc3')
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
        cluster_groups = idc.cluster_activation_values_all(selected_activations)
    else:    
        cluster_groups = idc.cluster_activation_values(selected_activations, trainable_module[args.layer_index])
    print("Activation values clustered.")

    ### Task 2-3: Compute IDC coverage
    if args.end2end:
        unique_cluster, coverage_rate = idc.compute_idc_test_whole(test_images, 
                        test_labels,
                        important_neuron_indices,
                        cluster_groups,
                        args.attr)
        if log:
            log.logger.info("Class: {}".format(args.test_image))
            log.logger.info("Testing Samples: {}, IDC Coverage: {}, Attribution: {}".format(len(test_images), coverage_rate, args.attr))
        else:
            print("Number of Testing Samples: {}".format(len(test_images)))

    else:
        # inputs_images, labels, indices, kmeans, net_layer, layer_name, attribution_method='lrp'
        unique_cluster, coverage_rate = idc.compute_idc_test(inputs_images=test_images, 
                                                                labels=test_labels,
                                                                indices=important_neuron_indices, 
                                                                cluster_groups=cluster_groups,
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
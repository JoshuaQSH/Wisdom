"""IDC Coverage Pipeline (Wisdom excluded)
========================

This script is designed to evaluate the coverage of a model's important neurons using the IDC (Important Neuron Coverage) method.
It includes the following steps:
1. Load the model and dataset.
2. Evaluate the model's performance on the test set.
3. Compute the importance scores for the model's neurons using various attribution methods.

Shenghao Qiu, 2025
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any

import torch

from src.attribution import (
    get_relevance_scores_dataloader,
    get_relevance_score_target_layer,
)
from src.idc import IDC
from src.utils import (
    _configure_logging,
    _select_testing_mode,
    eval_model_dataloder,
    get_model,
    get_trainable_modules_main,
    load_CIFAR,
    load_ImageNet,
    load_MNIST,
    parse_args,
    extract_class_to_dataloder,
)

# -----------------------------------------------------------------------------
# Constants & Type Aliases
# -----------------------------------------------------------------------------

DatasetLoaders = Dict[str, Any]

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def load_dataset(args):
    """Return `trainloader`, `testloader`, `train_dataset`, `test_dataset`, `classes`."""
    loaders: DatasetLoaders = {
        "cifar10": lambda: load_CIFAR(  # pylint: disable=unnecessary‑lambda
            batch_size=args.batch_size,
            root=args.data_path,
        ),
        "mnist": lambda: load_MNIST(
            batch_size=args.batch_size,
            root=args.data_path,
        ),
        "imagenet": lambda: load_ImageNet(
            batch_size=args.batch_size,
            root=os.path.join(args.data_path, 'ImageNet'), 
            num_workers=2,
            use_val=False,
        ),
    }
    try:
        return loaders[args.dataset]()
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {args.dataset}") from exc

# -----------------------------------------------------------------------------
# Main Routine
# -----------------------------------------------------------------------------
def main() -> None:
    """Entry‑point for IDC coverage computation pipeline.
        --all-class: If set, the model will be tested for all classes, equal to batch testing.
        --class_iters: If set, the model will be tested for each class separately.
        --end2end: If set, the model will be tested end-to-end, getting all layers relevance.
    For now, we have four testing modes:
    1. End2End with all classes
    2. End2End with single class
    3. Single layer with all classes
    4. Single layer with single class
    """
    args = parse_args()
    logger = _configure_logging(args.logging, args, 'debug')
    
    # ------------------------------------------------------------------
    # Model & Dataset setup
    # ------------------------------------------------------------------
    model_path = os.getenv("HOME") + args.saved_model
 
    trainloader, testloader, train_dataset, test_dataset, classes = load_dataset(args)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)
    
    testing_mode, mode_descriptions = _select_testing_mode(args)    
    logger.info("Model: %s, Dataset: %s, Topk: %s, Testing Mode: [%s]", args.model, args.dataset, args.top_m_neurons, ', '.join(mode_descriptions))

    accuracy, avg_loss, f1_score = eval_model_dataloder(model, testloader, device)
    logger.info("Test accuracy: %.4f,  loss: %.4f, F1 Score: %.4f", accuracy, avg_loss, f1_score)
        
    # ------------------------------------------------------------------
    # Task 1: Attribution & importance scores
    # ------------------------------------------------------------------
    if not testing_mode['all_class'] and not testing_mode['class_iters']:
        logger.info("Testing single class: %s", args.test_image)
        # trainloader = extract_class_to_dataloder(train_dataset, classes, args.batch_size, args.test_image)
        testloader = extract_class_to_dataloder(test_dataset, classes, args.batch_size, args.test_image)
    
    final_layer = trainable_module_name[-1]
    logger.info("Final layer: %s", final_layer)
        
    if testing_mode['end2end']:
        if testing_mode['all_class']:
            logger.info("End2End Testing with all classes. Attribution Method: %s", args.attr)
        else:
            logger.info("End2End Testing with single class: %s. Attribution Method: %s", args.test_image, args.attr) 
        mean_attribution = get_relevance_scores_dataloader(model, trainloader, device, attribution_method=args.attr)
    else:
        logger.info("Testing with single layer: %s. Attribution Method: %s", trainable_module_name[args.layer_index], args.attr)
        if testing_mode['all_class']:
            logger.info("Single Layer Testing with all classes.")
        else:
            logger.info("Single Layer Testing with single class: %s", args.test_image)
        mean_attribution = get_relevance_score_target_layer(model, trainloader, device, attribution_method=args.attr, target_layer=trainable_module[args.layer_index])
    
    # ------------------------------------------------------------------
    # Task 2-2: Select & cluster activations & Testing IDC coverage
    # ------------------------------------------------------------------
    
    idc = IDC(
        model,
        classes,
        args.top_m_neurons,
        args.n_clusters,
        args.use_silhouette,
        args.all_class,
        "KMeans",
    )
    
    if testing_mode['end2end']:
        logger.info("End2End Testing")
        important_neuron_indices, inorderd_indices = idc.select_top_neurons_all(mean_attribution, final_layer)
        activation_values, selected_activations = idc.get_activations_model_dataloader(
            trainloader,
            important_neuron_indices,
        )
        
        # Cluster activation values for all layers
        logger.info("Clustering activation values for all layers.")
        cluster_groups = idc.cluster_activation_values_all(selected_activations)
                
        if not testing_mode['all_class'] and not testing_mode['class_iters']:
            logger.info("Testing Samples: %s", args.test_image)
            coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(testloader, important_neuron_indices, cluster_groups)
            logger.info("Attribution Method: %s", args.attr)
            logger.info("Total INCC combinations: %d", total_combination)
            logger.info("Max Coverage (the best we can achieve): %.6f%%", max_coverage * 100)
            logger.info("IDC Coverage: %.6f%%", coverage_rate * 100)
        
        elif not testing_mode['all_class'] and testing_mode['class_iters']:
            logger.info("Iterating over all classes for IDC coverage computation.")
            results_dict = {}
            logger.info("Attribution Method: %s", args.attr)
            
            for test_class in classes:
                testloader_iter = extract_class_to_dataloder(test_dataset, classes, args.batch_size, test_class)
                coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(testloader_iter, important_neuron_indices, cluster_groups)
                logger.info("IDC Coverage: %.6f%%", coverage_rate * 100)
                idc.total_combination = 1
                
                # Save results for this test class
                results_dict[test_class] = {
                    'Test Class': test_class,
                    'Total Combination': total_combination,
                    'Max Coverage': max_coverage,
                    'Coverage Rate': coverage_rate
                }

            # Save results to JSON file
            json_filename = f"idc_results_{args.model}_{args.dataset}_class_iters.json"
            with open(json_filename, 'w') as f:
                json.dump(results_dict, f, indent=4)
            logger.info("Results saved to %s", json_filename)

        else:
            logger.info("Testing all classes.")
            # IDC coverage computation for end-to-end testing - dataloader
            coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(testloader, important_neuron_indices, cluster_groups)
            logger.info("Attribution Method: %s", args.attr)
            logger.info("Total INCC combinations: %d", total_combination)
            logger.info("Max Coverage (the best we can achieve): %.6f%%", max_coverage * 100)
            logger.info("IDC Coverage: %.6f%%", coverage_rate * 100)
        
    else:
        logger.info("Tested Layer: %s", trainable_module_name[args.layer_index])
        important_neuron_indices = idc.select_top_neurons(mean_attribution)
        activation_values, selected_activations = idc.get_activations_neurons_dataloader(trainloader, trainable_module_name[args.layer_index], important_neuron_indices)
        
        # Cluster activation values for a specific layer
        logger.info("Clustering activation values for layer: %s", trainable_module_name[args.layer_index])
        cluster_groups = idc.cluster_activation_values(
            selected_activations,
            trainable_module[args.layer_index],
        )
        
        if not testing_mode['all_class'] and not testing_mode['class_iters']:
            logger.info("Testing Samples: %s", args.test_image)
        
            # IDC coverage computation for a specific layer - dataloader
            coverage_rate, total_combination, max_coverage = idc.compute_idc_test_dataloader(
                dataloader=testloader,
                indices=important_neuron_indices,
                cluster_groups=cluster_groups,
                net_layer=trainable_module[args.layer_index],
                layer_name=trainable_module_name[args.layer_index],
            )
            
            logger.info("Attribution Method: %s", args.attr)
            logger.info("Total INCC combinations: %d", total_combination)
            logger.info("Max Coverage (the best we can achieve): %.6f%%", max_coverage * 100)
            logger.info("IDC Coverage: %.6f%%", coverage_rate * 100)
        
        elif not testing_mode['all_class'] and testing_mode['class_iters']:
            logger.info("Iterating over all classes for IDC coverage computation.")
            results_dict = {}
            logger.info("Attribution Method: %s", args.attr)
            
            for test_class in classes:
                testloader_iter = extract_class_to_dataloder(test_dataset, classes, args.batch_size, test_class)
                # IDC coverage computation for a specific layer - dataloader
                coverage_rate, total_combination, max_coverage = idc.compute_idc_test_dataloader(
                    dataloader=testloader_iter,
                    indices=important_neuron_indices,
                    cluster_groups=cluster_groups,
                    net_layer=trainable_module[args.layer_index],
                    layer_name=trainable_module_name[args.layer_index],
                )
                idc.total_combination = 1
                logger.info("IDC Coverage: %.6f%%", coverage_rate * 100)
                
                # Save results for this test class
                results_dict[test_class] = {
                    'Test Class': test_class,
                    'Test Layer': trainable_module_name[args.layer_index],
                    'Total Combination': total_combination,
                    'Max Coverage': max_coverage,
                    'Coverage Rate': coverage_rate
                }

            # Save results to JSON file
            json_filename = f"idc_results_{args.model}_{args.dataset}_class_iters.json"
            with open(json_filename, 'w') as f:
                json.dump(results_dict, f, indent=4)
            logger.info("Results saved to %s", json_filename)
        
        else:
            logger.info("Testing all classes.")
            # IDC coverage computation for a specific layer - dataloader
            coverage_rate, total_combination, max_coverage = idc.compute_idc_test_dataloader(
                dataloader=testloader,
                indices=important_neuron_indices,
                cluster_groups=cluster_groups,
                net_layer=trainable_module[args.layer_index],
                layer_name=trainable_module_name[args.layer_index],
            )
            logger.info("Attribution Method: %s", args.attr)
            logger.info("Total INCC combinations: %d", total_combination)
            logger.info("Max Coverage (the best we can achieve): %.6f%%", max_coverage * 100)
            logger.info("IDC Coverage: %.6f%%", coverage_rate * 100)
    logger.info("IDC coverage computation completed.")

if __name__ == "__main__":
    main()
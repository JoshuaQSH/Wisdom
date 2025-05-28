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

import logging
import os
import time
from typing import Tuple, Sequence, Optional, Dict, Any, List

import torch

from captum.attr import visualization as viz  # noqa: F401  # Imported for completeness.
from captum.metrics import (  # noqa: F401  # Imported for completeness.
    infidelity,
    infidelity_perturb_func_decorator,
)

from src.attribution import (
    get_relevance_scores,
    get_relevance_scores_for_all_classes,
    get_relevance_scores_for_all_layers,
)
from src.idc import IDC
from src.utils import (
    _configure_logging,
    _select_testing_mode,
    eval_model_dataloder,
    extract_random_class,
    get_class_data,
    get_model,
    get_trainable_modules_main,
    load_CIFAR,
    load_ImageNet,
    load_MNIST,
    parse_args,
)

# -----------------------------------------------------------------------------
# Constants & Type Aliases
# -----------------------------------------------------------------------------

DatasetLoaders = Dict[str, Any]
USE_QUICK_ITER_DATA = False

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def load_dataset(args):
    """Return `trainloader`, `testloader`, `train_dataset`, `test_dataset`, `classes`."""
    loaders: DatasetLoaders = {
        "cifar10": lambda: load_CIFAR(  # pylint: disable=unnecessary‑lambda
            batch_size=args.batch_size,
            root=args.data_path,
            large_image=args.large_image,
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
    """Entry‑point for IDC coverage computation pipeline."""
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
    
    testing_mode = _select_testing_mode(args)
    logger.info("Model: %s, Dataset: %s, Topk: %s, Testing Mode: [%s]", args.model, args.dataset, args.top_m_neurons, testing_mode)

    accuracy, avg_loss, f1_score = eval_model_dataloder(model, testloader, device)
    logger.info("Test accuracy: %.4f,  loss: %.4f, F1 Score: %.4f", accuracy, avg_loss, f1_score)
    
    # ------------------------------------------------------------------
    # Task 1: Attribution & importance scores
    # ------------------------------------------------------------------
    if args.all_class:
        images, labels = None, None
    else:
        if USE_QUICK_ITER_DATA:
            images, labels = next(iter(trainloader))
        else:
            images, labels = get_class_data(trainloader, classes, args.test_image)
    
    # Compute new importance scores
    if args.layer_by_layer and not args.end2end:
        logger.info("Computing relevance for *all* layers — class %s", args.test_image)
        for idx, name in enumerate(trainable_module_name):
            attribution, mean_attribution = get_relevance_scores(
                model,
                images,
                labels,
                classes,
                net_layer=trainable_module[idx],
                layer_name=name,
                top_m_images=-1,
                attribution_method=args.attr,
            )
    elif args.end2end:
        logger.info("End2End Testing - getting all layers relevance")
        attribution = get_relevance_scores_for_all_layers(
            model,
            images,
            labels,
            device,
            attribution_method=args.attr,
        )
    elif args.all_class:
        logger.info("Computing relevance for all classes — layer: %s", trainable_module_name[args.layer_index])
        mean_attribution = get_relevance_scores_for_all_classes(
            model,
            trainloader,
            net_layer=trainable_module[args.layer_index],
            layer_name=trainable_module_name[args.layer_index],
            attribution_method=args.attr,
        )
    else:
        logger.info(
            "Computing relevance — layer: %s - class: %s",
            trainable_module_name[args.layer_index],
            args.test_image,
        )
        attribution, mean_attribution = get_relevance_scores(
            model,
            images,
            labels,
            classes,
            net_layer=trainable_module[args.layer_index],
            layer_name=trainable_module_name[args.layer_index],
            top_m_images=-1,
            attribution_method=args.attr,
        )
    
    # ------------------------------------------------------------------
    # Task 2‑1: Select & cluster activations
    # ------------------------------------------------------------------
    if args.num_samples != 0:
        subset_loader, test_images, test_labels = extract_random_class(
            test_dataset,
            test_all=args.idc_test_all,
            num_samples=args.num_samples,
        )
    else:
        test_images, test_labels = get_class_data(testloader, classes, args.test_image)
    
    idc = IDC(
        model,
        classes,
        args.top_m_neurons,
        args.n_clusters,
        args.use_silhouette,
        args.all_class,
        "KMeans",
    )
    
    if args.end2end:
        important_neuron_indices, inorderd_indices = idc.select_top_neurons_all(attribution, "fc3")  # noqa: E501, pylint: disable=unused‑variable
        activation_values, selected_activations = idc.get_activation_values_for_model(
            images,
            important_neuron_indices,
        )
    else:
        important_neuron_indices = idc.select_top_neurons(mean_attribution)
        if args.all_class:
            activation_values, selected_activations = idc.get_activation_values_for_neurons(
                images,
                important_neuron_indices,
                trainable_module_name[args.layer_index],
                trainloader,
            )
        else:
            activation_values, selected_activations = idc.get_activation_values_for_neurons(
                images,
                important_neuron_indices,
                trainable_module_name[args.layer_index],
            )
    
    # ------------------------------------------------------------------
    # Task 2‑2: Cluster activations
    # ------------------------------------------------------------------
    if args.end2end:
        cluster_groups = idc.cluster_activation_values_all(selected_activations)
    else:
        cluster_groups = idc.cluster_activation_values(
            selected_activations,
            trainable_module[args.layer_index],
        )
    logger.info("Activation values clustered.")
    
    # ------------------------------------------------------------------
    # Task 2‑3: Compute IDC coverage
    # ------------------------------------------------------------------
    if args.end2end:
        unique_cluster, coverage_rate = idc.compute_idc_test_whole(
            test_images,
            test_labels,
            important_neuron_indices,
            cluster_groups,
            args.attr,
        )
        logger.info(
            "Test Class %s, Testing Samples: %d, Attribution: %s, IDC Coverage: %.4f",
            args.test_image,
            len(test_images),
            args.attr,
            coverage_rate,
        )
    else:
        unique_cluster, coverage_rate = idc.compute_idc_test(
            inputs_images=test_images,
            labels=test_labels,
            indices=important_neuron_indices,
            cluster_groups=cluster_groups,
            net_layer=trainable_module[args.layer_index],
            layer_name=trainable_module_name[args.layer_index],
            attribution_method=args.attr,
        )
        logger.info(
            "Testing Samples: %d, Tested Layer: %s, Attribution: %s, IDC Coverage: %.4f",
            len(test_images),
            trainable_module_name[args.layer_index],
            args.attr,
            coverage_rate,
        )
        
    # ------------------------------------------------------------------
    # (Optional) Infidelity metric
    # ------------------------------------------------------------------
    # infidelity_value = infidelity_metric(net, perturb_fn, images, attribution)
    # logger.info("Infidelity: %.2f", infidelity_value)

# -----------------------------------------------------------------------------
# Script entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
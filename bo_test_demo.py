#!/usr/bin/env python
"""
Integration test for:

  • src/clustering.make
  • IDC with clustering_params forwarding
  • src/cluster_bo.bo_pure.BOSearch  (pure‑BoTorch BO)

Optimizing with corr(coverage, F1) for 5 trials just to prove the whole path works.
"""
import os

import torch
import pandas as pd
from src.clustering import make
from src.search import BOSearch
from src.utils import get_data, parse_args, get_model, eval_model_dataloder, get_trainable_modules_main, _configure_logging


# Create a small subset for quick BO testing
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from src.utils import get_data, parse_args, get_model, eval_model_dataloder, get_trainable_modules_main, _configure_logging
from src.idc import IDC


N_TRIALS = 30

def prapare_data_models(args):
    # Logger settings
    logger = _configure_logging(args.logging, args, 'debug')
    
    # Model settings
    model_path = os.getenv("HOME") + args.saved_model
    
    # Model loading
    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)

    return model, module_name, module, trainable_module, trainable_module_name, logger

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def wisdom_coverage(csv_file, 
                    top_m_neurons, 
                    n_clusters, 
                    clustering_method_name,
                    clustering_params, 
                    model, 
                    train_loader, 
                    test_loader):
    
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(top_m_neurons)
    top_k_neurons = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        top_k_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)

    idc = IDC(model, top_m_neurons, n_clusters, False, True, clustering_method_name, clustering_params, None)

    activation_values, selected_activations = idc.get_activations_model_dataloader(train_loader, top_k_neurons)
    selected_activations = {k: v.half().cpu() for k, v in selected_activations.items()}
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(test_loader, top_k_neurons, cluster_groups)

    del activation_values
    del selected_activations
    del cluster_groups
    del idc
    torch.cuda.empty_cache()
    
    return coverage_rate

def mini_test(train_dataset, logger):
    indices = random.sample(range(len(train_dataset)), min(500, len(train_dataset)))
    small_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(small_subset, batch_size=32, shuffle=True)
    logger.info(f"Using subset with {len(small_subset)} samples")
    return train_loader
    
def main() -> None:
    set_seed()
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    model, module_name, module, trainable_module, trainable_module_name, logger = prapare_data_models(args)
    train_loader, test_loader, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path)

    # 1. sanity‑check the clustering factory
    km = make("KMeans", n_clusters=3, random_state=42, n_init=10)
    logger.info(f"Factory returned: {km}")

    # 3. BoTorch optimiser – tiny budget just to exercise the path
    searcher = BOSearch(
        csv_file=args.csv_file,
        train_loader=train_loader,
        model=model,
        idc_cfg=dict(
            classes=list(range(len(classes))),
            top_m_neurons=args.top_m_neurons,
            n_clusters=args.n_clusters,
        ),
        device=device,
        seed=42,
    )
    best_cfg = searcher.optimize(n_trials=N_TRIALS, init_points=2)
    logger.info("Best config:", best_cfg)

    coverage_rate_baseline = wisdom_coverage(args.csv_file, 
                    args.top_m_neurons, 
                    args.n_clusters, 
                    "KMeans",
                    {'random_state': 42, 'n_init': 10, 'n_clusters': args.n_clusters}, 
                    model, 
                    train_loader, 
                    test_loader)
    
    clustering_params = {k: v for k, v in best_cfg.items() if k != "algo"}
    coverage_rate_opt = wisdom_coverage(args.csv_file, 
                    args.top_m_neurons, 
                    args.n_clusters, 
                    best_cfg["algo"],
                    clustering_params, 
                    model, 
                    train_loader,
                    test_loader)
    
    logger.info(f"Baseline coverage rate: {coverage_rate_baseline}")
    logger.info(f"Optimised coverage rate: {coverage_rate_opt}")

# python bo_test_demo.py --model lenet --saved-model "/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth" --dataset mnist --data-path /data/shenghao/dataset --batch-size 128 --device 'cuda:0' --csv-file "./saved_files/pre_csv/lenet_mnist.csv" --attr lrp --top-m-neurons 10
# python bo_test_demo.py --model lenet --saved-model "/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth" --dataset cifar10 --data-path /data/shenghao/dataset --batch-size 64 --device 'cuda:0' --csv-file "./saved_files/pre_csv/lenet_cifar10.csv" --attr lrp --top-m-neurons 10
# python bo_test_demo.py --model vgg16 --saved-model "/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth" --dataset cifar10 --data-path /data/shenghao/dataset --batch-size 64 --device 'cuda:0' --csv-file "./saved_files/pre_csv/vgg16_cifar10.csv" --attr lrp --top-m-neurons 10
# python bo_test_demo.py --model resnet18 --saved-model "/torch-deepimportance/models_info/saved_models/resnet18_CIFAR10_whole.pth" --dataset cifar10 --data-path /data/shenghao/dataset --batch-size 64 --device 'cuda:0' --csv-file "./saved_files/pre_csv/resnet_cifar10.csv" --attr lrp --top-m-neurons 10
# python bo_test_demo.py --model resnet18 --saved-model "/torch-deepimportance/models_info/saved_models/resnet18_IMAGENET_patched_whole.pth" --dataset imagenet --data-path /data/shenghao/dataset --batch-size 32 --device 'cuda:0' --csv-file "./saved_files/pre_csv/resnet_imagenet.csv" --attr lrp --top-m-neurons 10
if __name__ == "__main__":
    main()
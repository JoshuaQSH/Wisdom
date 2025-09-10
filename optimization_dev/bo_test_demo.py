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
import matplotlib.pyplot as plt
from src.clustering import make
from src.search import BOSearch
from src.utils import get_data, parse_args, get_model, get_trainable_modules_main, _configure_logging


# Create a small subset for quick BO testing
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from src.utils import get_data, parse_args, get_model, get_trainable_modules_main, _configure_logging, eval_model_dataloder
from src.idc import IDC
from src.wisdom import WisdomIDC

from helper import get_adv_dataloader, get_generated_dataset_optimized


N_TRIALS = 5
MINI_TEST = False  # Set to True for quick testing with a small subset

PREDICTLESS = {
    "AgglomerativeClustering",
    "SpectralClustering",
    # "DBSCAN",
    # "OPTICS",
    # "HDBSCAN",
}
clustering_params_all = {
    "KMeans": {"n_clusters": 2, "random_state": 42, "n_init": 10},
    "MiniBatchKMeans": {"n_clusters": 2, "batch_size": 32, "max_iter": 100, "random_state": 42},
    "BisectingKMeans": {"n_clusters": 2, "random_state": 42, "n_init": 10},
    "AgglomerativeClustering": {"n_clusters": 2, "linkage": "ward", "metric": "euclidean"},
    "SpectralClustering": {"n_clusters": 2, "affinity": "rbf", "assign_labels": "kmeans"},
    # "DBSCAN": {"eps": 0.1, "min_samples": 10, "metric": "euclidean"},
    # "OPTICS": {"min_samples": 2, "xi": 0.05, "min_cluster_size": 2},
    # "HDBSCAN": {"min_cluster_size": 2, "min_samples": 2, "cluster_selection_epsilon": 0.01, "cluster_selection_method": "eom"},
    "MeanShift": {"bandwidth": None, "quantile": 0.2, "n_samples": None, "random_state": 0, "bin_seeding": True, "cluster_all": True, "max_iter": 300, "min_bin_freq": 1},
    # "AffinityPropagation": {"damping": 0.9, "preference": -50},
    "Birch": {"threshold": 0.5, "n_clusters": 2},
}

def prepare_data_models(args):
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
                    device,
                    train_loader, 
                    test_loader):

    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(top_m_neurons)
    top_k_neurons = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        top_k_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)
        
    wisdom_idc = WisdomIDC(
            model=model,
            top_m_neurons=top_m_neurons,
            n_clusters=n_clusters,
            use_silhouette=False,
            test_all_classes=True,
            clustering_method_name=clustering_method_name,
            device=device,
            clustering_params=clustering_params,
            cache_path=None,
    )

    train_acts = wisdom_idc.get_selected_activations(train_loader, top_k_neurons)
    cluster_groups = wisdom_idc.cluster_per_neuron(train_acts)

    test_acts = wisdom_idc.get_selected_activations(test_loader, top_k_neurons)
    coverage_rate, total_combination, max_coverage = wisdom_idc.compute_coverage(test_acts, cluster_groups)

    del train_acts
    del test_acts
    del cluster_groups
    del wisdom_idc
    torch.cuda.empty_cache()
    
    return coverage_rate, total_combination, max_coverage

def mini_test(train_dataset, test_dataset, logger, sample_size=500, test_size=100):
    train_indices = random.sample(range(len(train_dataset)), min(sample_size, len(train_dataset)))
    test_indices = random.sample(range(len(test_dataset)), min(test_size, len(test_dataset)))
    small_subset_train = Subset(train_dataset, train_indices)
    small_subset_test = Subset(test_dataset, test_indices)
    train_loader = DataLoader(small_subset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(small_subset_test, batch_size=32, shuffle=False)
    logger.info(f"Using subset with {len(small_subset_train)} samples")
    
    return train_loader, test_loader

def save_and_plot_results(
    csv_path,
    pdf_path_prefix,
    coverage_rate_origin,
    norm_coverage_origin,
    coverage_rate_io_baseline,
    norm_coverage_io_baseline,
    origin_improvement,
    coverage_rate_io_bo_baseline,
    norm_coverage_io_bo_baseline,
    coverage_rate_io,
    norm_coverage_io,
    bo_improvement
):
    
    df = pd.DataFrame([{
        "coverage_rate_origin": coverage_rate_origin,
        "norm_coverage_origin": norm_coverage_origin,
        "coverage_rate_io_baseline": coverage_rate_io_baseline,
        "norm_coverage_io_baseline": norm_coverage_io_baseline,
        "origin_improvement": origin_improvement,
        "coverage_rate_io_bo_baseline": coverage_rate_io_bo_baseline,
        "norm_coverage_io_bo_baseline": norm_coverage_io_bo_baseline,
        "coverage_rate_io": coverage_rate_io,
        "norm_coverage_io": norm_coverage_io,
        "bo_improvement": bo_improvement
    }])
    df.to_csv(csv_path, index=False)

    # Bar plot 1: normalized coverage rate with U_IO
    labels = ['Origin', 'BO-base', 'BO-Opti']
    norm_covs = [
        norm_coverage_io_baseline,
        norm_coverage_io_bo_baseline,
        norm_coverage_io
    ]
    plt.figure(figsize=(6,4))
    plt.bar(labels, norm_covs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('Normalized Coverage Rate (U_IO)')
    plt.title('Normalized Coverage Rate (U_IO)')
    plt.tight_layout()
    plt.savefig(f"{pdf_path_prefix}_coverage.pdf")
    plt.close()

    # Bar plot 2: improvements
    improvements = [
        origin_improvement,
        bo_improvement,
        None  # No improvement for BO-Opti itself
    ]
    plt.figure(figsize=(6,4))
    plt.bar(labels[:2], improvements[:2], color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Improvement')
    plt.title('Coverage Rate Improvement')
    plt.tight_layout()
    plt.savefig(f"{pdf_path_prefix}_improvement.pdf")
    plt.close()

def main() -> None:
    set_seed()
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    model, module_name, module, trainable_module, trainable_module_name, logger = prepare_data_models(args)
    train_loader, test_loader, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path)

    if MINI_TEST:
        train_loader, test_loader = mini_test(train_dataset, test_dataset, logger, sample_size=100, test_size=20)

    accuracy, avg_loss, f1 = eval_model_dataloder(model, test_loader, device)
    print(f"Model accuracy: {accuracy:.4f}, Avg loss: {avg_loss:.4f}, F1 score: {f1:.4f}")

    # Sanity-check the clustering factory
    km = make("KMeans", n_clusters=2, random_state=42, n_init=10)
    logger.info(f"Factory returned: {km}")

    # BoTorch optimiser – tiny budget just to exercise the path
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

    # 1. Cov(U_O)
    coverage_rate_origin, total_combination, max_coverage = wisdom_coverage(args.csv_file, 
                    args.top_m_neurons, 
                    args.n_clusters, 
                    "KMeans",
                    {'random_state': 42, 'n_init': 10, 'n_clusters': args.n_clusters}, 
                    model,
                    device,
                    train_loader, 
                    test_loader)
    
    # U_IO_loader, U_RO_loader = get_adv_dataloader(model, test_loader, device=device, batch_size=args.batch_size, csv_file=args.csv_file, attr='wisdom')
    U_IO_loader, U_RO_loader = get_generated_dataset_optimized(args, model, test_dataset, logger)

    # 2. Cov(U_IO) baseline
    coverage_rate_io_baseline, total_combination_io_baseline, max_coverage_io_baseline = wisdom_coverage(args.csv_file, 
                    args.top_m_neurons, 
                    args.n_clusters,
                    "KMeans",
                    {'random_state': 42, 'n_init': 10, 'n_clusters': args.n_clusters}, 
                    model, 
                    device,
                    train_loader, 
                    U_IO_loader)
    
    clustering_params = {k: v for k, v in best_cfg.items() if k != "algo"}
    clustering_params_base = clustering_params_all[best_cfg["algo"]]
    
    # 3. Baseline Cov(U_IO) and Cov(U_RO) with the BO origin config
    coverage_rate_io_bo_baseline, total_combination_io_bo_baseline, max_coverage_io_bo_baseline = wisdom_coverage(args.csv_file, 
                    args.top_m_neurons, 
                    args.n_clusters, 
                    best_cfg["algo"],
                    clustering_params_base, 
                    model,
                    device,
                    train_loader,
                    U_IO_loader)
    
    # 4. Optimized Cov(U_IO) and Cov(U_RO)
    coverage_rate_io, total_combination_io, max_coverage_io = wisdom_coverage(args.csv_file, 
                    args.top_m_neurons, 
                    args.n_clusters, 
                    best_cfg["algo"],
                    clustering_params, 
                    model, 
                    device,
                    train_loader,
                    U_IO_loader)

    logger.info(f"[U_O] Coverage rate: {coverage_rate_origin}, Total combinations: {total_combination}, Max coverage: {max_coverage}, Actual Coverage Rate: {coverage_rate_origin / max_coverage}")
    logger.info(f"[U_IO] Coverage rate: {coverage_rate_io_baseline}, Total combinations: {total_combination_io_baseline}, Max coverage: {max_coverage_io_baseline}, Actual Coverage Rate: {coverage_rate_io_baseline / max_coverage_io_baseline}")
    logger.info(f"[Origin Results] Improvement: {(coverage_rate_io_baseline / max_coverage_io_baseline) / (coverage_rate_origin / max_coverage)}x")
    logger.info(f"[U_IO-BO-Base] Coverage rate: {coverage_rate_io_bo_baseline}, Total combinations: {total_combination_io_bo_baseline}, Max coverage: {max_coverage_io_bo_baseline}, Actual Coverage Rate: {coverage_rate_io_bo_baseline / max_coverage_io_bo_baseline}")
    logger.info(f"[U_IO-BO-Opti] Coverage rate: {coverage_rate_io}, Total combinations: {total_combination_io}, Max coverage: {max_coverage_io}, Actual Coverage Rate: {coverage_rate_io / max_coverage_io}")
    logger.info(f"[BO Results] Improvement: {(coverage_rate_io / max_coverage_io) / (coverage_rate_io_bo_baseline / max_coverage_io_bo_baseline)}x")

    # Save and plot results
    save_and_plot_results(
        csv_path=f"bo_{args.model}_{args.dataset}.csv",
        pdf_path_prefix=f"bo_{args.model}_{args.dataset}",
        coverage_rate_origin=coverage_rate_origin,
        norm_coverage_origin=coverage_rate_origin / max_coverage,
        coverage_rate_io_baseline=coverage_rate_io_baseline,
        norm_coverage_io_baseline=coverage_rate_io_baseline / max_coverage_io_baseline,
        origin_improvement=(coverage_rate_io_baseline / max_coverage_io_baseline) / (coverage_rate_origin / max_coverage),
        coverage_rate_io_bo_baseline=coverage_rate_io_bo_baseline,
        norm_coverage_io_bo_baseline=coverage_rate_io_bo_baseline / max_coverage_io_bo_baseline,
        coverage_rate_io=coverage_rate_io,
        norm_coverage_io=coverage_rate_io / max_coverage_io,
        bo_improvement=(coverage_rate_io / max_coverage_io) / (coverage_rate_io_bo_baseline / max_coverage_io_bo_baseline)
    )

# python ./optimization_dev/bo_test_demo.py --model lenet --saved-model "/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth" --dataset mnist --data-path /data/shenghao/dataset --batch-size 128 --device 'cuda:0' --csv-file "./saved_files/pre_csv/lenet_mnist.csv" --attr lrp --top-m-neurons 10
# python ./optimization_dev/bo_test_demo.py --model lenet --saved-model "/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth" --dataset cifar10 --data-path /data/shenghao/dataset --batch-size 64 --device 'cuda:0' --csv-file "./saved_files/pre_csv/lenet_cifar10.csv" --attr lrp --top-m-neurons 10
# python ./optimization_dev/bo_test_demo.py --model vgg16 --saved-model "/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth" --dataset cifar10 --data-path /data/shenghao/dataset --batch-size 64 --device 'cuda:0' --csv-file "./saved_files/pre_csv/vgg16_cifar10.csv" --attr lrp --top-m-neurons 10
# python ./optimization_dev/bo_test_demo.py --model resnet18 --saved-model "/torch-deepimportance/models_info/saved_models/resnet18_CIFAR10_whole.pth" --dataset cifar10 --data-path /data/shenghao/dataset --batch-size 64 --device 'cuda:0' --csv-file "./saved_files/pre_csv/resnet18_cifar10.csv" --attr lrp --top-m-neurons 10
# python ./optimization_dev/bo_test_demo.py --model resnet18 --saved-model "/torch-deepimportance/models_info/saved_models/resnet18_IMAGENET_patched_whole.pth" --dataset imagenet --data-path /data/shenghao/dataset --batch-size 32 --device 'cuda:0' --csv-file "./saved_files/pre_csv/resnet18_imagenet.csv" --attr lrp --top-m-neurons 10
if __name__ == "__main__":
    main()
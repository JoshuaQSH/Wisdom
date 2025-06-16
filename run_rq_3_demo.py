from __future__ import annotations

import os
import copy
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchattacks as ta

from src.attribution import get_relevance_scores_dataloader
from src.utils import get_data, parse_args, get_model, get_trainable_modules_main, _configure_logging
from src.idc import IDC
from src.nlc_coverage import (
    NC, KMNC, NBC, SNAC, TKNC, TKNP, CC,
    NLC, LSC, DSC, MDSC
)
from src.nlc_tool import get_layer_output_sizes


# python run_rq_3_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --batch-size 32 --device 'cuda:0' --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/lenet_cifar_b32.csv'

"""
sample_sizes = [100, 1000, 3000]
adversarial_attacks = ['fgsm', 'pgd', 'cw']
error_rates = [0.01, 0.05, 0.10]

1. Sample the correct inputs based on the sample_sizes
2. Create clean dataset
3. Replace some of the clean dataset with adversarial examples based on the error_rates
4. Calculate the coverage metrics for each of the adversarial datasets -> mixed_coverage and clean_coverage
5. normalized_change = abs(mixed_coverage - clean_coverage) / clean_coverage
"""

# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------
def prapared_parameters(args):
    ### Logger settings
    logger = _configure_logging(args.logging, args, 'debug')
    
    ### Model settings
    model_path = os.getenv("HOME") + args.saved_model
    
    ### Model loading
    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)

    return model, module_name, module, trainable_module, trainable_module_name, logger

def set_seed(seed: int = 2025):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Coverage helpers for baselines
# ---------------------------------------------------------------------------

def _infer_layer_sizes(model: torch.nn.Module, sample_batch, device):
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    num_neuron = 0
    layer_size_dict = get_layer_output_sizes(model, sample_batch.to(device))
    
    for layer_name in layer_size_dict.keys():
        num_neuron += layer_size_dict[layer_name][0]
    # print('Total %d layers: ' % len(layer_size_dict.keys()))
    # print('Total %d neurons: ' % num_neuron)
    
    return layer_size_dict

def initialize_baseline_coverage(
    model: nn.Module,
    layer_size: Dict[str, List[int]],
    train_loader: DataLoader,
    num_classes: int,
    device: torch.device,
):
    """Initialize a coverage method with given configuration."""
    templates = {}
    # NC 
    templates["NC"] = NC(model, layer_size, hyper=0.5)

    # KMNC & NBC & SNAC & TKNC
    templates["KMNC"] = KMNC(model, layer_size, hyper=10)
    templates["NBC"] = NBC(model, layer_size)
    templates["SNAC"] = SNAC(model, layer_size)
    templates["TKNC"] = TKNC(model, layer_size, hyper=2)

    # # LSC & DSC & MDSC
    lsc_kwargs = {"hyper": 0.1, "min_var": 1e-5, "num_class": num_classes}
    templates["LSC"] = LSC(model, layer_size, **lsc_kwargs)
    templates["DSC"] = DSC(model, layer_size, **lsc_kwargs)
    templates["MDSC"] = MDSC(model, layer_size, **lsc_kwargs)
    
    # # NLC & CC
    templates["NLC"] = NLC(model, layer_size)
    templates["CC"] = CC(model, layer_size, hyper=0.1)

    # Build where needed
    for name, metric in templates.items():
        metric.build(train_loader)
        metric.model.to(device)

    return templates

# ---------------------------------------------------------------------------
# Sampling correct inputs
# ---------------------------------------------------------------------------

def sample_correct_inputs(sample_size, model, test_loader, device):
    """Sample correctly classified inputs from test set."""
    correct_indices = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
                
            batch_correct = (pred == target).cpu().numpy()
            batch_start_idx = batch_idx * test_loader.batch_size
                
            for i, is_correct in enumerate(batch_correct):
                if is_correct:
                    correct_indices.append(batch_start_idx + i)
                        
            if len(correct_indices) >= sample_size * 2:
                break
        
    # Randomly sample the required number
    selected_indices = random.sample(correct_indices, min(sample_size, len(correct_indices)))
    return selected_indices

# ---------------------------------------------------------------------------
# Adversarial example generation
# ---------------------------------------------------------------------------

def generate_adversarial_examples(attack_method, model, target, data, num_adv):
    """Generate adversarial examples using specified attack method."""
    if attack_method == 'fgsm':
        attack = ta.FGSM(model, eps=0.3)
    elif attack_method == 'pgd':
        attack = ta.PGD(model, eps=0.3, alpha=0.1, steps=20)
    elif attack_method == 'cw':
        attack = ta.CW(model, c=1, kappa=0, steps=50, lr=0.01)
    else:
        raise ValueError(f"Unknown attack method: {attack_method}")
        
    # Generate adversarial examples
    adv_data = attack(data[:num_adv], target[:num_adv])
        
    # Verify that adversarial examples are misclassified
    with torch.no_grad():
        adv_pred = model(adv_data).argmax(dim=1)
        successful_adv = adv_pred != target[:num_adv]
            
    return adv_data[successful_adv], successful_adv.sum().item()

# ---------------------------------------------------------------------------
# Coverage change calculation
# ---------------------------------------------------------------------------

def deepimportance_coverage_change(args, model, trainable_module_name, classes, layer_relevance_scores, clean_data, mixed_data):
    idc = IDC(
        model,
        classes,
        args.top_m_neurons,
        args.n_clusters,
        args.use_silhouette,
        args.all_class,
        "KMeans",
    )
    final_layer = trainable_module_name[-1]
    important_neuron_indices, inorderd_indices = idc.select_top_neurons_all(layer_relevance_scores, final_layer)
    clean_loader = DataLoader(clean_data, batch_size=args.batch_size, shuffle=False)
    activation_values, selected_activations = idc.get_activations_model_dataloader(clean_loader, important_neuron_indices)
    selected_activations = {k: v.cpu() for k, v in selected_activations.items()}
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    clean_coverage, _, _ = idc.compute_idc_test_whole_dataloader(clean_loader, important_neuron_indices, cluster_groups)
    idc.total_combination = 1
    mixed_loader = DataLoader(mixed_data, batch_size=args.batch_size, shuffle=False)
    mixed_coverage, _, _ = idc.compute_idc_test_whole_dataloader(mixed_loader, important_neuron_indices, cluster_groups)
    
    normalized_change = abs(mixed_coverage - clean_coverage) / clean_coverage
    return normalized_change    
    

def wisdom_coverage_change(args, model, classes, wisdom_k_neurons, clean_data, mixed_data):
    idc = IDC(model, classes, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class, "KMeans")
    clean_loader = DataLoader(clean_data, batch_size=args.batch_size, shuffle=False)
    activation_values, selected_activations = idc.get_activations_model_dataloader(clean_loader, wisdom_k_neurons)
    selected_activations = {k: v.cpu() for k, v in selected_activations.items()}
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    clean_coverage, _, _ = idc.compute_idc_test_whole_dataloader(clean_loader, wisdom_k_neurons, cluster_groups)
    idc.total_combination = 1
    mixed_loader = DataLoader(mixed_data, batch_size=args.batch_size, shuffle=False)
    mixed_coverage, _, _ = idc.compute_idc_test_whole_dataloader(mixed_loader, wisdom_k_neurons, cluster_groups)
    normalized_change = abs(mixed_coverage - clean_coverage) / clean_coverage
    return normalized_change


def calculate_coverage_change(coverage_method, test_dataset, batch_size, clean_data, mixed_data, method_name):
    coverage_method_clean = copy.deepcopy(coverage_method)
    coverage_method_mixed = copy.deepcopy(coverage_method)
    
    # Build coverage if needed (for surprise-based methods)
    if method_name in ['LSC', 'DSC', 'MDSC']:
        # Use clean data for building
        build_loader = DataLoader(
            Subset(test_dataset, list(range(min(1000, len(test_dataset))))),
            batch_size=32, shuffle=False)
        coverage_method_clean.build(build_loader)
        coverage_method_mixed.build(build_loader)
        
    if method_name in ['KMNC', 'NBC', 'SNAC', 'CC']:
        # Use clean data for building ranges
        build_loader = DataLoader(
            Subset(test_dataset, list(range(min(1000, len(test_dataset))))),
            batch_size=32, shuffle=False)
        coverage_method_clean.build(build_loader)
        coverage_method_mixed.build(build_loader)
    
    # Calculate coverage on clean data
    clean_loader = DataLoader(clean_data, batch_size=batch_size, shuffle=False)
    coverage_method_clean.assess(clean_loader)
    clean_coverage = coverage_method_clean.current
        
    # Calculate coverage on mixed data
    mixed_loader = DataLoader(mixed_data, batch_size=batch_size, shuffle=False)
    coverage_method_mixed.assess(mixed_loader)
    mixed_coverage = coverage_method_mixed.current
        
    # Calculate normalized change
    if clean_coverage == 0:
        return 0  # Avoid division by zero
        
    normalized_change = abs(mixed_coverage - clean_coverage) / clean_coverage
    return normalized_change

# ---------------------------------------------------------------------------
# Visualizations and analysis
# ---------------------------------------------------------------------------

def visualizations_coverage_err(results_df, dataset_name, model_name):
    # plt.style.use("seaborn-v0_8")
    attacks = sorted(results_df["Attack"].unique())
    n_attacks = len(attacks)

    fig, axes = plt.subplots(
        1, n_attacks, figsize=(6 * n_attacks, 5), sharey=True
    )
    if n_attacks == 1:
        axes = [axes]

    agg = (
        results_df
        .groupby(["Attack", "Method", "Adv_Ratio"], as_index=False)
        .agg(mean_cov=("Coverage_Change", "mean"))
    )
    
    handles, labels = [], []
    for ax, attack in zip(axes, attacks):
        sub = agg[agg["Attack"] == attack]

        for method, grp in sub.groupby("Method"):
            grp_sorted = grp.sort_values("Adv_Ratio")
            (line,) = ax.plot(
                grp_sorted["Adv_Ratio"],
                grp_sorted["mean_cov"],
                # marker="o",
                label=method,
            )
            
            if attack == attacks[0]:
                handles.append(line)
                labels.append(method)

        ax.set_title(f"Attack: {attack}")
        ax.set_xlabel("Adversarial Ratio")
        ax.grid(alpha=0.4, linewidth=0.5)

    axes[0].set_ylabel("Average Coverage Change")
    # axes[0].legend(title="Method", bbox_to_anchor=(1.04, 1.02), loc="top")
    
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(labels),
        frameon=True)
    
    plt.tight_layout()
    plt.savefig(f'rq3_results_{dataset_name.lower()}_{model_name}.pdf', dpi=1200, bbox_inches='tight', format='pdf')


def visualizations(results_df, dataset_name='MNIST', model_name='lenet'):
    """Create visualization plots for the results."""
        
    # 1. Coverage change by method and attack type
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Coverage Change Analysis - {dataset_name} - {model_name}', fontsize=16)
        
    # Plot 1: Coverage change by method (averaged across all conditions)
    ax1 = axes[0, 0]
    method_avg = results_df.groupby('Method')['Coverage_Change'].mean().sort_values(ascending=False)
    method_avg.plot(kind='bar', ax=ax1)
    ax1.set_title('Average Coverage Change by Method')
    ax1.set_ylabel('Normalized Coverage Change')
    ax1.tick_params(axis='x', rotation=45)
        
    # Plot 2: Coverage change by attack method
    ax2 = axes[0, 1]
    attack_avg = results_df.groupby('Attack')['Coverage_Change'].mean()
    attack_avg.plot(kind='bar', ax=ax2)
    ax2.set_title('Average Coverage Change by Attack Method')
    ax2.set_ylabel('Normalized Coverage Change')
    ax2.tick_params(axis='x', rotation=45)
        
    # Plot 3: Coverage change by adversarial ratio
    ax3 = axes[1, 0]
    ratio_avg = results_df.groupby('Adv_Ratio')['Coverage_Change'].mean()
    ratio_avg.plot(kind='bar', ax=ax3)
    ax3.set_title('Coverage Change by Adversarial Ratio')
    ax3.set_xlabel('Adversarial Ratio')
    ax3.set_ylabel('Normalized Coverage Change')
        
    # Plot 4: Coverage change by sample size
    ax4 = axes[1, 1]
    size_avg = results_df.groupby('Sample_Size')['Coverage_Change'].mean()
    size_avg.plot(kind='bar', ax=ax4)
    ax4.set_title('Coverage Change by Sample Size')
    ax4.set_xlabel('Sample Size')
    ax4.set_ylabel('Normalized Coverage Change')
        
    plt.tight_layout()
    plt.savefig(f'rq3_results_{dataset_name.lower()}_{model_name}.pdf', dpi=1200, bbox_inches='tight', format='pdf')
    # plt.show()

def analyze_results(results, dataset_name, model_name, logger):
    """Analyze and visualize the experimental results."""
    # Convert results to DataFrame for easier analysis
    rows = []
    for method in results:
        for sample_size in results[method]:
            for attack_config in results[method][sample_size]:
                attack_method, adv_ratio = attack_config.split('_')
                adv_ratio = float(adv_ratio)
                    
                values = results[method][sample_size][attack_config]
                if values:  # Only include if we have data
                    rows.append({
                        'Method': method,
                        'Sample_Size': sample_size,
                        'Attack': attack_method,
                        'Adv_Ratio': adv_ratio,
                        'Coverage_Change': np.mean(values),
                        'Coverage_Change_Std': np.std(values),
                        'Num_Runs': len(values)
                    })
        
    results_df = pd.DataFrame(rows)
        
    # Print summary statistics
    print("\nSummary Statistics:")
    logger.info(results_df.groupby(['Method', 'Attack'])['Coverage_Change'].agg(['mean', 'std', 'count']))
        
    # Create visualizations
    visualizations_coverage_err(results_df, dataset_name=dataset_name, model_name='lenet')
        
    return results_df

# ---------------------------------------------------------------------------
# Main experiment routine
# ---------------------------------------------------------------------------

def run_experiment(args, num_runs=3):
    set_seed()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")

    ### Model settings
    model, module_name, module, trainable_module, trainable_module_name, logger = prapared_parameters(args)
    model.to(device)
    
    ### Data settings
    train_loader, test_loader, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path, args.large_image)
    num_classes = len(classes)
    
    # Baseline templates 
    sample_batch, *_ = next(iter(train_loader))
    layer_size_dict = _infer_layer_sizes(model, sample_batch, device)
    base_templates = initialize_baseline_coverage(model, layer_size_dict, train_loader, num_classes, device)
    
    # IDC based (DeepImportance and WISDOM)
    dp_relevance_scores = get_relevance_scores_dataloader(
            model,
            train_loader,
            device,
            attribution_method='lrp',
        )
    df = pd.read_csv(args.csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(args.top_m_neurons)
    wisdom_k_neurons = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        wisdom_k_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)
    
    suite_sizes = [100, 1000, 3000]
    attack_names = ["fgsm", "pgd", "cw"]
    error_rates = [0.01, 0.05, 0.10]
    # [method_name: str, sample_size: int, attack_method+'_error_rate': str] -> List[float]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Main experiment loop
    for run in range(num_runs):
        for sample_size in suite_sizes:
            # Sample correct inputs
            correct_indices = sample_correct_inputs(sample_size, model, test_loader, device)
            if len(correct_indices) < sample_size:
                logger.warning(f"Only found {len(correct_indices)} correct inputs")
                continue
            
            # Create clean dataset
            clean_subset = Subset(test_dataset, correct_indices[:sample_size])
            clean_loader = DataLoader(clean_subset, batch_size=sample_size, shuffle=False)
            clean_data, clean_target = next(iter(clean_loader))
            clean_data, clean_target = clean_data.to(device), clean_target.to(device)

            for error_ in error_rates:
                logger.info(f"Adversarial Ratio: {error_}")
                num_adv = int(sample_size * error_)
                for attack_ in attack_names:
                    logger.info(f"Attack Method: {attack_}")
                    
                    # Generate adversarial examples
                    adv_data, num_successful = generate_adversarial_examples(attack_, model, clean_target, clean_data, num_adv)
                    if num_successful < num_adv:
                        logger.warning(f"Only {num_successful}/{num_adv} adversarial examples generated")
                        continue
                            
                    # Create mixed dataset (replace some clean with adversarial)
                    mixed_data = clean_data.clone()
                    mixed_target = clean_target.clone()
                    mixed_data[:num_successful] = adv_data[:num_successful]
                    
                    # Create datasets for coverage calculation
                    clean_dataset = torch.utils.data.TensorDataset(clean_data, clean_target)
                    mixed_dataset = torch.utils.data.TensorDataset(mixed_data, mixed_target)
                    
                    # Test each coverage method
                    for method_name, template in base_templates.items():
                        coverage_change = calculate_coverage_change(
                            template, test_dataset, args.batch_size, clean_dataset, mixed_dataset, method_name
                        )
                        coverage_change_dp = deepimportance_coverage_change(args, model, trainable_module_name, classes, dp_relevance_scores, clean_dataset, mixed_dataset)
                        coverage_change_wisdom = wisdom_coverage_change(args, model, classes, wisdom_k_neurons, clean_dataset, mixed_dataset)
                        results[method_name][sample_size][f"{attack_}_{error_}"].append(coverage_change)
                        results["DeepImportance"][sample_size][f"{attack_}_{error_}"].append(coverage_change_dp)
                        results["Wisdom"][sample_size][f"{attack_}_{error_}"].append(coverage_change_wisdom)
                        logger.info(f"{method_name} coverage change: {coverage_change:.4f}")
                        logger.info(f"DeepImportance coverage change: {coverage_change_dp:.4f}")
                        logger.info(f"Wisdom coverage change: {coverage_change_wisdom:.4f}")
                        
    # Analyze and Save results to CSV
    results_df = analyze_results(results, args.dataset, args.model, logger)
    results_df.to_csv(f'rq3_results_{args.dataset}_{args.model}.csv', index=False)
    
if __name__ == "__main__":
    args = parse_args()
    # python run_rq_3_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --batch-size 32 --device 'cuda:0' --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/lenet_cifar_b32.csv'
    run_experiment(args, num_runs=5)
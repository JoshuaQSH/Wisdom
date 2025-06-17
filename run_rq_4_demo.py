import math
import os
import time
import random
from collections import Counter
from typing import Dict, List, Sequence
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.utils import get_data, parse_args, get_model, get_trainable_modules_main, _configure_logging
from src.nlc_tool import get_layer_output_sizes
from src.nlc_coverage import (
    NC,
    KMNC,
    NBC,
    LSC,
    DSC,
    NLC,
)
import torchattacks


# python run_rq_4_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --batch-size 32 --device 'cuda:0' --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/lenet_cifar_b32.csv'


# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------
def set_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prapared_parameters(args):
    # Logger settings
    logger = _configure_logging(args.logging, args, 'debug')

    # Model settings
    model_path = os.getenv("HOME") + args.saved_model
    
    # Model loading
    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)

    return model, module_name, module, trainable_module, trainable_module_name, logger

def get_predictions(model, dataloader, device='cpu'):
    """Get model predictions for a dataset."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_images, _ in dataloader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    
    return predictions

# -----------------------------------------------------------
# Adversarial functions
# -----------------------------------------------------------
def generate_adversarial_examples(model, dataloader, attack_method='CW', device='cpu'):
    """
    Generate adversarial examples.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        attack_method: Type of attack ('CW', 'PGD', etc.)
        device: Device to run on
    
    Returns:
        tuple: (adversarial_images, original_labels)
    """
    model.eval()
    
    if attack_method == 'CW':
        attack = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
        # attack = torchattacks.CW(model, c=1e-2, kappa=0, steps=1000, lr=0.01)
    elif attack_method == 'PGD':
        attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
    else:
        raise ValueError(f"Unsupported attack method: {attack_method}")
    
    attack.set_model_training_mode(False)
    adversarial_images = []
    original_labels = []
    
    print(f"Generating adversarial examples using {attack_method}...")
    for batch_images, batch_labels in tqdm(dataloader):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        
        # Generate adversarial examples
        adv_images = attack(batch_images, batch_labels)
        
        adversarial_images.append(adv_images.cpu())
        original_labels.append(batch_labels.cpu())
    
    adversarial_images = torch.cat(adversarial_images, dim=0)
    original_labels = torch.cat(original_labels, dim=0)
    
    return adversarial_images, original_labels

# -----------------------------------------------------------
# Pielou Evenness and Pearson correlation coefficient
# -----------------------------------------------------------
def calculate_pielou_evenness(predictions):
    """
    Calculate Pielou's evenness index for output impartiality.
    
    Args:
        predictions: List or array of predicted class labels
    
    Returns:
        float: Pielou's evenness score (0 to 1, where 1 is perfectly even)
    """
    if len(predictions) == 0:
        return 0.0
    
    # Count frequency of each class
    class_counts = Counter(predictions)
    total_samples = len(predictions)
    num_classes = len(class_counts)
    
    if num_classes <= 1:
        return 0.0
    
    # Calculate Shannon diversity index (H)
    shannon_diversity = 0
    for count in class_counts.values():
        if count > 0:
            p = count / total_samples
            shannon_diversity -= p * np.log(p)
    
    # Calculate maximum possible diversity (H_max = ln(S))
    max_diversity = np.log(num_classes)
    
    # Pielou's evenness = H / H_max
    if max_diversity == 0:
        return 0.0
    
    evenness = shannon_diversity / max_diversity
    return evenness

def calculate_correlations(df):
    """
    Calculate Pearson correlation coefficients between coverage and impartiality.
    
    Args:
        df: Results dataframe
    
    Returns:
        pd.DataFrame: Correlation results
    """
    correlation_results = []
    
    # Calculate correlations for each coverage method
    for method in df['coverage_method'].unique():
        method_data = df[df['coverage_method'] == method]
        
        # Overall correlation
        if len(method_data) > 1:
            corr_coef, p_value = pearsonr(method_data['coverage_score'].astype(float),  method_data['impartiality_score'].astype(float))
            correlation_results.append({
                'coverage_method': method,
                'test_type': 'overall',
                'correlation_coefficient': corr_coef,
                'p_value': p_value,
                'sample_count': len(method_data)
            })
        
        # Separate correlations for clean and adversarial
        for test_type in ['clean', 'adversarial']:
            subset_data = method_data[method_data['test_type'] == test_type]
            if len(subset_data) > 1:
                corr_coef, p_value = pearsonr(subset_data['coverage_score'].astype(float), subset_data['impartiality_score'].astype(float))
                correlation_results.append({
                    'coverage_method': method,
                    'test_type': test_type,
                    'correlation_coefficient': corr_coef,
                    'p_value': p_value,
                    'sample_count': len(subset_data)
                })
    
    return pd.DataFrame(correlation_results)

def pielou_evenness(preds: np.ndarray) -> float:
    counter = Counter(preds.tolist())
    counts = np.array(list(counter.values()), dtype=float)
    S = (counts > 0).sum()
    if S == 0:
        return 0.0
    p = counts / counts.sum()
    H = -(p * np.log(p)).sum()
    return H / np.log(S)

def pielou_evenness_torch(preds: torch.Tensor) -> float:
    
    counter = Counter(preds.tolist())
    counts = torch.tensor(list(counter.values()), dtype=torch.float32)
    S = (counts > 0).sum().item() # richness, how many distinct classes are present
    if S == 0:
        return torch.tensor(0.0, device=preds.device)
    p = counts / counts.sum()
    H = -(p * torch.log(p)).sum()  # Shannon entropy
    return H / torch.log(torch.tensor(S, dtype=torch.float32))  # normalized by log(S)


# -----------------------------------------------------------
# Coverage helpers for baselines
# -----------------------------------------------------------

def evaluate_coverage_methods(model, dataloader, layer_size_dict):
    model.eval()
    coverage_scores = {}
    
    # Initialize coverage methods
    coverage_methods = {
        'NC': NC(model, layer_size_dict, hyper=0.75),
        'NBC': NBC(model, layer_size_dict),
        'KMNC': KMNC(model, layer_size_dict, hyper=1000),
        'LSC': LSC(model, layer_size_dict, hyper=2000, min_var=1e-5, num_class=10),
        'DSC': DSC(model, layer_size_dict, hyper=0.1, min_var=1e-5, num_class=10),
        'NLC': NLC(model, layer_size_dict)
    }
    
    # Build coverage methods that require training data
    build_methods = ['LSC', 'DSC', 'KMNC', 'NBC']
    
    print("Building coverage methods...")
    for method_name in build_methods:
        if method_name in coverage_methods:
            print(f"Building {method_name}...")
            coverage_methods[method_name].build(dataloader)
    
    # Assess coverage
    print("Assessing coverage...")
    for method_name, coverage_method in coverage_methods.items():
        print(f"Assessing {method_name}...")
        try:
            coverage_method.assess(dataloader)
            coverage_scores[method_name] = coverage_method.current
        except Exception as e:
            print(f"Error in {method_name}: {e}")
            coverage_scores[method_name] = 0.0
    
    return coverage_scores

def stratified_sample(dataset, sample_size, num_classes=10):
    """
    Create a stratified sample maintaining class distribution.
    
    Args:
        dataset: PyTorch dataset
        sample_size: Number of samples to draw
        num_classes: Number of classes in dataset
    
    Returns:
        Subset: Stratified subset of the dataset
    """
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Calculate samples per class
    samples_per_class = sample_size // num_classes
    remainder = sample_size % num_classes
    
    selected_indices = []
    
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        
        # Add extra sample for some classes if there's remainder
        n_samples = samples_per_class + (1 if class_idx < remainder else 0)
        n_samples = min(n_samples, len(class_indices))
        
        selected = np.random.choice(class_indices, n_samples, replace=False)
        selected_indices.extend(selected)
    
    return Subset(dataset, selected_indices)


# -----------------------------------------------------------
# Main experimental routine
# -----------------------------------------------------------

def run_experiment(model, test_loader, test_dataset, num_classes=10, sample_sizes=[100, 500, 1000], device='cpu'):
    # Set random seeds for reproducibility
    set_seed()
    
    # Get layer sizes for coverage computation
    sample_input, *_ = next(iter(test_loader))
    sample_input = sample_input.to(device)
    layer_size_dict = get_layer_output_sizes(model, sample_input)
    
    results = []
    
    for sample_size in sample_sizes:
        print(f"Processing sample size: {sample_size}")
        
        # 1. Sample clean test cases (U_t)
        print(f"Creating stratified sample of {sample_size} clean test cases...")
        clean_subset = stratified_sample(test_dataset, sample_size, num_classes)
        clean_dataloader = DataLoader(clean_subset, batch_size=32, shuffle=False)
        
        # Get clean predictions for impartiality calculation
        clean_predictions = get_predictions(model, clean_dataloader, device)
        clean_impartiality = calculate_pielou_evenness(clean_predictions)
        
        # Calculate coverage for clean test cases
        print("Evaluating coverage on clean test cases...")
        clean_coverage_scores = evaluate_coverage_methods(model, clean_dataloader, layer_size_dict)
        
        # 2. Generate adversarial examples (U_b)
        print(f"Generating {sample_size} adversarial examples...")
        adv_images, adv_labels = generate_adversarial_examples(model, clean_dataloader, 'CW', device)
        
        # Create adversarial dataset
        adv_dataset = torch.utils.data.TensorDataset(adv_images, adv_labels)
        adv_dataloader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Get adversarial predictions for impartiality calculation
        adv_predictions = get_predictions(model, adv_dataloader, device)
        adv_impartiality = calculate_pielou_evenness(adv_predictions)
        
        # Calculate coverage for adversarial test cases
        print("Evaluating coverage on adversarial test cases...")
        adv_coverage_scores = evaluate_coverage_methods(model, adv_dataloader, layer_size_dict)
        
        # Store results
        for method_name in clean_coverage_scores.keys():
            # Clean test cases
            results.append({
                'sample_size': sample_size,
                'test_type': 'clean',
                'coverage_method': method_name,
                'coverage_score': clean_coverage_scores[method_name],
                'impartiality_score': clean_impartiality,
                'num_samples': sample_size
            })
            
            # Adversarial test cases
            results.append({
                'sample_size': sample_size,
                'test_type': 'adversarial',
                'coverage_method': method_name,
                'coverage_score': adv_coverage_scores[method_name],
                'impartiality_score': adv_impartiality,
                'num_samples': sample_size
            })
    
    return pd.DataFrame(results)

def main(args):
    """Main function to run the RQ4 experiment."""
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    sample_sizes = [100, 500, 1000]
    
    # Model settings
    model, module_name, module, trainable_module, trainable_module_name, logger = prapared_parameters(args)
    model.eval()
    model.to(device)
    
    # Data settings
    train_loader, test_loader, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path)
    num_classes = len(classes)
    
    logger.info("Starting RQ4 experiment...")
    results_df = run_experiment(model, test_loader, test_dataset, num_classes=num_classes, sample_sizes=sample_sizes, device=device)
    logger.info("Calculating correlations...")
    correlation_df = calculate_correlations(results_df)
    
    # Save results to CSV files
    results_df.to_csv('rq4_coverage_impartiality_results.csv', index=False)
    correlation_df.to_csv('rq4_correlation_results.csv', index=False)
    
    logger.info("Experiment completed! Files saved!")
    
    # Display summary
    logger.info("\nCorrelation Summary:")
    logger.info(correlation_df.round(4))
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
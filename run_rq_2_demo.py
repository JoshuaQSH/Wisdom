import random
import time
import os
import math
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from collections import defaultdict
from captum.attr import LRP

from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
from src.attribution import get_relevance_scores_dataloader
from src.utils import get_data, parse_args, get_model, eval_model_dataloder, get_trainable_modules_main, _configure_logging
from src.idc import IDC
from src.nlc_coverage import calculate_coverage_ratio


"""
U_I (Importance-perturbed): Each image has Gaussian white noise (mean 0, std 0.3) added to its most important 2% pixels.
U_R (Random-perturbed): Each image has noise added to a random 2% of its pixels.

U_IO (Importance-perturbed Dataset + Original Dataset): Original images with noise added to the most important 2% pixels.
U_RO (Random-perturbed Dataset + Original Dataset): Original images with noise added to a random 2% of its pixels.

# Step 1: Get the relevance maps for the test set using Attribution methods (e.g., LRP).
# Step 2: For each image, find the top 2% most important pixels based on the relevance map (Baseline: random & LRP) and form the new dataset.
# Step 3: Coverage testing for the perturbed dataset.
# Optional: Visualize the perturbed images and their relevance maps - this goes to the `./unittest/sanity_check.py` file.

[Optional Check] Run with: 
$ python ./unittest/sanity_check.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth' --dataset mnist --data-path /path/to/mnist --batch-size 128 --device 'cuda:0' --csv-file './saved_files/pre_csv/lenet_mnist.csv' --attr lrp --top-m-neurons 10
$ python ./unittest/sanity_check.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth' --dataset mnist --data-path /path/to/mnist --batch-size 128 --device 'cuda:0' --csv-file './saved_files/pre_csv/lenet_mnist.csv' --attr wisdom --top-m-neurons 10
"""

# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------

TOPK = 0.02  # Top-k fraction of pixels to perturb (2%)
gausian_STD = 0.5
start_ms = int(time.time() * 1000)
TIMESTAMP = time.strftime("%Y%m%d‑%H%M%S", time.localtime(start_ms / 1000))
acts = defaultdict(list)

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


# Other coverage methods with hyperparameters
"""
choices = ['NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'CC', 'NLC', 'LSC', 'DSC', 'MDSC']

Hyperparameters:
@NC: Activation threshold \in {0.25, 0.50, 0.75}
@KMNC: Number of segments per neuron \in {100, 1000, 10000}
@SNAC: Parameter-free
@NBC: Parameter-free
@TKNC: Top k neurons \in {1, 10, 50}
@TKNP: Pattern width \in {1, 10, 50}
@CC: Radius T (#neurons per cluster) \in {10, 20, 50}
@NLC: Parameter-free
@LSC: Bucket width/threshold \in {1, 10, 100}
@DSC: Bucket width \in {0.01, 0.1, 1}
@MDSC: Bucket width \in {1, 10, 100}
"""

def calculate_all_coverage_ratios(build_loader, target_loader, model, device, num_class=10):

    results = {}
    
    # Methods with their typical hyperparameters
    methods_config = {
        'NC': {'hyper': 0.5},
        'KMNC': {'hyper': 1000},
        'SNAC': {'hyper': None},
        'NBC': {'hyper': None}, 
        'TKNC': {'hyper': 10},
        'TKNP': {'hyper': 10},
        'LSC': {'hyper': 10, 'min_var': 1e-5, 'num_class': num_class},
        'DSC': {'hyper': 0.1, 'min_var': 1e-5, 'num_class': num_class},
        'MDSC': {'hyper': 10, 'min_var': 1e-5, 'num_class': num_class},
        'NLC': {'hyper': None},
        'CC': {'hyper': 10}
    }
    
    for method, config in methods_config.items():
        ratio = calculate_coverage_ratio(
                build_loader, target_loader, method, model,
                hyper=config['hyper'],
                device=device,
                min_var=config.get('min_var', 1e-5),
                num_class=config.get('num_class', num_class)
            )
        results[method] = ratio
        print(f"{method}: {ratio:.4f}")
    return results

def save_csv_results(updated_column_dict, csv_path='results.csv', tag='original'):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        for column_name, value in updated_column_dict.items():
            df.loc[tag, column_name] = value
        df.to_csv(csv_path)
    else:
        df = pd.DataFrame(updated_column_dict, index=[tag])
        mode  = 'a' if os.path.exists(csv_path) else 'w'
        header= False if mode == 'a' else True
        df.to_csv(csv_path, mode=mode, header=header)

    print(f"[{tag}] Updated results saved to {csv_path}")
        
# -----------------------------------------------------------
# Wisdom-based input trace
# -----------------------------------------------------------
def wisdom_importance_scores(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df.Score != 0]
    layer2score = defaultdict(dict)
    for _, row in df.iterrows():
        layer2score[row.LayerName][int(row.NeuronIndex)] = float(row.Score)
    return layer2score

def register_hooks(model, csv_path):
    """
    For every layer that appears in the CSV we grab its output.
    Returns list of handles and a dict that will fill up per batch.
    """
    layer2score = wisdom_importance_scores(csv_path)
    
    activations = defaultdict(list)
    handles = []

    for name, module in model.named_modules():
        if name in layer2score:
            scores_for_layer = layer2score[name]

            def _make_hook(layer_name, score_dict):
                def _hook(_, __, out):
                    # out shape: (B, C, H, W) for conv; (B, C) for FC
                    acts = out.detach()
                    # conv → (B, C, H, W); fc → (B, C, 1, 1)
                    if acts.dim() == 2:
                        acts = acts.unsqueeze(-1).unsqueeze(-1)
                    score_vec = torch.zeros(acts.size(1), device=acts.device)
                    for idx, s in score_dict.items():
                        if idx < score_vec.size(0):
                            score_vec[idx] = s
                    weighted = acts * score_vec.view(1, -1, 1, 1)
                    activations[layer_name].append(weighted)
                return _hook

            handles.append(module.register_forward_hook(
                _make_hook(name, scores_for_layer)))
    return handles, activations

# -----------------------------------------------------------
# Data generation
# -----------------------------------------------------------

class LabelToIntDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dataset):
        self.dataset = tensor_dataset  # e.g., TensorDataset(image_tensor, label_tensor)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y.item()  # convert label from Tensor to int
    
    def __len__(self):
        return len(self.dataset)

def add_gaussian_noise(imgs: torch.Tensor, mask: torch.Tensor,
                       mean=0., std=0.01):
    noise = torch.randn_like(imgs) * std + mean
    return torch.where(mask, imgs + noise, imgs).clamp(torch.min(imgs), torch.max(imgs))
    # return torch.where(mask, imgs + noise, imgs)

def build_mask(attributions, k=0.02, exclude_imp=True):
    """
    Return boolean masks (important and random) with the top‐k fraction (e.g. 0.02 = 2 %)
    of attribution magnitudes set to True (per-sample).
    """
    bs = attributions.size(0)
    flat = attributions.view(bs, -1).abs()
    kth = math.ceil((flat.size(1) * k))
    
    idx = flat.topk(kth, dim=1).indices
    
    # Important-based mask
    mask_imp_flat = torch.zeros_like(flat, dtype=torch.bool)
    mask_imp = mask_imp_flat.scatter_(1, idx, True)
    mask_imp = mask_imp.view_as(attributions)
    
    if exclude_imp:
        # Create a random mask (exclude important pixels)
        avail_mask = (~mask_imp_flat)
        rand_scores = torch.rand_like(flat, dtype=torch.float32)
        rand_scores[~avail_mask] = float('-inf')
        rand_idx = rand_scores.topk(kth, dim=1).indices
        mask_rand = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, rand_idx, True)
        mask_rand = mask_rand.view_as(attributions)
    
    else:
        # Create a random mask
        rand_scores = torch.rand_like(flat, dtype=torch.float32) 
        rand_idx = rand_scores.topk(kth, dim=1).indices   # (B, kth)
        mask_rand = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, rand_idx, True)
        mask_rand = mask_rand.view_as(attributions)
    
    return mask_imp, mask_rand

def build_mask_wisdom(attributions, k=0.02, exclude_imp=True):
    """
    attributions: (B, 1, H, W) – relevance per pixel
    returns boolean mask where top-k fraction is True.
    """
    B, _, H, W = attributions.shape
    flat = attributions.view(B, -1).abs()
    kth = math.ceil((flat.size(1) * k))
    idx = flat.topk(kth, dim=1).indices
    
    # Important-based mask
    mask_imp_flat = torch.zeros_like(flat, dtype=torch.bool)
    mask_imp_flat = mask_imp_flat.scatter_(1, idx, True)
    mask_imp = mask_imp_flat.view(B, 1, H, W)
    
    # Random-based mask
    if exclude_imp:
        avail_mask = (~mask_imp_flat)
        rand_scores = torch.rand_like(flat, dtype=torch.float32)
        rand_scores[~avail_mask] = float('-inf')
        rand_idx = rand_scores.topk(kth, dim=1).indices
        mask_rand = torch.zeros_like(mask_imp_flat, dtype=torch.bool).scatter_(1, rand_idx, True)
        mask_rand = mask_rand.view_as(mask_imp)
    else:
        rand_scores = torch.rand_like(flat, dtype=torch.float32)
        rand_idx = rand_scores.topk(kth, dim=1).indices
        mask_rand = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, rand_idx, True)
        mask_rand = mask_rand.view_as(mask_imp)
        
    return mask_imp, mask_rand

def build_sets(model, loader, device, k, std, name):
    """
    Returns tensors (U_I, U_R, y) for a dataset.
    """
    model.to(device)
    lrp = LRP(model)
    U_I, U_R, y = [], [], []

    for inputs, labels in tqdm(loader, desc=f"Attribution for {name}"):
        inputs, labels = inputs.to(device), labels.to(device)
        # LRP expects prediction target
        preds = model(inputs).argmax(1)
        attributions = lrp.attribute(inputs, target=preds)  # shape = X
        mask_imp, mask_rand = build_mask(attributions, k=k, exclude_imp=True)

        # LRP-based important pixels
        inputs_I = add_gaussian_noise(inputs, mask_imp, std=std)
        # Random-based pixels
        inputs_R = add_gaussian_noise(inputs, mask_rand, std=std) 
        
        U_I.append(inputs_I.cpu())
        U_R.append(inputs_R.cpu())
        y.append(labels.cpu())
    
    U_I_dataset = TensorDataset(torch.cat(U_I), torch.cat(y).long())
    U_R_dataset = TensorDataset(torch.cat(U_R), torch.cat(y).long())
    
    U_I_dataset = LabelToIntDataset(U_I_dataset)  # Convert labels to int
    U_R_dataset = LabelToIntDataset(U_R_dataset)  # Convert labels to int

    return U_I_dataset, U_R_dataset


def build_sets_wisdom(model, loader, device, csv_path, k, std, name):
    """
    Returns tensors (U_I, U_R, y) for a dataset, a WISDOM-based version.
    """
    model.to(device)
    model.eval()
    handles, acts_dict = register_hooks(model, csv_path)
    U_I, U_R, y = [], [], []
    
    for inputs, labels in tqdm(loader, desc=f"Attribution for {name}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        for ac_ in acts_dict:
            acts_dict[ac_].clear()
        
        # Forward pass to get activations
        model(inputs)
        
        # Merge all layers: upsample to input resolution & sum
        heat = torch.zeros_like(inputs[:, :1]) # (B,1,H,W)
        for layer_name, batch_list in acts_dict.items():
            if not batch_list:
                continue
            acts = batch_list[0] # (B,C,h,w)
            if acts.size(2) != inputs.size(2):
                acts = F.interpolate(acts, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
            
            # sum across channels → (B,1,H,W)
            layer_heat = acts.sum(1, keepdim=True)
            heat += layer_heat
        
        mask_imp, mask_rand = build_mask_wisdom(heat, k=k, exclude_imp=True)
        # WISDOM-based important pixels
        inputs_I = add_gaussian_noise(inputs, mask_imp, std=std)
        # Random-based pixels
        inputs_R = add_gaussian_noise(inputs, mask_rand, std=std)
        
        U_I.append(inputs_I.cpu())
        U_R.append(inputs_R.cpu())
        y.append(labels.cpu())
    
    # tidy up hooks
    for h in handles:
        h.remove()
    
    U_I_dataset = TensorDataset(torch.cat(U_I), torch.cat(y))
    U_R_dataset = TensorDataset(torch.cat(U_R), torch.cat(y))
    
    U_I_dataset = LabelToIntDataset(U_I_dataset)  # Convert labels to int
    U_R_dataset = LabelToIntDataset(U_R_dataset)  # Convert labels to int
        
    return U_I_dataset, U_R_dataset
    
# -----------------------------------------------------------
# Run full coverage suite on one dataloader with other methods
# -----------------------------------------------------------

def run_other_coverage_suite(model,
                       build_loader,   # clean data for build phases
                       target_loader,  # UI_loader / UR_loader / etc.
                       num_classes,
                       device,
                       logger,
                       tag: str = 'dataset', *args, **kwargs):
    """
    Returns a dict {metric_name: coverage_value} for the given target_loader.
    """
    
    results = calculate_all_coverage_ratios(build_loader, target_loader, model, device, num_class=num_classes)
    model_name = kwargs.get('model_name')
    dataset_name = kwargs.get('dataset_name')
    df = pd.DataFrame(results, index=[tag])
    logger.info(f"=== Coverage results for {tag} ===")
    logger.info(df.to_string(float_format=lambda x: f"{x:.4f}"))
    
    save_csv_results(results, "rq2_results_{}_{}_{}.csv".format(dataset_name, model_name, TIMESTAMP), tag=tag)
    return results, df

# -----------------------------------------------------------
# IDC coverage testing
# -----------------------------------------------------------
def idc_coverage(args, model, train_loader, test_loader, classes, trainable_module_name, device, logger, tag='original'):
    if args.use_silhouette:
        cluster_info = "silhouette"
    else:
        cluster_info = str(args.n_clusters)
    cache_path = "./cluster_pkl/" + args.model + "_" + args.dataset + "_top_" + str(args.top_m_neurons) + "_cluster_" + cluster_info + "_deepimportance_clusters.pkl"
    layer_relevance_scores = get_relevance_scores_dataloader(
            model,
            train_loader,
            device,
            attribution_method='lrp',
        )
    
    idc = IDC(
        model,
        args.top_m_neurons,
        args.n_clusters,
        args.use_silhouette,
        args.all_class,
        "KMeans",
        None,
        cache_path
    )
    
    final_layer = trainable_module_name[-1]
    important_neuron_indices, inorderd_indices = idc.select_top_neurons_all(layer_relevance_scores, final_layer)
    activation_values, selected_activations = idc.get_activations_model_dataloader(train_loader, important_neuron_indices)
    
    selected_activations = {k: v.half().cpu() for k, v in selected_activations.items()}
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(test_loader, important_neuron_indices, cluster_groups)
    
    results = {}
    results['IDC'] = coverage_rate

    df = pd.DataFrame(results, index=[tag])
    save_csv_results(results, "rq2_results_{}_{}_top_{}_{}.csv".format(args.dataset, args.model, args.top_m_neurons, TIMESTAMP), tag=tag)
    logger.info(f"Total Combination: {total_combination}, Max Coverage: {max_coverage:.4f}, IDC Coverage: {coverage_rate:.4f}, Attribution: {args.attr}")
    return coverage_rate

def wisdom_coverage(args, model, train_loader, test_loader, classes, logger, tag='original'):
    df = pd.read_csv(args.csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(args.top_m_neurons)
    top_k_neurons = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        top_k_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)
    if args.use_silhouette:
        cluster_info = "silhouette"
    else:
        cluster_info = str(args.n_clusters)
    cache_path = "./cluster_pkl/" + args.model + "_" + args.dataset + "_top_" + str(args.top_m_neurons) + "_cluster_" + cluster_info + "_wisdom_clusters.pkl"
    idc = IDC(model, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class, "KMeans", None,  cache_path)

    activation_values, selected_activations = idc.get_activations_model_dataloader(train_loader, top_k_neurons)
    selected_activations = {k: v.half().cpu() for k, v in selected_activations.items()}
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(test_loader, top_k_neurons, cluster_groups)

    results = {}
    results['WISDOM'] = coverage_rate
    df = pd.DataFrame(results, index=[tag])
    save_csv_results(results, "rq2_results_{}_{}_top_{}_{}.csv".format(args.dataset, args.model, args.top_m_neurons, TIMESTAMP), tag=tag)
    logger.info(f"Total Combination: {total_combination}, Max Coverage: {max_coverage:.4f}, IDC Coverage: {coverage_rate:.4f}, Attribution: WISDOM")
    return coverage_rate

def run_idc_suite(args, model, trainable_module_name, train_loader, test_loader, U_IO_loader, U_RO_loader, device, logger, classes):
    """
    Runs the IDC coverage suite on the given model and data loaders.
    """
    dpo_results = idc_coverage(args, model, train_loader, test_loader, classes, trainable_module_name, device, logger, tag='original')
    dpr_results = idc_coverage(args, model, train_loader, U_RO_loader, classes, trainable_module_name, device, logger, tag=args.attr+'_U_RO')
    dpi_results = idc_coverage(args, model, train_loader, U_IO_loader, classes, trainable_module_name, device, logger, tag=args.attr+'_U_IO')
    
    dpo_results_w = wisdom_coverage(args, model, train_loader, test_loader, classes, logger, tag='original')
    dpr_results_w = wisdom_coverage(args, model, train_loader, U_RO_loader, classes, logger, tag=args.attr+'_U_RO')
    dpi_results_w = wisdom_coverage(args, model, train_loader, U_IO_loader, classes, logger, tag=args.attr+'_U_IO')
    

def run_coverage_suite(args, model, train_loader, test_loader, U_IO_loader, U_RO_loader, device, classes, logger, tag_pre='lrp_'):
    """
    Runs the full coverage suite on the given model and data loaders.
    """
    uo_results, _ = run_other_coverage_suite(model, train_loader, test_loader, len(classes), device, logger, tag='original', skip_train=True, model_name=args.model, dataset_name=args.dataset)
    ur_results, _ = run_other_coverage_suite(model, train_loader, U_RO_loader, len(classes), device, logger, tag=tag_pre + 'U_RO', skip_train=True, model_name=args.model, dataset_name=args.dataset)
    ui_results, _ = run_other_coverage_suite(model, train_loader, U_IO_loader, len(classes), device, logger, tag=tag_pre + 'U_IO', skip_train=True, model_name=args.model, dataset_name=args.dataset)

# -----------------------------------------------------------
# Main entry point
# -----------------------------------------------------------
def main():
    
    set_seed()
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    
    # Model settings
    model, module_name, module, trainable_module, trainable_module_name, logger = prapare_data_models(args)

    # Data settings
    train_loader, test_loader, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path)
    
    if args.attr == 'wisdom':
        U_I_dataset, U_R_dataset = build_sets_wisdom(model, test_loader, device, args.csv_file, TOPK, gausian_STD, args.dataset)
        U_IO_dataset = ConcatDataset([test_dataset, U_I_dataset])   # original + important
        U_RO_dataset = ConcatDataset([test_dataset, U_R_dataset])   # original + random
    else:
        U_I_dataset, U_R_dataset = build_sets(model, test_loader, device, TOPK, gausian_STD, args.dataset)
        U_IO_dataset = ConcatDataset([test_dataset, U_I_dataset])   # original + important
        U_RO_dataset = ConcatDataset([test_dataset, U_R_dataset])   # original + random
    
    U_IO_loader = DataLoader(U_IO_dataset, batch_size=args.batch_size, shuffle=False)
    U_RO_loader = DataLoader(U_RO_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"[Sanity] Generated datasets: U_I: {len(U_I_dataset)}, U_R: {len(U_R_dataset)}, U_IO: {len(U_IO_dataset)}, U_RO: {len(U_RO_dataset)}")

    # Run the coverage suite
    logger.info("=== Running coverage suite ===")
    run_coverage_suite(args, model, train_loader, test_loader, U_IO_loader, U_RO_loader, device, classes, logger, tag_pre=args.attr + '_')
    run_idc_suite(args, model, trainable_module_name, train_loader, test_loader, U_IO_loader, U_RO_loader, device, logger, classes)
    

if __name__ == '__main__':
    main()
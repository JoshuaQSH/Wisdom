import random
import time
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from collections import defaultdict
from captum.attr import LRP

from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
from src.attribution import get_relevance_scores_dataloader
from src.utils import get_data, parse_args, get_model, eval_model_dataloder, get_trainable_modules_main
from src.idc import IDC
from src.nlc_coverage import (
    NC, KMNC, NBC, SNAC, TKNC, TKNP, CC,
    NLC, LSC, DSC, MDSC
)
from src.nlc_tool import get_layer_output_sizes


"""
U_I (Importance-perturbed): Each image has Gaussian white noise (mean 0, std 0.3) added to its most important 2% pixels.
U_R (Random-perturbed): Each image has noise added to a random 2% of its pixels.

U_IO (Importance-perturbed Dataset + Original Dataset): Original images with noise added to the most important 2% pixels.
U_RO (Random-perturbed Dataset + Original Dataset): Original images with noise added to a random 2% of its pixels.

# Step 1: Get the relevance maps for the test set using Attribution methods (e.g., LRP).
# Step 2: For each image, find the top 2% most important pixels based on the relevance map (Baseline: random & LRP) and form the new dataset.
# Step 3: Coverage testing for the perturbed dataset.

"""

# python run_rq_2_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --batch-size 128 --device 'cuda:0' --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/lenet_cifar_b32.csv' --idc-test-all --attr lrp --top-m-neurons 10 --use-silhouette

# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------

SEED = 2025  # Random seed for reproducibility
TOPK = 0.02  # Top-k fraction of pixels to perturb (2%)
SANITY_CHECK = False
start_ms = int(time.time() * 1000)
TIMESTAMP = time.strftime("%Y%m%d‑%H%M%S", time.localtime(start_ms / 1000))
acts = defaultdict(list)

def vis_santity_check(dataloder1, dataloader2, classes, n_per_class=5):
    visualize_pairs(dataloder1, dataloader2, n_per_class=n_per_class)
    avg_dist, per_class = image_distance_visualization(dataloder1, dataloader2, n_classes=len(classes), show_plot=True)
    print(f"Average perturbation distance: {avg_dist:.4f}")
    print(f"Per-class perturbation distances: {per_class}")

def prapare_data_models(args):
    ### Model settings
    model_path = os.getenv("HOME") + args.saved_model
    
    ### Model loading
    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)

    return model, module_name, module, trainable_module, trainable_module_name

# A toy train loader for pre-assessing the coverage methods (e.g., CC), use as needed
def toy_train_loader(train_dataset, batch_size, ratio=0.2):
    n_build = int(len(train_dataset) * ratio)
    n_rest  = len(train_dataset) - n_build
    build_set_A, _ = random_split(train_dataset, [n_build, n_rest])
    build_loader_A = DataLoader(build_set_A, batch_size=batch_size, shuffle=True)
    return build_loader_A

def visualize_pairs(original_loader,
                    perturbed_loader,
                    class_names=None,
                    n_per_class: int = 4,
                    figsize=(10, 6)):
    # lbl → list[(orig, pert)]
    collected = {}                                     
    for (x, y), (x_t, y_t) in zip(original_loader, perturbed_loader):
        assert torch.equal(y, y_t), "Loaders out of sync!"
        for orig, pert, lbl in zip(x, x_t, y):
            lbl = lbl.item()
            if lbl not in collected:
                collected[lbl] = []
            if len(collected[lbl]) < n_per_class:
                collected[lbl].append((orig, pert))
        # early stop when every class satisfied
        if all(len(v) == n_per_class for v in collected.values()):
            break

    # flatten into [orig1, pert1, orig2, pert2, …] keeping class order
    grid_imgs, y_ticks = [], []
    for lbl in sorted(collected.keys()):
        for orig, pert in collected[lbl]:
            grid_imgs.extend([orig, pert])
        y_ticks.append(lbl)

    # 2 images per pair  →  nrow = 2*n_per_class
    grid = make_grid(torch.stack(grid_imgs),
                     nrow=2 * n_per_class,
                     padding=2)
    npimg = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=figsize)
    cmap = 'gray' if npimg.shape[2] == 1 else None
    plt.imshow(npimg, cmap=cmap)
    plt.axis('off')

    # optional class labels on left margin
    if class_names is None:
        class_names = [str(i) for i in y_ticks]
    step = grid.size(2) // (n_per_class * 2)
    for row, lbl in enumerate(y_ticks):
        y = row * step + step * 0.5
        plt.text(-5, y, class_names[lbl], va='center', ha='right')
    plt.title("Original (left) vs. perturbed (right) – each pair")
    plt.savefig(f"perturbation_samples.pdf", dpi=1200, format='pdf')

def image_distance_visualization(original_loader, perturbed_loader, n_classes=10, show_plot=True):
    """
    Visualizes the original and perturbed images side by side.
    """
    class_sum  = np.zeros(n_classes, dtype=np.float64)
    class_cnt  = np.zeros(n_classes, dtype=np.int64)
    
    for (x, y), (x_tilde, y_2) in zip(original_loader, perturbed_loader):
        assert torch.equal(y, y_2), "Loaders are out of sync!"
        diff = (x_tilde - x).view(x.size(0), -1)
        d = diff.norm(p=2, dim=1).cpu().numpy()   # (B,)
        for lbl in range(n_classes):
            mask = (y.cpu().numpy() == lbl)
            class_sum[lbl] += d[mask].sum()
            class_cnt[lbl] += mask.sum()
            
    per_class = class_sum / np.maximum(class_cnt, 1)  # avoid div-by-zero
    avg_dist  = per_class[class_cnt > 0].mean()
    
    if show_plot:
        plt.figure(figsize=(8,4))
        plt.bar(range(n_classes), per_class, tick_label=list(range(n_classes)))
        plt.ylabel("L2-distance")
        plt.xlabel("Class")
        plt.title("Perturbation strength per class")
        plt.savefig(f"perturbation_strength.pdf", dpi=1200, format='pdf')
    
    return avg_dist, per_class

def set_seed(seed: int = 2025):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _infer_layer_sizes(model: torch.nn.Module, sample_batch, device):
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    num_neuron = 0
    layer_size_dict = get_layer_output_sizes(model, sample_batch.to(device))
    
    for layer_name in layer_size_dict.keys():
        num_neuron += layer_size_dict[layer_name][0]
    print('Total %d layers: ' % len(layer_size_dict.keys()))
    print('Total %d neurons: ' % num_neuron)
    
    return layer_size_dict

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
def _spawn_coverage_objects(model, layer_size_dict, num_classes):
    return {
        # --- plain neuron-coverage family --------------------
        'NC'   : NC  (model, layer_size_dict, hyper=0.5),
        'KMNC' : KMNC(model, layer_size_dict, hyper=1000),
        'NBC'  : NBC (model, layer_size_dict),
        'SNAC' : SNAC(model, layer_size_dict),
        'TKNC' : TKNC(model, layer_size_dict, hyper=10),
        'TKNP' : TKNP(model, layer_size_dict, hyper=10), 
        'CC'   : CC  (model, layer_size_dict, hyper=10),

        # --- statistical / surprise-based --------------------
        'NLC'  : NLC (model, layer_size_dict),                      # no hyper
        'LSC'  : LSC (model, layer_size_dict,
                      hyper=10, min_var=1e-5, num_class=num_classes),
        'DSC'  : DSC (model, layer_size_dict,
                      hyper=0.1, min_var=1e-5, num_class=num_classes),
        'MDSC' : MDSC(model, layer_size_dict,
                      hyper=10, min_var=1e-5, num_class=num_classes),
    }

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

def build_mask(attributions: torch.Tensor, k: float = 0.02):
    """
    Return boolean masks (important and random) with the top‐k fraction (e.g. 0.02 = 2 %)
    of attribution magnitudes set to True (per-sample).
    """
    bs = attributions.size(0)
    flat = attributions.view(bs, -1).abs()
    kth = math.ceil((flat.size(1) * k))
    
    idx = flat.topk(kth, dim=1).indices
    
    # Create a random mask
    rand_scores = torch.rand_like(flat, dtype=torch.float32) 
    rand_idx = rand_scores.topk(kth, dim=1).indices   # (B, kth)
    mask_rand = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, rand_idx, True)
    mask_rand = mask_rand.view_as(attributions)
    
    # Important-based mask
    mask = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True)
    mask = mask.view_as(attributions)
    
    return mask, mask_rand

def build_mask_wisdom(heat: torch.Tensor, k: float = 0.02) -> torch.Tensor:
    """
    heat: (B, 1, H, W) – relevance per pixel
    returns boolean mask where top-k fraction is True.
    """
    B, _, H, W = heat.shape
    flat = heat.view(B, -1).abs()
    kth = math.ceil((flat.size(1) * k))
    idx = flat.topk(kth, dim=1).indices
    
    rand_scores = torch.rand_like(flat, dtype=torch.float32)
    rand_idx = rand_scores.topk(kth, dim=1).indices
    mask_rand = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, rand_idx, True)
    
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    mask = mask.view(B, 1, H, W)
    mask_rand = mask_rand.view_as(mask)
    
    return mask, mask_rand

def add_gaussian_noise(imgs: torch.Tensor, mask: torch.Tensor,
                       mean=0., std=0.1):
    noise = torch.randn_like(imgs) * std + mean
    return torch.where(mask, imgs + noise, imgs).clamp(0., 1.)

def build_sets(model, loader, device, k, name):
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
        mask_imp, mask_rand = build_mask(attributions, k=k)

        # LRP-based important pixels
        inputs_I = add_gaussian_noise(inputs, mask_imp)
        # Random-based pixels
        inputs_R = add_gaussian_noise(inputs, mask_rand) 
        
        U_I.append(inputs_I.cpu())
        U_R.append(inputs_R.cpu())
        y.append(labels.cpu())
    
    U_I_dataset = TensorDataset(torch.cat(U_I), torch.cat(y))
    U_R_dataset = TensorDataset(torch.cat(U_R), torch.cat(y))

    return U_I_dataset, U_R_dataset

def build_sets_wisdom(model, loader, device, csv_path, k, name):
    """
    Returns tensors (U_I, U_R, y) for a dataset, a WISDOM-based version.
    """
    model.to(device)
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
        
        mask_imp, mask_rand = build_mask_wisdom(heat, k=k)
        # WISDOM-based important pixels
        inputs_I = add_gaussian_noise(inputs, mask_imp)
        # Random-based pixels
        inputs_R = add_gaussian_noise(inputs, mask_rand)
        
        U_I.append(inputs_I.cpu())
        U_R.append(inputs_R.cpu())
        y.append(labels.cpu())
    
    # tidy up hooks
    for h in handles:
        h.remove()
    
    U_I_dataset = TensorDataset(torch.cat(U_I), torch.cat(y))
    U_R_dataset = TensorDataset(torch.cat(U_R), torch.cat(y))
    
    return U_I_dataset, U_R_dataset

def eval_model(model, test_loader, U_I_loader, U_R_loader, device):
    model.eval()
    original_accuracy, original_avg_loss, original_f1 = eval_model_dataloder(model, test_loader, device)
    accuracy_I, avg_loss_I, f1_I = eval_model_dataloder(model, U_I_loader, device)
    accuracy_R, avg_loss_R, f1_R = eval_model_dataloder(model, U_R_loader, device)
    
    print(f"Original Accuracy: {original_accuracy:.4f}, Average Loss: {original_avg_loss:.4f}, F1 Score: {original_f1:.4f}")
    print(f"Accuracy on U_I: {accuracy_I:.4f}, Average Loss on U_I: {avg_loss_I:.4f}, F1 Score on U_I: {f1_I:.4f}")
    print(f"Accuracy on U_R: {accuracy_R:.4f}, Average Loss on U_R: {avg_loss_R:.4f}, F1 Score on U_R: {f1_R:.4f}")
    

# -----------------------------------------------------------
# Run full coverage suite on one dataloader with other methods
# -----------------------------------------------------------
def quick_patch(name, val, target_loader, total_buckets, cc_ref):
    # LSC counts the number of test inputs, thus requires to divide by the number of samples
    if name == 'LSC':
        val = val / len(target_loader.dataset)
    # DSC and MDSC are special cases where we normalize by the number of buckets
    if name in ('DSC', 'MDSC'):
        val = val / total_buckets
    if name == 'TKNP':
        val = val / len(target_loader.dataset)
    if name == 'NLC':
        val = val / total_buckets
    if name == 'CC':
        val = val / cc_ref
    
    return val

def run_other_coverage_suite(model,
                       build_loader,   # clean data for build phases
                       target_loader,  # UI_loader / UR_loader / etc.
                       num_classes,
                       device,
                       skip_train: bool = True,
                       tag: str = 'dataset', *args, **kwargs):
    """
    Returns a dict {metric_name: coverage_value} for the given target_loader.
    """
    # Discover layer sizes
    sample_batch, *_ = next(iter(build_loader))
    layer_size_dict = _infer_layer_sizes(model, sample_batch, device)
    # A hack here, we assume 1000 buckets for DSC/MDSC
    total_buckets = 1000
    cc_ref = 22000
    
    model_name = kwargs.get('model_name')
    dataset_name = kwargs.get('dataset_name')
    
    # Create all metrics
    cov_objs = _spawn_coverage_objects(model, layer_size_dict, num_classes)

    # Optional build() for metrics that need reference stats
    print(f"[{tag}] Building reference statistics …")
    for name, cov in cov_objs.items():
        try:
            cov.build(build_loader)
        except Exception as e:
            warnings.warn(f"{name}.build() failed ({e}); continuing without build.")
    
    # TODO: For LSC/DSC/MDSC/CC/TKNP, initialization with train loader is required, but it is quite slow, we skip it for now
    if name not in ['CC', 'TKNP', 'LSC', 'DSC', 'MDSC'] and not skip_train:
        cov.assess(build_loader)
    
    # if name == 'CC':
    #     cov.assess(build_loader)
    #     cc_ref = sum(len(v) for v in cov.distant_dict.values())
    #     print(f"[{tag}] CC reference size: {cc_ref}")
    
    # Assess coverage on target set
    print(f"[{tag}] Assessing coverage on target loader …")
    results = {}
    for name, cov in cov_objs.items():
        try:
            cov.assess(target_loader)
            val = cov.current

            if torch.is_tensor(val):
                val = val.item()
            val = quick_patch(name, val, target_loader, total_buckets, cc_ref)
            results[name] = val
        except Exception as e:
            warnings.warn(f"{name}.assess() failed ({e}); value set to NaN.")
            results[name] = float('nan')

    df = pd.DataFrame(results, index=[tag])
    print("\n=== Coverage results for", tag, "===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    
    save_csv_results(results, "rq2_results_{}_{}_{}.csv".format(dataset_name, model_name, TIMESTAMP), tag=tag)
    return results, df

# -----------------------------------------------------------
# IDC coverage testing
# -----------------------------------------------------------
def idc_coverage(args, model, train_loader, test_loader, classes, trainable_module_name, device, tag='original'):
    
    layer_relevance_scores = get_relevance_scores_dataloader(
            model,
            train_loader,
            device,
            attribution_method='lrp',
        )
    
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
    activation_values, selected_activations = idc.get_activations_model_dataloader(test_loader, important_neuron_indices)
    
    selected_activations = {k: v.cpu() for k, v in selected_activations.items()}
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(test_loader, important_neuron_indices, cluster_groups)
    
    results = {}
    results['IDC'] = coverage_rate
    df = pd.DataFrame(results, index=[tag])
    save_csv_results(results, "rq2_results_{}_{}_{}.csv".format(args.dataset, args.model, TIMESTAMP), tag=tag)
    print(f"Total Combination: {total_combination}, Max Coverage: {max_coverage:.4f}, IDC Coverage: {coverage_rate:.4f}, Attribution: {args.attr}")
    return coverage_rate

def wisdom_coverage(args, model, test_loader, device, classes, tag='original'):
    df = pd.read_csv(args.csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(args.top_m_neurons)
    top_k_neurons = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        top_k_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)
    
    idc = IDC(model, classes, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class, "KMeans")

    activation_values, selected_activations = idc.get_activations_model_dataloader(test_loader, top_k_neurons)
    selected_activations = {k: v.cpu() for k, v in selected_activations.items()}
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(test_loader, top_k_neurons, cluster_groups)

    results = {}
    results['WISDOM'] = coverage_rate
    df = pd.DataFrame(results, index=[tag])
    save_csv_results(results, "rq2_results_{}_{}_{}.csv".format(args.dataset, args.model, TIMESTAMP), tag=tag)
    print(f"Total Combination: {total_combination}, Max Coverage: {max_coverage:.4f}, IDC Coverage: {coverage_rate:.4f}, Attribution: WISDOM")
    return coverage_rate

def run_idc_suite(args, model, trainable_module_name, train_loader, test_loader, U_IO_loader, U_RO_loader, device, classes):
    """
    Runs the IDC coverage suite on the given model and data loaders.
    """
    dpo_results = idc_coverage(args, model, train_loader, test_loader, classes, trainable_module_name, device, tag='original')
    dpr_results = idc_coverage(args, model, train_loader, U_RO_loader, classes, trainable_module_name, device, tag=args.attr+'_U_RO')
    dpi_results = idc_coverage(args, model, train_loader, U_IO_loader, classes, trainable_module_name, device, tag=args.attr+'_U_IO')
    
    dpo_results_w = wisdom_coverage(args, model, test_loader, device, classes, tag='original')
    dpr_results_w = wisdom_coverage(args, model, U_RO_loader, device, classes, tag=args.attr+'_U_RO')
    dpi_results_w = wisdom_coverage(args, model, U_IO_loader, device, classes, tag=args.attr+'_U_IO')
    

def run_coverage_suite(model, train_loader, test_loader, U_IO_loader, U_RO_loader, device, classes, tag_pre='lrp_'):
    """
    Runs the full coverage suite on the given model and data loaders.
    """
    uo_results, _ = run_other_coverage_suite(model, train_loader, test_loader, len(classes), device, tag='original', skip_train=True, model_name=args.model, dataset_name=args.dataset)
    ur_results, _ = run_other_coverage_suite(model, train_loader, U_RO_loader, len(classes), device, tag=tag_pre + 'U_RO', skip_train=True, model_name=args.model, dataset_name=args.dataset)
    ui_results, _ = run_other_coverage_suite(model, train_loader, U_IO_loader, len(classes), device, tag=tag_pre + 'U_IO', skip_train=True, model_name=args.model, dataset_name=args.dataset)

# -----------------------------------------------------------
# Main entry point
# -----------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    
    ### Model settings
    model, module_name, module, trainable_module, trainable_module_name = prapare_data_models(args)

    ### Data settings
    train_loader, test_loader, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path, args.large_image)
    if args.attr == 'wisdom':
        U_I_dataset, U_R_dataset = build_sets_wisdom(model, test_loader, device, args.csv_file, TOPK, args.dataset)
        U_IO_dataset = ConcatDataset([test_dataset, U_I_dataset])   # original + important
        U_RO_dataset = ConcatDataset([test_dataset, U_R_dataset])   # original + random
    else:
        U_I_dataset, U_R_dataset = build_sets(model, test_loader, device, TOPK, args.dataset)
        U_IO_dataset = ConcatDataset([test_dataset, U_I_dataset])   # original + important
        U_RO_dataset = ConcatDataset([test_dataset, U_R_dataset])   # original + random
    
    build_loader_toy = toy_train_loader(train_dataset, args.batch_size, ratio=0.2)  # Optional: for CC or other methods that need a build loader
    
    U_I_loader = DataLoader(U_I_dataset, batch_size=args.batch_size, shuffle=False)
    U_R_loader = DataLoader(U_R_dataset, batch_size=args.batch_size, shuffle=False)
    U_IO_loader = DataLoader(U_IO_dataset, batch_size=args.batch_size, shuffle=False)
    U_RO_loader = DataLoader(U_RO_dataset, batch_size=args.batch_size, shuffle=False)
    
    ### Helper - visualization and distance between original and perturbed images
    if SANITY_CHECK:
        vis_santity_check(test_loader, U_I_loader, classes, n_per_class=4)
    
    # A simple acc test for the perturbed datasets
    eval_model(model, test_loader, U_IO_loader, U_RO_loader, device)
    
    # Run the coverage suite
    print("\n=== Running coverage suite ===")
    # run_coverage_suite(model, build_loader_toy, test_loader, U_IO_loader, U_RO_loader, device, classes, tag_pre=args.attr + '_')
    run_idc_suite(args, model, trainable_module_name, train_loader, test_loader, U_IO_loader, U_RO_loader, device, classes)
    

if __name__ == '__main__':
    set_seed(SEED)
    args = parse_args()
    main(args)
    
    
    
    
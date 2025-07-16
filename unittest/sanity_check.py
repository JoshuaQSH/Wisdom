import random
import time
import os
import math
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from collections import defaultdict
from captum.attr import LRP


from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from src.attribution import get_relevance_scores_dataloader
from src.utils import get_data, parse_args, get_model, eval_model_dataloder, get_trainable_modules_main, _configure_logging, viz_attr
from src.idc import IDC
import src.idc_old

import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------

TOPK = 0.02  # Top-k fraction of pixels to perturb (2%)
cmap = ['PuBuGn', 'Greens', 'Purples', 'Reds', 'Blues', 'YlGn', 'summer', 'cool', 'bwr']
start_ms = int(time.time() * 1000)
TIMESTAMP = time.strftime("%Y%m%d‑%H%M%S", time.localtime(start_ms / 1000))
acts = defaultdict(list)

def analyze_model(model, input_size=(1, 3, 224, 224)):
    total_params = sum(p.numel() for p in model.parameters())
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer_count = sum(1 for _ in model.modules())

    # Forward hook to capture activations for neuron count
    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activations.append(output)
        elif isinstance(output, (tuple, list)):
            activations.extend(o for o in output if isinstance(o, torch.Tensor))

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.ReLU, nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hooks.append(layer.register_forward_hook(hook_fn))

    # Run dummy input through the model
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        model(dummy_input)

    # Estimate number of neurons from activation maps
    neuron_count = sum(a.numel() for a in activations)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    return {
        'Total Parameters': total_params,
        'Learnable Parameters': learnable_params,
        'Total Layers (nn.Module)': layer_count,
        'Estimated Neurons (from activations)': neuron_count
    }

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
# Visualization
# -----------------------------------------------------------

def viz_attr_check(args, model, test_loader):
    model.eval()
    lrp = LRP(model)
    input, target = next(iter(test_loader))
    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    attributions_lrp = lrp.attribute(input, target=pred_label_idx)
    viz_attr(args, input[0], attributions_lrp[0], args.dataset, args.model, with_original=False)

def viz_attr_diff(args, logger, orig_loader, pert_loader, cmap="bwr", alpha=0.8, tag='random'):
    
    idx = random.randrange(len(orig_loader.dataset))
    img_orig, y = orig_loader.dataset[1234]
    img_pert, _ = pert_loader.dataset[1234]
    
    # tensors -> HWC numpy
    o = img_orig.cpu().detach().numpy().transpose(1,2,0)
    p = img_pert.cpu().detach().numpy().transpose(1,2,0)
    d = np.abs(p - o)

    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].imshow(o.squeeze() if o.shape[2]==1 else o)
    ax[0].set_title("Original")
    ax[1].imshow(p.squeeze() if p.shape[2]==1 else p)
    ax[1].set_title(f"Perturbed ({tag})")
    ax[2].imshow(o.squeeze() if o.shape[2]==1 else o, alpha=1-alpha)
    ax[2].imshow(d.squeeze() if d.shape[2]==1 else d,
                 cmap=cmap, alpha=alpha)
    ax[2].set_title(f"Overlay ({tag})")
    for a in ax: a.axis("off")

    fig.tight_layout()
    out = f"{args.dataset}_{args.model}_TOP_{TOPK}_{args.attr}_{tag}_diff.pdf"
    fig.savefig(out, dpi=1200, format='pdf', bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[viz] saved to {out}")

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

def build_mask_old(attributions, k=0.02):
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

def build_mask_wisdom_old(heat, k=0.02):
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

def build_mask(attributions: torch.Tensor, k: float = 0.02, exclude_imp: bool = True) -> tuple:
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
        # rand_idx = torch.multinomial(avail_mask, kth, replacement=False)
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

def build_mask_wisdom(heat: torch.Tensor, k: float = 0.02, exclude_imp: bool = True) -> torch.Tensor:
    """
    heat: (B, 1, H, W) – relevance per pixel
    returns boolean mask where top-k fraction is True.
    """
    B, _, H, W = heat.shape
    flat = heat.view(B, -1).abs()
    kth = math.ceil((flat.size(1) * k))
    idx = flat.topk(kth, dim=1).indices
    
    # Important-based mask
    mask_imp_flat = torch.zeros_like(flat, dtype=torch.bool)
    mask_imp_flat = mask_imp_flat.scatter_(1, idx, True)
    mask_imp = mask_imp_flat.view(B, 1, H, W)
    
    # Random-based mask
    if exclude_imp:
        avail_mask = (~mask_imp_flat)
        # rand_idx = torch.multinomial(avail_mask, kth, replacement=False)
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

def add_gaussian_noise(imgs: torch.Tensor, mask: torch.Tensor,
                       mean=0., std=0.01):
    noise = torch.randn_like(imgs) * std + mean
    return torch.where(mask, imgs + noise, imgs).clamp(torch.min(imgs), torch.max(imgs))

def build_sets(model, loader, device, k, std, name):
    """
    Returns tensors (U_I, U_R, y) for a dataset.
    """
    model.to(device)
    model.eval()
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

def eval_model(model, test_loader, U_I_loader, U_R_loader, device, logger):
    model.eval()
    original_accuracy, original_avg_loss, original_f1 = eval_model_dataloder(model, test_loader, device)
    accuracy_I, avg_loss_I, f1_I = eval_model_dataloder(model, U_I_loader, device)
    accuracy_R, avg_loss_R, f1_R = eval_model_dataloder(model, U_R_loader, device)
    
    logger.info(f"Original Accuracy: {original_accuracy:.4f}, Average Loss: {original_avg_loss:.4f}, F1 Score: {original_f1:.4f}")
    logger.info(f"Accuracy on U_I: {accuracy_I:.4f}, Average Loss on U_I: {avg_loss_I:.4f}, F1 Score on U_I: {f1_I:.4f}")
    logger.info(f"Accuracy on U_R: {accuracy_R:.4f}, Average Loss on U_R: {avg_loss_R:.4f}, F1 Score on U_R: {f1_R:.4f}")
    

# -----------------------------------------------------------
# IDC coverage testing
# -----------------------------------------------------------
def idc_coverage(args, model, train_loader, test_loader, trainable_module_name, device, logger, tag='original'):
    if args.use_silhouette:
        cluster_info = "silhouette"
    else:
        cluster_info = str(args.n_clusters)
    cache_path = "./cluster_pkl/" + args.model + "_" + args.dataset + "_top_" + str(args.top_m_neurons) + "_cluster_" + cluster_info + "_deepimportance_clusters.pkl"
    extra = dict(
        n_clusters = args.n_clusters,    # same as IDC’s n_clusters, but OK to repeat
        random_state = 42,   # fixes RNG
        n_init = 10    # keep best of 10 centroid seeds
    )
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
        extra,
        cache_path
    )
    
    final_layer = trainable_module_name[-1]
    important_neuron_indices, inorderd_indices = idc.select_top_neurons_all(layer_relevance_scores, final_layer)
    activation_values, selected_activations = idc.get_activations_model_dataloader(train_loader, important_neuron_indices)
    
    # quantize activations to half precision for memory efficiency
    selected_activations = {k: v.half().cpu() for k, v in selected_activations.items()}
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(test_loader, important_neuron_indices, cluster_groups)
    
    results = {}
    results['IDC'] = coverage_rate
    df = pd.DataFrame(results, index=[tag])
    save_csv_results(results, "rq2_sanity_{}_{}_{}.csv".format(args.dataset, args.model, TIMESTAMP), tag=tag)
    logger.info(f"Total Combination: {total_combination}, Max Coverage: {max_coverage:.4f}, IDC Coverage: {coverage_rate:.4f}, Attribution: {args.attr}")
    return coverage_rate

def wisdom_coverage(args, model, train_loader, test_loader, logger, tag='original'):
    df = pd.read_csv(args.csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(args.top_m_neurons)
    top_k_neurons = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        top_k_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)
    if args.use_silhouette:
        cluster_info = "silhouette"
    else:
        cluster_info = str(args.n_clusters)
    extra = dict(
        n_clusters =  args.n_clusters,    # same as IDC’s n_clusters, but OK to repeat
        random_state = 42,   # fixes RNG
        n_init = 10    # keep best of 10 centroid seeds
    )

    cache_path = "./cluster_pkl/" + args.model + "_" + args.dataset + "_top_" + str(args.top_m_neurons) + "_cluster_" + cluster_info + "_wisdom_clusters.pkl"
    idc = IDC(model, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class, "KMeans", extra, cache_path)

    activation_values, selected_activations = idc.get_activations_model_dataloader(train_loader, top_k_neurons)
    # selected_activations = {k: v.cpu() for k, v in selected_activations.items()}
    selected_activations = {k: v.half().cpu() for k, v in selected_activations.items()}
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(test_loader, top_k_neurons, cluster_groups)

    results = {}
    results['WISDOM'] = coverage_rate
    df = pd.DataFrame(results, index=[tag])
    save_csv_results(results, "rq2_sanity_{}_{}_{}.csv".format(args.dataset, args.model, TIMESTAMP), tag=tag)
    logger.info(f"Total Combination: {total_combination}, Max Coverage: {max_coverage:.4f}, IDC Coverage: {coverage_rate:.4f}, Attribution: WISDOM")
    return coverage_rate

# thin wrapper for running IDC and WISDOM on dataloader
def _run_idc_for_loader(tag, loader, args, model,
                        train_loader, trainable_module_name,
                        device, logger):
    """Compute both IDC and WISDOM on the given loader."""
    idc_val = idc_coverage(args, model, train_loader,
                           loader,
                           trainable_module_name, device, logger, tag=tag)
    wisdom_val = wisdom_coverage(args, model, train_loader,
                                 loader, logger, tag=tag)
    return dict(IDC=idc_val, WISDOM=wisdom_val)

# -----------------------------------------------------------
# 1.  DUPLICATED DATASET:  U_O  +  U_O
# -----------------------------------------------------------
def duplicated_testset_coverage(test_dataset, args, model,
                                train_loader, trainable_module_name,
                                device, logger):
    """U_O + U_O  (concatenated)"""
    dup_dataset = ConcatDataset([test_dataset, test_dataset])
    dup_loader = DataLoader(dup_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info("[Sanity] Evaluating duplicated test set  (U_O + U_O)")
    return _run_idc_for_loader("UO_DUP",
                               dup_loader, args, model,
                               train_loader, trainable_module_name,
                               device, logger)

# -----------------------------------------------------------
# 2.  PARTIAL-IMPORTANCE DATASETS:  U_O + U_I[:N]
# -----------------------------------------------------------
def partial_importance_coverage(test_dataset, U_I_dataset, args, model,
                                train_loader, trainable_module_name,
                                device, logger,
                                sizes=(100, 500, 1000, 2000)):
    """
    Build several datasets of the form  U_O + U_I[:N]
    sizes is an iterable of N values
    """
    results = {}
    for N in sizes:
        tag = f"UO_UI{N}"
        logger.info(f"[Sanity] Evaluating  U_O + U_I[:{N}]")
        subset     = torch.utils.data.Subset(U_I_dataset,
                                             range(min(N, len(U_I_dataset))))
        concat_set = ConcatDataset([test_dataset, subset])
        concat_ld  = DataLoader(concat_set,
                                batch_size=args.batch_size, shuffle=False)
        res = _run_idc_for_loader(tag, concat_ld, args, model, 
                                  train_loader, trainable_module_name,
                                  device, logger)
        results[tag] = res
    return results

# -----------------------------------------------------------
# 3.  HIGH‑TOPK or HIGH‑NOISE  (U_O + U_I_new)
# -----------------------------------------------------------
def high_param_importance_coverage(test_dataset, args, model,
                                   test_loader,
                                   train_loader, trainable_module_name,
                                   device, logger,
                                   new_topk=[0.05, 0.1, 0.2, 0.25, 0.3, 0.5], new_std=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Re‑generate U_I with a *higher* top‑k fraction or noise std,
    then concatenate with U_O and evaluate.
    """
    res_list = []
    for k_ in new_topk:
        for std_ in new_std:
            logger.info(f"[Sanity] Building high‑param U_I  (topk={k_}, std={std_})")

            if args.attr == "wisdom":
                U_I_dataset, U_R_dataset = build_sets_wisdom(model, test_loader,
                                                device, args.csv_file,
                                                k=k_, std=std_, name="highUI_WISDOM")
            else:
                U_I_dataset, U_R_dataset = build_sets(model, test_loader, device,
                                            k=k_, std=std_, name="highUI_LRP")
            
            U_IO_dataset = ConcatDataset([test_dataset, U_I_dataset])   # original + important
            U_RO_dataset = ConcatDataset([test_dataset, U_R_dataset])   # original + random
            U_IO_loader = DataLoader(U_IO_dataset, batch_size=args.batch_size, shuffle=False)
            U_RO_loader = DataLoader(U_RO_dataset, batch_size=args.batch_size, shuffle=False)
            
            tag = f"UO_UI_highT_Top_{k_}_std_{std_}"
            res = _run_idc_for_loader(tag, U_IO_loader, args, model,
                               train_loader, trainable_module_name,
                               device, logger)
            res_list.append(res)
            logger.info(f"[Sanity] Results for {tag}: {res}")

def keys_diff(s1, s2, logger):
    # Check if s1 and s2 are equal
    logger.info("=== Comparing activations between new and old IDC ===")

    # Check if they have the same keys
    s1_keys = set(s1.keys())
    s2_keys = set(s2.keys())

    keys_equal = s1_keys == s2_keys
    logger.info(f"Keys equal: {keys_equal}")
    if not keys_equal:
        logger.info(f"s1 keys: {s1_keys}")
        logger.info(f"s2 keys: {s2_keys}")
        logger.info(f"Keys only in s1: {s1_keys - s2_keys}")
        logger.info(f"Keys only in s2: {s2_keys - s1_keys}")
    
    # Check tensor equality for common keys
    all_tensors_equal = True
    common_keys = s1_keys & s2_keys
    
    for key in common_keys:
        tensors_equal = torch.equal(s1[key], s2[key])
        logger.info(f"Layer '{key}': tensors equal = {tensors_equal}")
        
        if not tensors_equal:
            all_tensors_equal = False
            # Additional details about the differences
            s1_shape = s1[key].shape
            s2_shape = s2[key].shape
            logger.info(f"  s1[{key}] shape: {s1_shape}")
            logger.info(f"  s2[{key}] shape: {s2_shape}")
            
            if s1_shape == s2_shape:
                # Calculate some difference metrics
                diff = (s1[key] - s2[key]).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                logger.info(f"  Max absolute difference: {max_diff}")
                logger.info(f"  Mean absolute difference: {mean_diff}")
                
                # Check if differences are just due to floating point precision
                close_equal = torch.allclose(s1[key], s2[key], rtol=1e-5, atol=1e-8)
                logger.info(f"  Close (within tolerance): {close_equal}")
    
    logger.info(f"Overall: All activations equal = {keys_equal and all_tensors_equal}")

def sanity_check_idc(args, model, train_loader, test_loader, logger):
    model.eval()
    
    logger.info("=== SANITY-IDC Original ===")
    df = pd.read_csv(args.csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(args.top_m_neurons)
    top_k_neurons = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        top_k_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)
    if args.use_silhouette:
        cluster_info = "silhouette"
    else:
        cluster_info = str(args.n_clusters)
    extra = dict(
        n_clusters = args.n_clusters,    # same as IDC’s n_clusters, but OK to repeat
        random_state = 42,   # fixes RNG
        n_init = 10    # keep best of 10 centroid seeds
    )

    cache_path = "./cluster_pkl/" + args.model + "_" + args.dataset + "_top_" + str(args.top_m_neurons) + "_cluster_" + cluster_info + "_wisdom_clusters.pkl"
    
    idc_new = IDC(model, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class, "KMeans", extra, cache_path)
    # idc_old = src.idc_old.IDC(model, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class, "KMeans", cache_path)

    _, s1 = idc_new.get_activations_model_dataloader(train_loader, top_k_neurons)
    # _, s2 = idc_old.get_activations_model_dataloader(train_loader, top_k_neurons)

    s1 = {k: v.half().cpu() for k, v in s1.items()}
    # s2 = {k: v.half().cpu() for k, v in s2.items()}
    
    # keys_diff(s1, s2, logger)

    # Commented out the cache_path so far for testing purposes
    cluster_groups_1 = idc_new.cluster_activation_values_all(s1)
    # cluster_groups_2 = idc_old.cluster_activation_values_all(s2)
    coverage_rate_1, t1, max_coverage_1 = idc_new.compute_idc_test_whole_dataloader(test_loader, top_k_neurons, cluster_groups_1)
    # coverage_rate_2, t2, max_coverage_2 = idc_old.compute_idc_test_whole_dataloader(test_loader, top_k_neurons, cluster_groups_2)

    logger.info(f"Total Combination: {t1}, Max Coverage: {max_coverage_1:.4f}, IDC Coverage: {coverage_rate_1:.4f}, Attribution: WISDOM_New")
    # logger.info(f"Total Combination: {t2}, Max Coverage: {max_coverage_2:.4f}, IDC Coverage: {coverage_rate_2:.4f}, Attribution: WISDOM_Old")


def sanity_check(args, model, train_loader, test_loader, U_IO_loader, U_RO_loader, test_dataset, U_I_dataset, trainable_module_name, device, logger):
    model.eval()
    
    logger.info("=== SANITY‑CHECK COVERAGE ===")
    origin_res = _run_idc_for_loader("U_O", test_loader, args, model,
                        train_loader, trainable_module_name,
                        device, logger)
    logger.info(origin_res)
    
    dup_res = duplicated_testset_coverage(test_dataset,
                                        args, model,
                                        train_loader, trainable_module_name,
                                        device, logger)

    logger.info(dup_res)
    
    random_res = _run_idc_for_loader(f"{args.attr}_U_O_UR", U_RO_loader, args, model,
                        train_loader, trainable_module_name,
                        device, logger)
    logger.info(random_res)
    
    new_res = _run_idc_for_loader(f"{args.attr}_U_O_UI", U_IO_loader, args, model,
                        train_loader, trainable_module_name,
                        device, logger)
    logger.info(new_res)
    
    
    partial_res = partial_importance_coverage(test_dataset, U_I_dataset,
                                            args, model,
                                            train_loader, trainable_module_name,
                                            device, logger,
                                            sizes=(100, 500, 1000, 2000))
    logger.info(partial_res)
    
    # Top-k and std variations
    high_param_importance_coverage(test_dataset,
                                    args, model, test_loader,
                                    train_loader, trainable_module_name,
                                    device, logger,
                                    new_topk=[0.05, 0.1, 0.2, 0.25, 0.3, 0.5],
                                    new_std=[0.1, 0.2, 0.3, 0.4, 0.5])

def get_generated_datasets(args, model, test_loader, test_dataset, device):
    if args.attr == 'wisdom':
        U_I_dataset, U_R_dataset = build_sets_wisdom(model, test_loader, device, args.csv_file, TOPK, 0.5, args.dataset)
        U_IO_dataset = ConcatDataset([test_dataset, U_I_dataset])   # original + important
        U_RO_dataset = ConcatDataset([test_dataset, U_R_dataset])   # original + random
    else:
        U_I_dataset, U_R_dataset = build_sets(model, test_loader, device, TOPK, 0.5, args.dataset)
        U_IO_dataset = ConcatDataset([test_dataset, U_I_dataset])   # original + important
        U_RO_dataset = ConcatDataset([test_dataset, U_R_dataset])   # original + random
    
    U_I_loader = DataLoader(U_I_dataset, batch_size=args.batch_size, shuffle=False)
    U_R_loader = DataLoader(U_R_dataset, batch_size=args.batch_size, shuffle=False)
    U_IO_loader = DataLoader(U_IO_dataset, batch_size=args.batch_size, shuffle=False)
    U_RO_loader = DataLoader(U_RO_dataset, batch_size=args.batch_size, shuffle=False)
    
    return U_I_loader, U_R_loader, U_IO_loader, U_RO_loader, U_I_dataset, U_R_dataset, U_IO_dataset, U_RO_dataset

# -----------------------------------------------------------
# Main entry point
# -----------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    
    # Model settings
    model, module_name, module, trainable_module, trainable_module_name, logger = prapare_data_models(args)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params}")

    # Data settings
    train_loader, test_loader, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path)
    
    # --- Get U_I, U_R, U_IO, URO datasets -------------------
    U_I_loader, U_R_loader, U_IO_loader, U_RO_loader, U_I_dataset, U_R_dataset, U_IO_dataset, U_RO_dataset = get_generated_datasets(args, model, test_loader, test_dataset, device)
    logger.info(f"[Sanity] Generated datasets: U_I: {len(U_I_dataset)}, U_R: {len(U_R_dataset)}, U_IO: {len(U_IO_dataset)}, U_RO: {len(U_RO_dataset)}")

    # --- A simple acc test for the perturbed datasets -------------------
    # eval_model(model, test_loader, U_IO_loader, U_RO_loader, device, logger)
    
    # ---  Visualization Checks ------------------------------------------
    # viz_attr_check(args, model, test_loader)
    # viz_attr_diff(args, logger, test_loader, U_I_loader, cmap=cmap[0], alpha=0.8, tag='importance')
    # viz_attr_diff(args, logger, test_loader, U_R_loader, cmap=cmap[0], alpha=0.8, tag='random')
    # viz_attr_diff(args, logger, U_I_loader, U_R_loader, cmap=cmap[1], alpha=0.8, tag='mix')
    
    # ---  Sanity checks -------------------------------------------------
    # sanity_check(args, model, train_loader, test_loader, U_IO_loader, U_RO_loader, test_dataset, U_I_dataset, trainable_module_name, device, logger)

    # ---  IDC new and old test ------------------------------------------
    sanity_check_idc(args, model, train_loader, test_loader, logger)

# python ./unittest/sanity_check.py --model resnet18 --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_IMAGENET_patched_whole.pth' --dataset imagenet --data-path /data/shenghao/dataset --batch-size 64 --device 'cuda:0' --csv-file './saved_files/pre_csv/resnet18_imagenet.csv' --attr lrp --top-m-neurons 10
# python ./unittest/sanity_check.py --model vgg16 --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --data-path /data/shenghao/dataset --batch-size 32 --device 'cuda:0' --csv-file './saved_files/pre_csv/vgg16_cifar10.csv' --attr lrp --top-m-neurons 10
# python ./unittest/sanity_check.py --model resnet18 --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_CIFAR10_whole.pth' --dataset cifar10 --data-path /data/shenghao/dataset --batch-size 32 --device 'cuda:0' --csv-file './saved_files/pre_csv/resnet18_cifar10.csv' --attr lrp --top-m-neurons 10
# python ./unittest/sanity_check.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path /data/shenghao/dataset --batch-size 64 --device 'cuda:0' --csv-file './saved_files/pre_csv/lenet_cifar10.csv' --attr lrp --top-m-neurons 10
if __name__ == '__main__':
    set_seed()
    args = parse_args()
    main(args)
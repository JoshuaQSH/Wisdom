import random
import time
import os
import math
import pandas as pd
from responses import logger
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import nn

from collections import defaultdict
from captum.attr import LRP
from captum.attr import LayerLRP


from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from src.attribution import get_relevance_scores_dataloader
from src.utils import get_data, parse_args, get_model, eval_model_dataloder, get_trainable_modules_main, _configure_logging, viz_attr


from src.idc import IDC
from src.wisdom import WisdomIDC
from src.deeepidc import DeepIDC

import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IS_IMAGENET = True
gaussian_STD = 0.5
TOPK = 0.02  # Top-k fraction of pixels to perturb (2%)
cmap = ['PuBuGn', 'Greens', 'Purples', 'Reds', 'Blues', 'YlGn', 'summer', 'cool', 'bwr']
start_ms = int(time.time() * 1000)
TIMESTAMP = time.strftime("%Y%m%d‑%H%M%S", time.localtime(start_ms / 1000))
acts = defaultdict(list)

cluster_name_all = ["KMeans", "MiniBatchKMeans", "BisectingKMeans", 
                    "AgglomerativeClustering", "SpectralClustering", "DBSCAN", 
                    "OPTICS", "HDBSCAN", "MeanShift", "AffinityPropagation", "Birch"]
clustering_params_all = {
    "KMeans": {"n_clusters": 2, "random_state": 42, "n_init": 10},
    "MiniBatchKMeans": {"n_clusters": 2, "batch_size": 32, "max_iter": 100, "random_state": 42},
    "BisectingKMeans": {"n_clusters": 2, "random_state": 42, "n_init": 10},
    "AgglomerativeClustering": {"n_clusters": 2, "linkage": "ward", "metric": "euclidean"},
    "SpectralClustering": {"n_clusters": 2, "affinity": "rbf", "assign_labels": "kmeans"},
    "DBSCAN": {"eps": 0.1, "min_samples": 10, "metric": "euclidean"},
    "OPTICS": {"min_samples": 2, "xi": 0.05, "min_cluster_size": 2},
    "HDBSCAN": {"min_cluster_size": 2, "min_samples": 2, "cluster_selection_epsilon": 0.01, "cluster_selection_method": "eom"},
    "MeanShift": {"bandwidth": 0.5, "bin_seeding": True, "cluster_all": False, "max_iter": 300, "min_bin_freq": 1},
    "AffinityPropagation": {"damping": 0.9, "preference": -50},
    "Birch": {"threshold": 0.5, "n_clusters": 2},
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


def add_gaussian_noise(imgs: torch.Tensor, mask: torch.Tensor,
                       mean: float = 0.0, std: float = 0.01):
    if std <= 0:
        out = imgs
    else:
        noise = torch.randn_like(imgs) * std + mean
        if mask.dtype is torch.bool:
            mask_f = mask.to(imgs.dtype)
        else:
            mask_f = mask
        # arithmetic selection (no boolean requirement)
        out = imgs * (1 - mask_f) + (imgs + noise) * mask_f
    return out.clamp(imgs.min(), imgs.max())


def denormalize(img: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    img: (C,H,W) normalized
    returns (H,W,C) numpy in [0,1]
    """
    if img.dim() != 3:
        raise ValueError("img must be (C,H,W)")
    c, h, w = img.shape
    m = torch.tensor(mean, dtype=img.dtype, device=img.device).view(c,1,1)
    s = torch.tensor(std,  dtype=img.dtype, device=img.device).view(c,1,1)
    x = img * s + m
    x = x.clamp(0, 1)
    return x.permute(1,2,0).cpu().numpy()

# -----------------------------------------------------------
# Model evaluations
# -----------------------------------------------------------
def eval_model(model, test_loader, U_IO_loader, U_RO_loader, device, logger):
    model.eval()
    original_accuracy, original_avg_loss, original_f1 = eval_model_dataloder(model, test_loader, device)
    accuracy_I, avg_loss_I, f1_I = eval_model_dataloder(model, U_IO_loader, device)
    accuracy_R, avg_loss_R, f1_R = eval_model_dataloder(model, U_RO_loader, device)
    
    logger.info(f"Original Accuracy: {original_accuracy:.4f}, Average Loss: {original_avg_loss:.4f}, F1 Score: {original_f1:.4f}")
    logger.info(f"Accuracy on U_IO: {accuracy_I:.4f}, Average Loss on U_IO: {avg_loss_I:.4f}, F1 Score on U_IO: {f1_I:.4f}")
    logger.info(f"Accuracy on U_RO: {accuracy_R:.4f}, Average Loss on U_RO: {avg_loss_R:.4f}, F1 Score on U_RO: {f1_R:.4f}")


# -----------------------------------------------------------
# Visualization
# -----------------------------------------------------------
def viz_attr_diff(args, logger, orig_loader, pert_loader, cmap="bwr", alpha=0.8, tag='random'):
    
    idx = random.randrange(len(orig_loader.dataset))
    img_orig, y = orig_loader.dataset[10]
    img_pert, _ = pert_loader.dataset[10]
    
    # tensors -> HWC numpy
    if IS_IMAGENET:
        o = denormalize(img_orig)
        p = denormalize(img_pert)
    else:
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
# Data wrapper and generator - old version
# -----------------------------------------------------------
class LabelToIntDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dataset):
        # e.g., TensorDataset(image_tensor, label_tensor)
        self.dataset = tensor_dataset  
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # convert label from Tensor to int
        return x, y.item()  
    
    def __len__(self):
        return len(self.dataset)

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

def get_generated_datasets(args, model, test_loader, test_dataset, device, logger):
    if args.attr == 'wisdom':
        U_I_dataset, U_R_dataset = build_sets_wisdom(model, test_loader, device, args.csv_file, TOPK, gaussian_STD, args.dataset)
        U_IO_dataset = ConcatDataset([test_dataset, U_I_dataset])   # original + important
        U_RO_dataset = ConcatDataset([test_dataset, U_R_dataset])   # original + random
    else:
        U_I_dataset, U_R_dataset = build_sets(model, test_loader, device, TOPK, gaussian_STD, args.dataset)
        U_IO_dataset = ConcatDataset([test_dataset, U_I_dataset])   # original + important
        U_RO_dataset = ConcatDataset([test_dataset, U_R_dataset])   # original + random

    U_IO_loader = DataLoader(U_IO_dataset, batch_size=args.batch_size, shuffle=False)
    U_RO_loader = DataLoader(U_RO_dataset, batch_size=args.batch_size, shuffle=False)

    U_I_loader = DataLoader(U_I_dataset, batch_size=args.batch_size, shuffle=False)
    U_R_loader = DataLoader(U_R_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"[Sanity] Generated datasets: U_I: {len(U_I_dataset)}, U_R: {len(U_R_dataset)}, U_IO: {len(U_IO_dataset)}, U_RO: {len(U_RO_dataset)}")

    del U_I_dataset, U_R_dataset, U_IO_dataset, U_RO_dataset

    return U_I_loader, U_IO_loader, U_RO_loader, U_R_loader

# -----------------------------------------------------------
# Dataset Wrapper - new version
# -----------------------------------------------------------

def build_masks_exclusive_channelwise(saliency: torch.Tensor,
                                      k: float,
                                      gen: torch.Generator | None = None,
                                      device: torch.device | None = None):
    if not (0 < k <= 1):
        raise ValueError("k must be in (0, 1].")

    if saliency.dim() == 4 and saliency.size(0) == 1:
        sal = saliency[0]             # (C,H,W)
    elif saliency.dim() == 3:
        sal = saliency                # (C,H,W)
    else:
        raise ValueError("saliency must be (C,H,W) or (1,C,H,W).")

    device = device or sal.device
    C, H, W = sal.shape
    n = C * H * W
    kpix = max(1, int(round(k * n)))

    flat = sal.abs().reshape(-1)      # (C*H*W,)
    if kpix >= n:
        imp_flat = torch.ones(n, dtype=torch.bool, device=device)
    else:
        # Use topk indices (on CPU for safety), then scatter into a bool mask
        work = flat if flat.device.type == "cpu" else flat.cpu()
        idx = torch.topk(work, kpix, largest=True).indices
        imp_flat = torch.zeros(n, dtype=torch.bool, device='cpu')
        imp_flat[idx] = True
        imp_flat = imp_flat.to(device)

    # sample random from the complement (non-overlap)
    comp_idx = torch.nonzero(~imp_flat, as_tuple=False).view(-1).cpu()
    if comp_idx.numel() == 0:
        rand_flat = imp_flat.clone()
    else:
        # generator must be CPU for torch.randperm
        g = gen if gen is not None else torch.Generator(device='cpu')
        if g.device.type != 'cpu':
            g = torch.Generator(device='cpu').manual_seed(42)
        perm = torch.randperm(comp_idx.numel(), generator=g, device='cpu')
        sel = comp_idx[perm[:kpix]]
        rand_flat = torch.zeros(n, dtype=torch.bool, device='cpu')
        rand_flat[sel] = True
        rand_flat = rand_flat.to(device)

    imp = imp_flat.view(C, H, W)
    rand = rand_flat.view(C, H, W)
    return imp, rand

def make_lrp_strategy_channelwise(model, target_layer=None):
    """
    (1,C,H,W) -> (C,H,W) channel-wise saliency
    """
    model.eval().to(next(model.parameters()).device)
    global_lrp = LRP(model)
    layer_lrp = None

    def _is_spatial_layer(layer):
        return hasattr(layer, "weight") and getattr(layer, "weight", None) is not None and hasattr(layer, "stride")

    if target_layer is not None and _is_spatial_layer(target_layer):
        layer_lrp = LayerLRP(model, target_layer)

    def strategy(x: torch.Tensor) -> torch.Tensor:
        device = next(model.parameters()).device
        x = x.to(device)
        x.requires_grad_(True)
        with torch.no_grad():
            pred = model(x).argmax(dim=1)

        if layer_lrp is not None:
            attr = layer_lrp.attribute(x, target=pred)     # (1,C,h,w)
            H, W = x.shape[-2:]
            if attr.shape[-2:] != (H, W):
                attr = F.interpolate(attr, size=(H, W), mode="bilinear", align_corners=False)
            sal = attr[0].abs().detach().cpu()             # (C,H,W)
            return sal

        attr = global_lrp.attribute(x, target=pred)        # (1,C,H,W)
        sal = attr[0].abs().detach().cpu()                 # (C,H,W)
        return sal

    return strategy

def make_lrp_strategy_channelwise_buildsets_parity(model):
    """
    Parity with build_sets(): whole-model LRP, channel-wise saliency (C,H,W).
    """
    model.eval().to(next(model.parameters()).device)
    lrp = LRP(model)

    def strategy(x: torch.Tensor) -> torch.Tensor:
        device = next(model.parameters()).device
        x = x.to(device)
        x.requires_grad_(True)
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        attr = lrp.attribute(x, target=pred)   # (1,C,H,W)
        return attr[0].abs().detach().cpu()    # (C,H,W)

    return strategy

def build_masks_exclusive(saliency: torch.Tensor, 
                          k: float,
                          gen = None,
                          device = 'cuda'):
    if not (0 < k <= 1):
        raise ValueError("k must be in (0, 1].")

    if saliency.dim() == 3 and saliency.size(0) == 1:
        sal = saliency[0]
    elif saliency.dim() == 2:
        sal = saliency
    else:
        raise ValueError("saliency must be (H,W) or (1,H,W).")

    device = device or sal.device
    H, W = sal.shape
    n = H * W
    kpix = max(1, int(round(k * n)))

    flat = sal.abs().reshape(-1)
    if kpix >= n:
        imp_flat = torch.ones(n, dtype=torch.float32, device=device)
    else:
        work = flat if flat.device.type == "cpu" else flat.cpu()
        thresh = torch.topk(work, kpix, largest=True).values.min()
        mask_cpu = (work >= thresh).float()
        imp_flat = mask_cpu.to(device)

    # Random from complement
    comp_idx = torch.nonzero(imp_flat == 0, as_tuple=False).view(-1)
    if comp_idx.numel() == 0:
        rand_flat = imp_flat.clone()
    else:
        cpu_gen = gen if (gen is not None and gen.device == torch.device('cpu')) else gen
        perm = torch.randperm(comp_idx.numel(), generator=cpu_gen, device='cpu')
        sel = comp_idx.cpu()[perm[:kpix]]
        rand_flat = torch.zeros(n, dtype=torch.float32, device='cpu')
        rand_flat[sel] = 1.0
        rand_flat = rand_flat.to(device)

    imp = imp_flat.view(1, H, W)
    rand = rand_flat.view(1, H, W)
    return imp, rand


class PerturbedDataset(Dataset):
    def __init__(self,
                 base_dataset: Dataset,
                 strategy,
                 k: float,
                 std: float,
                 mode: str = "important",
                 seed: int = 42):
        if mode not in ("important", "random"):
            raise ValueError("mode must be 'important' or 'random'")
        if not (0 < k <= 1):
            raise ValueError("k must be in (0, 1].")

        self.base_dataset = base_dataset
        self.strategy = strategy
        self.k = float(k)
        self.std = float(std)
        self.mode = mode

        # Deterministic generator for random masks
        self.gen = torch.Generator(device='cpu')
        self.gen.manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, label = self.base_dataset[idx]  # expects img as (C,H,W) tensor
        if not torch.is_tensor(img):
            raise TypeError("Base dataset must return tensors for images (C,H,W).")

        # 1. Compute saliency (CPU or GPU inside strategy)
        x = img.unsqueeze(0)  # (1,C,H,W)
        sal = self.strategy(x)  # (H,W), typically on CPU

        # 2. Build non-overlapping masks using the same saliency
        imp_mask, rand_mask = build_masks_exclusive(
            saliency=sal,
            k=self.k,
            gen=self.gen,
            device=img.device
        )

        # 2. Build non-overlapping masks using the same saliency. [channel-wise]
        # imp_mask, rand_mask = build_masks_exclusive_channelwise(
        #     saliency=sal,
        #     k=self.k,
        #     gen=self.gen,
        #     device=img.device
        # )

        # 3. Choose mask and perturb
        mask = imp_mask if self.mode == "important" else rand_mask
        x_pert = add_gaussian_noise(img, mask, std=self.std)

        return x_pert, int(label)
    
def make_wisdom_strategy(model, csv_path: str):
    model.eval().to(next(model.parameters()).device)
    df = pd.read_csv(csv_path)
    df = df[df.Score != 0]
    layer2score = defaultdict(dict)
    for _, row in df.iterrows():
        layer2score[row.LayerName][int(row.NeuronIndex)] = float(row.Score)

    named = dict(model.named_modules())

    def strategy(x: torch.Tensor) -> torch.Tensor:
        device = next(model.parameters()).device
        x = x.to(device)
        H, W = x.shape[-2:]
        activations = {}

        handles = []
        try:
            for name, score_dict in layer2score.items():
                if name not in named:
                    continue
                mod = named[name]
                def _make_hook(lname=name, sd=score_dict):
                    def _hook(_, __, out):
                        acts = out.detach()
                        # (B,C,H,W) or (B,C)
                        if acts.dim() == 2:
                            acts = acts.unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
                        # weight channels
                        C = acts.size(1)
                        score_vec = torch.zeros(C, device=acts.device)
                        for idx, s in sd.items():
                            if 0 <= idx < C: score_vec[idx] = s
                        weighted = acts * score_vec.view(1, -1, 1, 1)  # (B,C,h,w)
                        activations[lname] = weighted
                    return _hook
                handles.append(mod.register_forward_hook(_make_hook()))

            with torch.no_grad():
                _ = model(x)

            heat = torch.zeros(1, 1, H, W, device=device)
            for lname, acts in activations.items():
                if acts.size(2) != H or acts.size(3) != W:
                    acts = F.interpolate(acts, size=(H, W), mode="bilinear", align_corners=False)
                layer_heat = acts.sum(dim=1, keepdim=True)  # (1,1,H,W)
                heat += layer_heat
            return heat[0, 0].detach().cpu()  # (H,W)

        finally:
            for h in handles:
                h.remove()

    return strategy

def make_lrp_strategy(model, target_layer=None):
    """
    Returns a callable: (1,C,H,W) -> (H,W) saliency map on CPU.
    - If target_layer is a conv: use LayerLRP at that layer, upsample to input size.
    - Otherwise: use whole-model LRP and reduce channels.
    """
    model.eval().to(next(model.parameters()).device)
    global_lrp = LRP(model)
    layer_lrp = None

    # helper to check if a layer is spatial (conv-like)
    def _is_spatial_layer(layer):
        return hasattr(layer, "weight") and getattr(layer, "weight", None) is not None and hasattr(layer, "stride")

    if target_layer is not None and _is_spatial_layer(target_layer):
        layer_lrp = LayerLRP(model, target_layer)

    def strategy(x: torch.Tensor) -> torch.Tensor:
        device = next(model.parameters()).device
        x = x.to(device)
        x.requires_grad_(True)  # silence the Captum warning & be explicit

        with torch.no_grad():
            pred = model(x).argmax(dim=1)

        if layer_lrp is not None:
            # LayerLRP at conv layer → (1,C,h,w)
            attr = layer_lrp.attribute(x, target=pred)
            sal = attr.abs().mean(dim=1, keepdim=True)   # (1,1,h,w)
            # upsample to input size
            H, W = x.shape[-2:]
            sal = F.interpolate(sal, size=(H, W), mode="bilinear", align_corners=False)[0, 0].detach().cpu()
            return sal  # (H,W)

        # Fallback: whole-model LRP → (1,C,H,W)
        attr = global_lrp.attribute(x, target=pred)
        sal = attr.abs().mean(dim=1, keepdim=True)[0, 0].detach().cpu()  # (H,W)
        return sal

    return strategy

def get_generated_dataset_optimized(args, model, test_dataset, target_layer, logger):

    # After loading train_loader, test_loader, train_dataset, test_dataset, classes:
    # Wrap the test_dataset in lazy perturbed datasets
    
    # strategy = make_lrp_strategy(model, target_layer=target_layer)
    # strategy = make_lrp_strategy_channelwise(model, target_layer=target_layer)
    strategy = make_wisdom_strategy(model, args.csv_file)
    # strategy = make_lrp_strategy_channelwise_buildsets_parity(model)
    
    
    U_I_dataset = PerturbedDataset(base_dataset=test_dataset, 
                                   strategy=strategy,
                                   k=TOPK, 
                                   std=gaussian_STD,
                                   mode='important',
                                   seed=42)
    U_R_dataset = PerturbedDataset(base_dataset=test_dataset, 
                                   strategy=strategy,
                                   k=TOPK, 
                                   std=gaussian_STD,
                                   mode='random',
                                   seed=42)

    # Build the concatenated datasets without pre‑allocating all images
    U_IO_dataset = ConcatDataset([test_dataset, U_I_dataset])  # original + importance‑perturbed
    U_RO_dataset = ConcatDataset([test_dataset, U_R_dataset])  # original + random‑perturbed

    U_I_loader = DataLoader(U_I_dataset, batch_size=args.batch_size, shuffle=False)
    U_R_loader = DataLoader(U_R_dataset, batch_size=args.batch_size, shuffle=False)

    # Create dataloaders (no additional memory overhead beyond batch size)
    U_IO_loader = DataLoader(U_IO_dataset, batch_size=args.batch_size, shuffle=False)
    U_RO_loader = DataLoader(U_RO_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"[Sanity] Generated datasets: U_I: {len(U_I_dataset)}, U_R: {len(U_R_dataset)}, U_IO: {len(U_IO_dataset)}, U_RO: {len(U_RO_dataset)}")
    
    del U_I_dataset, U_R_dataset, U_IO_dataset, U_RO_dataset
    
    return U_I_loader, U_IO_loader, U_RO_loader, U_R_loader


# -----------------------------------------------------------
# IDC coverage testing
# -----------------------------------------------------------
def idc_coverage(args, model, train_loader, test_loader, trainable_module_name, device, logger, cluster_method_name, tag='original'):
    
    if args.use_silhouette:
        cluster_info = f"_{cluster_method_name}_silhouette_"
    else:
        cluster_info = f"_{cluster_method_name}_"

    cache_path = "./cluster_pkl/" + args.model + "_" + args.dataset + "_top_" + str(args.top_m_neurons) + cluster_info + "deepimportance_clusters.pkl"
    extra = clustering_params_all[cluster_method_name]
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
        cluster_method_name,
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
    
    # results = {}
    # results['IDC'] = coverage_rate
    # df = pd.DataFrame(results, index=[tag])
    # save_csv_results(results, "rq2_sanity_{}_{}_{}.csv".format(args.dataset, args.model, TIMESTAMP), tag=tag)
    logger.info(f"Total Combination: {total_combination}, Max Coverage: {max_coverage:.4f}, IDC Coverage: {coverage_rate:.4f}, Attribution: {args.attr}")
    return coverage_rate



# -----------------------------------------------------------
# WISDOM IDC Sanity Check
# -----------------------------------------------------------
def wisdom_fit_once(args, model, train_loader, top_k_neurons, cluster_method_name, device):
    extra = clustering_params_all[cluster_method_name]
    cache_path = (
        "./cluster_pkl/"
        f"{args.model}_{args.dataset}_top_{args.top_m_neurons}_"
        f"{cluster_method_name}{'_sil' if args.use_silhouette else ''}_wisdom.pkl"
    )
    # Force determinism if not present
    if "random_state" not in extra:
        extra = {**extra, "random_state": 42}
    if cluster_method_name == "KMeans" and "n_init" not in extra:
        extra = {**extra, "n_init": 10}

    wisdom_idc = WisdomIDC(
        model=model,
        top_m_neurons=args.top_m_neurons,
        n_clusters=extra.get("n_clusters", args.n_clusters),
        use_silhouette=args.use_silhouette,
        test_all_classes=args.all_class,
        clustering_method_name=cluster_method_name,
        device=device,
        clustering_params=extra,
        cache_path=cache_path,
    )

    train_acts = wisdom_idc.get_selected_activations(train_loader, top_k_neurons)
    cluster_groups = wisdom_idc.cluster_per_neuron(train_acts)
    
    return wisdom_idc, cluster_groups

def sanity_check_wisdomIDC(args, model, train_loader, test_loader, U_IO_loader, U_RO_loader, cluster_method_name, device, logger):
    logger.info("=== SANITY-CHECK WISDOM IDC COVERAGE ===")
    
    # prepare neurons once
    df = pd.read_csv(args.csv_file).sort_values(by="Score", ascending=False).head(args.top_m_neurons)
    top_k_neurons = {ln: torch.tensor(g["NeuronIndex"].values) for ln, g in df.groupby("LayerName")}

    model.eval()
    
    # fit clusters ONCE (deterministically) and reuse
    wisdom_idc, cluster_groups = wisdom_fit_once(args, model, train_loader, top_k_neurons, cluster_method_name, device)
    logger.info("WISDOM fits once DONE")
    
    # coverage on U_O
    test_acts = wisdom_idc.get_selected_activations(test_loader, top_k_neurons)
    cov_O, total, maxcov = wisdom_idc.compute_coverage(test_acts, cluster_groups)
    logger.info(f"[Sanity] U_O: cov={cov_O:.6f} total={total} max={maxcov:.6f}")

    # duplicate TEST set
    dup_dataset = ConcatDataset([test_loader.dataset, test_loader.dataset])
    dup_loader = DataLoader(dup_dataset, batch_size=args.batch_size, shuffle=False)
    dup_acts = wisdom_idc.get_selected_activations(dup_loader, top_k_neurons)
    cov_dup, total_dup, maxcov_dup = wisdom_idc.compute_coverage(dup_acts, cluster_groups)
    logger.info(f"[Sanity] U_O + U_O: cov={cov_dup:.6f} total={total_dup} max={maxcov_dup:.6f}")

    # U_RO (random)
    uro_acts = wisdom_idc.get_selected_activations(U_RO_loader, top_k_neurons)
    cov_UR, total_UR, maxcov_UR = wisdom_idc.compute_coverage(uro_acts, cluster_groups)
    logger.info(f"[Sanity] U_RO: cov={cov_UR:.6f} total={total_UR} max={maxcov_UR:.6f}")

    # U_IO (important)
    uio_acts = wisdom_idc.get_selected_activations(U_IO_loader, top_k_neurons)
    cov_UI, total_UI, maxcov_UI = wisdom_idc.compute_coverage(uio_acts, cluster_groups)
    logger.info(f"[Sanity] U_IO: cov={cov_UI:.6f} total={total_UI} max={maxcov_UI:.6f}")

    return dict(U_O=cov_O, U_OO=cov_dup, U_RO=cov_UR, U_IO=cov_UI)

# -----------------------------------------------------------
# DP IDC Sanity Check
# -----------------------------------------------------------
def dpidc_fit_once(args, model, train_loader, layer_scores, cluster_method_name, device):
    """
    layer_scores: Dict[str, Tensor] from get_relevance_scores_dataloader
    returns: (idc, selected_indices, cluster_groups)
    """
    extra = clustering_params_all[cluster_method_name]
    cache_path = (
        "./cluster_pkl/"
        f"{args.model}_{args.dataset}_top_{args.top_m_neurons}_"
        f"{cluster_method_name}{'_sil' if args.use_silhouette else ''}_deepimportance.pkl"
    )
    if "random_state" not in extra:
        extra = {**extra, "random_state": 42}
    if cluster_method_name == "KMeans" and "n_init" not in extra:
        extra = {**extra, "n_init": 10}

    idc = DeepIDC(
        model=model,
        top_m_neurons=args.top_m_neurons,
        n_clusters=extra.get("n_clusters", args.n_clusters),
        use_silhouette=args.use_silhouette,
        test_all_classes=args.all_class,
        clustering_method_name=cluster_method_name,
        device=device,
        clustering_params=extra,
        cache_path=cache_path,
    )

    # 1) pick top-m **indices** per layer (not scores)
    selected_indices = idc.select_top_neurons(layer_scores)  # Dict[layer]->LongTensor

    # 2) fit clusters on train activations of the selected neurons
    train_acts = idc.get_selected_activations(train_loader, selected_indices)
    cluster_groups = idc.cluster_per_neuron(train_acts)

    return idc, selected_indices, cluster_groups

def sanity_check_deepimportanceIDC(args, model, train_loader, test_loader,
                                   U_IO_loader, U_RO_loader,
                                   cluster_method_name, device, logger):
    logger.info("=== SANITY-CHECK DEEP IMPORTANCE IDC COVERAGE ===")

    # 0) get layer-wise scores (not indices)
    layer_scores = get_relevance_scores_dataloader(
        model, train_loader, device, attribution_method='lrp'
    )

    # 1) fit once and keep selected indices
    idc, selected_indices, cluster_groups = dpidc_fit_once(
        args, model, train_loader, layer_scores, cluster_method_name, device
    )
    logger.info("DeepImportance IDC fits once DONE")

    # 2) coverage on U_O
    test_acts = idc.get_selected_activations(test_loader, selected_indices)
    cov_O, total, maxcov = idc.compute_coverage(test_acts, cluster_groups)
    logger.info(f"[Sanity] U_O: cov={cov_O:.6f} total={total} max={maxcov:.6f}")

    # 3) duplicated test
    dup_dataset = ConcatDataset([test_loader.dataset, test_loader.dataset])
    dup_loader = DataLoader(dup_dataset, batch_size=args.batch_size, shuffle=False)
    dup_acts = idc.get_selected_activations(dup_loader, selected_indices)
    cov_dup, total_dup, maxcov_dup = idc.compute_coverage(dup_acts, cluster_groups)
    logger.info(f"[Sanity] U_O + U_O: cov={cov_dup:.6f} total={total_dup} max={maxcov_dup:.6f}")

    # 4) U_RO
    uro_acts = idc.get_selected_activations(U_RO_loader, selected_indices)
    cov_UR, total_UR, maxcov_UR = idc.compute_coverage(uro_acts, cluster_groups)
    logger.info(f"[Sanity] U_RO: cov={cov_UR:.6f} total={total_UR} max={maxcov_UR:.6f}")

    # 5) U_IO
    uio_acts = idc.get_selected_activations(U_IO_loader, selected_indices)
    cov_UI, total_UI, maxcov_UI = idc.compute_coverage(uio_acts, cluster_groups)
    logger.info(f"[Sanity] U_IO: cov={cov_UI:.6f} total={total_UI} max={maxcov_UI:.6f}")

    return dict(U_O=cov_O, U_OO=cov_dup, U_RO=cov_UR, U_IO=cov_UI)


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
    U_I_loader, U_IO_loader, U_RO_loader, U_R_loader = get_generated_datasets(args, model, test_loader, test_dataset, device, logger)
    # U_I_loader, U_IO_loader, U_RO_loader, U_R_loader = get_generated_dataset_optimized(args, model, test_dataset, trainable_module[-1], logger)

    cluster_method_name = cluster_name_all[0]

    # --- A simple acc test for the perturbed datasets -------------------
    # eval_model(model, test_loader, U_IO_loader, U_RO_loader, device, logger)
    
    # ---  Visualization Checks ------------------------------------------
    viz_attr_diff(args, logger, test_loader, U_I_loader, cmap=cmap[0], alpha=0.8, tag='importance')
    viz_attr_diff(args, logger, test_loader, U_R_loader, cmap=cmap[0], alpha=0.8, tag='random')
    
    # ---  Sanity checks WisdomIDC -------------------------------------------------
    # sanity_check_wisdomIDC(args, model, train_loader, test_loader, U_IO_loader, U_RO_loader, cluster_method_name, device, logger)
    
    # ---  Sanity checks DeepImportance IDC -------------------------------------------------
    # sanity_check_deepimportanceIDC(args, model, train_loader, test_loader, U_IO_loader, U_RO_loader, cluster_method_name, device, logger)

if __name__ == '__main__':
    set_seed()
    args = parse_args()
    main(args)
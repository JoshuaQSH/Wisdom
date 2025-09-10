import random
import time
import os
import math
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict
from captum.attr import LRP
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


TOPK = 0.02  # Top-k fraction of pixels to perturb (2%)
gaussian_STD = 0.5
start_ms = int(time.time() * 1000)
TIMESTAMP = time.strftime("%Y%m%d‑%H%M%S", time.localtime(start_ms / 1000))
acts = defaultdict(list)

class LabelToIntDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dataset):
        self.dataset = tensor_dataset  # e.g., TensorDataset(image_tensor, label_tensor)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y.item()  # convert label from Tensor to int
    
    def __len__(self):
        return len(self.dataset)

def wisdom_importance_scores(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df.Score != 0]
    layer2score = defaultdict(dict)
    for _, row in df.iterrows():
        layer2score[row.LayerName][int(row.NeuronIndex)] = float(row.Score)
    return layer2score

def register_hooks(model, csv_path):
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

def add_gaussian_noise_old(imgs: torch.Tensor, mask: torch.Tensor,
                       mean=0., std=0.01):
    noise = torch.randn_like(imgs) * std + mean
    return torch.where(mask, imgs + noise, imgs).clamp(torch.min(imgs), torch.max(imgs))


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

def build_mask(attributions, k=0.02, exclude_imp=True):
    """
    Return boolean masks (important and random) with the top-k fraction (e.g. 0.02 = 2 %)
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
    attributions: (B, 1, H, W) - relevance per pixel
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
        inputs_I = add_gaussian_noise_old(inputs, mask_imp, std=std)
        # Random-based pixels
        inputs_R = add_gaussian_noise_old(inputs, mask_rand, std=std) 
        
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
        inputs_I = add_gaussian_noise_old(inputs, mask_imp, std=std)
        # Random-based pixels
        inputs_R = add_gaussian_noise_old(inputs, mask_rand, std=std)
        
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

def get_adv_dataloader(model, test_loader, device='cpu', batch_size=32, csv_file=None, attr='wisdom', k=TOPK, std=gaussian_STD, name="bo_test"):
    if attr == 'wisdom':
        U_I_dataset, U_R_dataset = build_sets_wisdom(model, test_loader, device, csv_file, TOPK, gaussian_STD, name)
        U_IO_dataset = ConcatDataset([test_loader.dataset, U_I_dataset])   # original + important
        U_RO_dataset = ConcatDataset([test_loader.dataset, U_R_dataset])   # original + random
    else:
        U_I_dataset, U_R_dataset = build_sets(model, test_loader, device, TOPK, gaussian_STD, name)
        U_IO_dataset = ConcatDataset([test_loader.dataset, U_I_dataset])   # original + important
        U_RO_dataset = ConcatDataset([test_loader.dataset, U_R_dataset])   # original + random
    
    U_IO_loader = DataLoader(U_IO_dataset, batch_size=batch_size, shuffle=False)
    U_RO_loader = DataLoader(U_RO_dataset, batch_size=batch_size, shuffle=False)

    print(f"[Dataset] Generated datasets: U_I: {len(U_I_dataset)}, U_R: {len(U_R_dataset)}, U_IO: {len(U_IO_dataset)}, U_RO: {len(U_RO_dataset)}")

    return U_IO_loader, U_RO_loader


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

def get_generated_dataset_optimized(args, model, test_dataset, logger):

    strategy = make_wisdom_strategy(model, args.csv_file)
    
    
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

    # Create dataloaders (no additional memory overhead beyond batch size)
    U_IO_loader = DataLoader(U_IO_dataset, batch_size=args.batch_size, shuffle=False)
    U_RO_loader = DataLoader(U_RO_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"[Sanity] Generated datasets: U_I: {len(U_I_dataset)}, U_R: {len(U_R_dataset)}, U_IO: {len(U_IO_dataset)}, U_RO: {len(U_RO_dataset)}")
    
    del U_I_dataset, U_R_dataset, U_IO_dataset, U_RO_dataset
    
    return U_IO_loader, U_RO_loader
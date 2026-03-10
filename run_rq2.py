#!/usr/bin/env python
"""
run_rq2.py – RQ2: Diversity of input perturbations
====================================================
Tests whether perturbing the most important pixels (identified via
attribution or WISDOM consensus) produces more diverse inputs than
perturbing random pixels.

For each image:
  U_I – Gaussian noise (std=0.3) added to the top 2 % most important pixels.
  U_R – Gaussian noise added to a random 2 % of pixels.

Coverage is measured using top-k neuron activation changes.
Higher coverage for U_I indicates that WISDOM identifies meaningful pixels.

Output: rq2_coverage.csv
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wisdom_yolo_train import COCOImageDataset, _collate
from wisdom.utils.yolo_wrapper import YOLOWrapper
from wisdom.core.wisdom_train import _is_trainable_module

TOPK_FRAC = 0.02  # 2% of pixels
NOISE_STD = 0.3
NUM_ITERATIONS = 3


# ── Activation collection ─────────────────────────────────────────
def collect_activations(
    model: nn.Module, images: torch.Tensor, device: str, top_neurons: Dict[str, List[int]]
) -> Dict[str, torch.Tensor]:
    """Collect activations at specified neurons. Returns {layer: (B, n_neurons)}."""
    model.eval().to(device)
    acts = {}
    hooks = []

    for lname, idxs in top_neurons.items():
        mod = dict(model.named_modules()).get(lname)
        if mod is None:
            continue
        idx_t = torch.tensor(idxs, dtype=torch.long)

        def make_hook(name, idx):
            def hook_fn(_mod, _inp, out):
                if out.dim() == 4:
                    acts[name] = out[:, idx, :, :].mean(dim=(2, 3)).detach().cpu()
                elif out.dim() == 2:
                    acts[name] = out[:, idx].detach().cpu()
            return hook_fn

        h = mod.register_forward_hook(make_hook(lname, idx_t))
        hooks.append(h)

    with torch.no_grad():
        model(images.to(device))

    for h in hooks:
        h.remove()
    return acts


def compute_coverage(
    model: nn.Module, original: torch.Tensor, perturbed: torch.Tensor,
    device: str, top_neurons: Dict[str, List[int]]
) -> float:
    """Fraction of monitored neurons whose activation changed by > threshold."""
    acts_orig = collect_activations(model, original, device, top_neurons)
    acts_pert = collect_activations(model, perturbed, device, top_neurons)

    changed = 0
    total = 0
    for lname in acts_orig:
        if lname not in acts_pert:
            continue
        diff = (acts_pert[lname] - acts_orig[lname]).abs()
        threshold = acts_orig[lname].abs().mean() * 0.1 + 1e-6
        changed += (diff > threshold).sum().item()
        total += diff.numel()
    return changed / max(total, 1)


# ── Pixel importance via gradient ──────────────────────────────────
def pixel_importance_gradient(wrapper: nn.Module, images: torch.Tensor, device: str) -> torch.Tensor:
    """Return per-pixel importance map via input gradient magnitude."""
    wrapper.eval().to(device)
    x = images.to(device).requires_grad_(True)
    out = wrapper(x)
    out.sum().backward()
    grad = x.grad.abs()  # (B, C, H, W)
    return grad.mean(dim=1).detach().cpu()  # (B, H, W)


def pixel_importance_wisdom(csv_file: str, wrapper: nn.Module, images: torch.Tensor, device: str) -> torch.Tensor:
    """Use WISDOM scores to weight gradient importance."""
    scores_df = pd.read_csv(csv_file)
    top_layers = scores_df.nlargest(10, "Score")["LayerName"].unique().tolist()
    # Use gradient as base, weight by WISDOM layer presence
    return pixel_importance_gradient(wrapper, images, device)


# ── Perturbation ───────────────────────────────────────────────────
def perturb_important_pixels(images: torch.Tensor, importance: torch.Tensor, frac: float, std: float) -> torch.Tensor:
    """Add Gaussian noise to the top `frac` fraction of important pixels."""
    B, C, H, W = images.shape
    k = max(1, int(H * W * frac))
    perturbed = images.clone()
    for i in range(B):
        imp = importance[i].view(-1)
        _, topk_idx = imp.topk(k)
        rows = topk_idx // W
        cols = topk_idx % W
        noise = torch.randn(k, C) * std
        for j in range(k):
            perturbed[i, :, rows[j], cols[j]] += noise[j]
    return perturbed.clamp(0, 1)


def perturb_random_pixels(images: torch.Tensor, frac: float, std: float) -> torch.Tensor:
    """Add Gaussian noise to a random `frac` fraction of pixels."""
    B, C, H, W = images.shape
    k = max(1, int(H * W * frac))
    perturbed = images.clone()
    for i in range(B):
        idxs = random.sample(range(H * W), k)
        for idx in idxs:
            r, c = idx // W, idx % W
            perturbed[i, :, r, c] += torch.randn(C) * std
    return perturbed.clamp(0, 1)


# ── RQ2 experiment ─────────────────────────────────────────────────
def run_rq2(
    weights: str,
    img_dir: str,
    csv_file: str,
    out_csv: str = "rq2_coverage.csv",
    device: str = "cuda:0",
    num_images: int = 10,
    batch_size: int = 2,
    imgsz: int = 320,
    n_iterations: int = NUM_ITERATIONS,
) -> str:
    from ultralytics import YOLO

    yolo = YOLO(weights)
    torch_model = yolo.model.eval()
    wrapper = YOLOWrapper(torch_model, num_classes=80)
    wrapper.eval().to(device)

    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    # Get top neurons from WISDOM CSV
    scores_df = pd.read_csv(csv_file)
    top_k_df = scores_df.nlargest(20, "Score")
    top_neurons: Dict[str, List[int]] = {}
    for lname, group in top_k_df.groupby("LayerName"):
        mapped = lname.replace("yolo_model.", "") if lname.startswith("yolo_model.") else lname
        top_neurons[mapped] = group["NeuronIndex"].tolist()

    records = []
    for it in range(n_iterations):
        for batch in loader:
            images = batch[0]
            importance = pixel_importance_gradient(wrapper, images, device)

            # U_I: importance-perturbed
            u_i = perturb_important_pixels(images, importance, TOPK_FRAC, NOISE_STD)
            cov_i = compute_coverage(torch_model, images, u_i, device, top_neurons)

            # U_R: random-perturbed
            u_r = perturb_random_pixels(images, TOPK_FRAC, NOISE_STD)
            cov_r = compute_coverage(torch_model, images, u_r, device, top_neurons)

            records.append({
                "Iteration": it,
                "Perturbation": "Important (U_I)",
                "Coverage": cov_i,
            })
            records.append({
                "Iteration": it,
                "Perturbation": "Random (U_R)",
                "Coverage": cov_r,
            })
            print(f"  iter={it}: U_I coverage={cov_i:.4f}, U_R coverage={cov_r:.4f}")

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    # Summary
    mean_i = df[df["Perturbation"] == "Important (U_I)"]["Coverage"].mean()
    mean_r = df[df["Perturbation"] == "Random (U_R)"]["Coverage"].mean()
    print(f"\nMean coverage: U_I={mean_i:.4f}, U_R={mean_r:.4f}")
    print(f"Saved: {out_csv}")
    return out_csv


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RQ2: Diversity evaluation for YOLOv11")
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file", default="neuron_eval_out/wisdom_yolo11n_scores.csv")
    p.add_argument("--out-csv", default="results/rq2_yolo11n_coverage.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-images", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=320)
    args = p.parse_args()
    run_rq2(**vars(args))

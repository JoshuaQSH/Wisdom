#!/usr/bin/env python
"""
run_rq4.py – RQ4: Correlation with traditional coverage metrics
================================================================
Evaluates the correlation between WISDOM's neuron importance scores
and traditional neuron-coverage metrics (NC, top-k neuron coverage).

For test suites of different sizes:
  1. Compute Pielou's evenness (diversity of class distribution).
  2. Compute WISDOM-based coverage (fraction of top neurons activated).
  3. Compute baseline neuron coverage (fraction of all neurons above threshold).
  4. Calculate Pearson correlation between coverage and diversity.

Pielou's evenness J' = H' / ln(S), where H' is Shannon entropy and S is
the number of species (classes). See: Pielou, E. C. (1966).

Output: rq4_correlation.csv
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wisdom_yolo_train import COCOImageDataset, _collate
from wisdom.utils.yolo_wrapper import YOLOWrapper
from wisdom.core.wisdom_train import _is_trainable_module
from run_rq2 import collect_activations

SUITE_SIZES = [5, 10, 15, 20]


# ── Pielou's evenness ─────────────────────────────────────────────
def pielou_evenness(predictions: List[int]) -> float:
    """
    J' = H' / ln(S)
    H' = -Σ(p_i * ln(p_i))  (Shannon entropy)
    S  = number of unique classes observed
    """
    if not predictions:
        return 0.0
    counts = Counter(predictions)
    S = len(counts)
    if S <= 1:
        return 1.0
    N = len(predictions)
    H = -sum((c / N) * math.log(c / N) for c in counts.values())
    return H / math.log(S)


# ── Coverage metrics ───────────────────────────────────────────────
def wisdom_coverage(
    model: nn.Module, images: torch.Tensor, device: str,
    top_neurons: Dict[str, List[int]],
) -> float:
    """Fraction of WISDOM-selected neurons that are activated above threshold."""
    acts = collect_activations(model, images, device, top_neurons)
    active = 0
    total = 0
    for lname, act_t in acts.items():
        threshold = act_t.abs().mean() * 0.1 + 1e-6
        active += (act_t.abs() > threshold).sum().item()
        total += act_t.numel()
    return active / max(total, 1)


def neuron_coverage(model: nn.Module, images: torch.Tensor, device: str, threshold: float = 0.5) -> float:
    """Traditional NC: fraction of all neurons activated above threshold."""
    model.eval().to(device)
    acts_list = []
    hooks = []

    for lname, m in model.named_modules():
        if not _is_trainable_module(m):
            continue

        def make_hook(name):
            def hook_fn(_mod, _inp, out):
                if out.dim() == 4:
                    acts_list.append(out.mean(dim=(2, 3)).detach().cpu())
                elif out.dim() == 2:
                    acts_list.append(out.detach().cpu())
            return hook_fn

        h = m.register_forward_hook(make_hook(lname))
        hooks.append(h)

    with torch.no_grad():
        model(images.to(device))

    for h in hooks:
        h.remove()

    if not acts_list:
        return 0.0
    all_acts = torch.cat(acts_list, dim=1)  # (B, total_neurons)
    return (all_acts.abs() > threshold).float().mean().item()


# ── Get YOLO predictions ──────────────────────────────────────────
def get_yolo_predictions(model: nn.Module, images: torch.Tensor, device: str) -> List[int]:
    """Return list of predicted class IDs across all images and detections."""
    model.eval().to(device)
    with torch.no_grad():
        out = model(images.to(device))
        preds = out[0] if isinstance(out, (tuple, list)) else out
        cls_scores = preds[:, 4:, :]  # (B, 80, A)
        top_classes = cls_scores.argmax(dim=1)  # (B, A)
        top_conf = cls_scores.max(dim=1).values  # (B, A)
        # Filter by confidence
        mask = top_conf > 0.25
        predictions = top_classes[mask].cpu().tolist()
    return predictions


# ── RQ4 experiment ─────────────────────────────────────────────────
def run_rq4(
    weights: str,
    img_dir: str,
    csv_file: str,
    out_csv: str = "rq4_correlation.csv",
    device: str = "cuda:0",
    num_images: int = 30,
    imgsz: int = 320,
) -> str:
    from ultralytics import YOLO

    yolo = YOLO(weights)
    torch_model = yolo.model.eval()

    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    all_images = torch.stack([ds[i][0] for i in range(len(ds))])

    # Get WISDOM top neurons
    scores_df = pd.read_csv(csv_file)
    top_k_df = scores_df.nlargest(20, "Score")
    top_neurons: Dict[str, List[int]] = {}
    for lname, group in top_k_df.groupby("LayerName"):
        mapped = lname.replace("yolo_model.", "") if lname.startswith("yolo_model.") else lname
        top_neurons[mapped] = group["NeuronIndex"].tolist()

    records = []
    for suite_size in SUITE_SIZES:
        n = min(suite_size, len(all_images))
        for trial in range(3):  # 3 random trials per size
            indices = random.sample(range(len(all_images)), n)
            suite = all_images[indices]

            # Diversity (Pielou's evenness)
            preds = get_yolo_predictions(torch_model, suite, device)
            J = pielou_evenness(preds)

            # WISDOM coverage
            w_cov = wisdom_coverage(torch_model, suite, device, top_neurons)

            # Baseline neuron coverage
            nc = neuron_coverage(torch_model, suite, device)

            records.append({
                "Suite Size": n,
                "Trial": trial,
                "Pielou Evenness": J,
                "WISDOM Coverage": w_cov,
                "Neuron Coverage": nc,
            })
            print(f"  n={n}, trial={trial}: J'={J:.4f}, wisdom_cov={w_cov:.4f}, NC={nc:.4f}")

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    # Compute Pearson correlations
    if len(df) > 2:
        corr_wisdom = df["Pielou Evenness"].corr(df["WISDOM Coverage"])
        corr_nc = df["Pielou Evenness"].corr(df["Neuron Coverage"])
        print(f"\nPearson correlation (evenness vs WISDOM coverage): {corr_wisdom:.4f}")
        print(f"Pearson correlation (evenness vs Neuron Coverage): {corr_nc:.4f}")

    print(f"Saved: {out_csv}")
    return out_csv


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RQ4: Correlation evaluation for YOLOv11")
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file", default="neuron_eval_out/wisdom_yolo11n_scores.csv")
    p.add_argument("--out-csv", default="results/rq4_yolo11n_correlation.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-images", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=320)
    args = p.parse_args()
    run_rq4(**vars(args))

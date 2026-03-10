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

SUITE_SIZES = [10, 50, 100, 200]


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
    batch_size: int = 16,
) -> float:
    """Fraction of WISDOM-selected neurons that are activated above threshold."""
    all_acts: Dict[str, list] = {k: [] for k in top_neurons}
    for i in range(0, len(images), batch_size):
        acts = collect_activations(model, images[i:i + batch_size], device, top_neurons)
        for lname, act_t in acts.items():
            all_acts[lname].append(act_t.cpu())
    active = 0
    total = 0
    for lname in top_neurons:
        if not all_acts[lname]:
            continue
        combined = torch.cat(all_acts[lname], dim=0)
        threshold = combined.abs().mean() * 0.1 + 1e-6
        active += (combined.abs() > threshold).sum().item()
        total += combined.numel()
    return active / max(total, 1)


def neuron_coverage(model: nn.Module, images: torch.Tensor, device: str,
                    threshold: float = 0.5, batch_size: int = 16) -> float:
    """Traditional NC: fraction of all neurons activated above threshold.

    Processes images in batches to avoid GPU OOM.
    """
    model.eval().to(device)
    all_batch_acts = []

    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        acts_list: list = []
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
            model(chunk.to(device))

        for h in hooks:
            h.remove()

        if acts_list:
            all_batch_acts.append(torch.cat(acts_list, dim=1))

    if not all_batch_acts:
        return 0.0
    all_acts = torch.cat(all_batch_acts, dim=0)  # (total_images, total_neurons)
    return (all_acts.abs() > threshold).float().mean().item()


# ── Get YOLO predictions ──────────────────────────────────────────
def get_yolo_predictions(model: nn.Module, images: torch.Tensor, device: str,
                         batch_size: int = 16) -> List[int]:
    """Return list of predicted class IDs across all images and detections."""
    model.eval().to(device)
    predictions = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            out = model(chunk.to(device))
            preds = out[0] if isinstance(out, (tuple, list)) else out
            cls_scores = preds[:, 4:, :]  # (B, 80, A)
            top_classes = cls_scores.argmax(dim=1)  # (B, A)
            top_conf = cls_scores.max(dim=1).values  # (B, A)
            mask = top_conf > 0.25
            predictions.extend(top_classes[mask].cpu().tolist())
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
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Summary table
    table_lines = []
    table_lines.append("=" * 80)
    table_lines.append("RQ4: Correlation – WISDOM Coverage vs Pielou's Evenness")
    table_lines.append("=" * 80)
    table_lines.append(f"{'Suite Size':>10} {'Trial':>6} {'Pielou J':>10} {'WISDOM Cov':>12} {'Neuron Cov':>12}")
    table_lines.append("-" * 80)
    for _, row in df.iterrows():
        table_lines.append(
            f"{int(row['Suite Size']):>10} {int(row['Trial']):>6} "
            f"{row['Pielou Evenness']:10.4f} {row['WISDOM Coverage']:12.4f} {row['Neuron Coverage']:12.4f}"
        )
    table_lines.append("-" * 80)

    # Per-suite-size summary
    table_lines.append(f"\n{'Suite Size':>10} {'Avg J':>10} {'Avg WISDOM':>12} {'Avg NC':>12}")
    table_lines.append("-" * 50)
    for ss in sorted(df["Suite Size"].unique()):
        sub = df[df["Suite Size"] == ss]
        table_lines.append(
            f"{int(ss):>10} {sub['Pielou Evenness'].mean():10.4f} "
            f"{sub['WISDOM Coverage'].mean():12.4f} {sub['Neuron Coverage'].mean():12.4f}"
        )

    # Pearson correlations
    corr_wisdom = float("nan")
    corr_nc = float("nan")
    if len(df) > 2:
        corr_wisdom = df["Pielou Evenness"].corr(df["WISDOM Coverage"])
        corr_nc = df["Pielou Evenness"].corr(df["Neuron Coverage"])
        table_lines.append("")
        table_lines.append(f"Pearson r (Evenness vs WISDOM Coverage): {corr_wisdom:.4f}")
        table_lines.append(f"Pearson r (Evenness vs Neuron Coverage):  {corr_nc:.4f}")
        table_lines.append(f"WISDOM {'shows stronger' if abs(corr_wisdom) > abs(corr_nc) else 'shows weaker'} correlation than baseline NC")
    table_lines.append("=" * 80)

    table_str = "\n".join(table_lines)
    print(table_str)

    # Save log
    log_dir = os.path.join(os.path.dirname(out_csv) or ".", "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "rq4_results.log")
    with open(log_path, "w") as f:
        f.write(table_str + "\n\n")
        f.write("Full records:\n")
        f.write(df.to_string(index=False))
        f.write("\n")
    print(f"Log saved: {log_path}")

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

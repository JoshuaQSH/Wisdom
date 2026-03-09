#!/usr/bin/env python
"""
run_rq3.py – RQ3: Effectiveness at detecting adversarial examples
==================================================================
Generates adversarial examples (FGSM, PGD) and measures whether
WISDOM-identified neurons show higher coverage change when adversarial
inputs are injected into the test set.

For different sample sizes and error rates:
  1. Sample correct inputs.
  2. Replace a fraction with adversarial examples.
  3. Compute coverage for clean vs mixed datasets.
  4. Report normalised change: |mixed_coverage - clean_coverage| / clean_coverage

Output: rq3_effectiveness.csv
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wisdom_yolo_train import COCOImageDataset, _collate
from wisdom.utils.yolo_wrapper import YOLOWrapper
from run_rq2 import collect_activations

SAMPLE_SIZES = [10]  # Reduced for feasibility; paper uses [100, 1000, 3000]
ERROR_RATES = [0.01, 0.05, 0.10]
ATTACKS = ["fgsm", "pgd"]


# ── Adversarial attack helpers ─────────────────────────────────────
def fgsm_attack(wrapper: nn.Module, images: torch.Tensor, device: str, eps: float = 0.03) -> torch.Tensor:
    """Fast Gradient Sign Method."""
    wrapper.eval().to(device)
    x = images.to(device).requires_grad_(True)
    out = wrapper(x)
    loss = out.sum()
    loss.backward()
    adv = x + eps * x.grad.sign()
    return adv.clamp(0, 1).detach().cpu()


def pgd_attack(
    wrapper: nn.Module, images: torch.Tensor, device: str,
    eps: float = 0.03, alpha: float = 0.01, steps: int = 5,
) -> torch.Tensor:
    """Projected Gradient Descent attack."""
    wrapper.eval().to(device)
    x_adv = images.clone().to(device)
    x_orig = images.to(device)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        out = wrapper(x_adv)
        loss = out.sum()
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x_orig, -eps, eps)
            x_adv = torch.clamp(x_orig + delta, 0, 1)
    return x_adv.detach().cpu()


ATTACK_FNS = {
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
}


# ── Coverage computation ──────────────────────────────────────────
def batch_coverage(
    model: nn.Module, images: torch.Tensor, device: str,
    top_neurons: Dict[str, List[int]],
) -> float:
    """Compute fraction of top neurons that are 'active' (above mean)."""
    acts = collect_activations(model, images, device, top_neurons)
    active = 0
    total = 0
    for lname, act_t in acts.items():
        mean_act = act_t.abs().mean()
        active += (act_t.abs() > mean_act * 0.5).sum().item()
        total += act_t.numel()
    return active / max(total, 1)


# ── RQ3 experiment ─────────────────────────────────────────────────
def run_rq3(
    weights: str,
    img_dir: str,
    csv_file: str,
    out_csv: str = "rq3_effectiveness.csv",
    device: str = "cuda:0",
    num_images: int = 20,
    batch_size: int = 2,
    imgsz: int = 320,
) -> str:
    from ultralytics import YOLO

    yolo = YOLO(weights)
    torch_model = yolo.model.eval()
    wrapper = YOLOWrapper(torch_model, num_classes=80)
    wrapper.eval().to(device)

    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    all_images = torch.stack([ds[i][0] for i in range(len(ds))])

    # Get top neurons from WISDOM CSV
    scores_df = pd.read_csv(csv_file)
    top_k_df = scores_df.nlargest(20, "Score")
    top_neurons: Dict[str, List[int]] = {}
    for lname, group in top_k_df.groupby("LayerName"):
        mapped = lname.replace("yolo_model.", "") if lname.startswith("yolo_model.") else lname
        top_neurons[mapped] = group["NeuronIndex"].tolist()

    records = []
    for attack_name in ATTACKS:
        attack_fn = ATTACK_FNS[attack_name]
        print(f"\n=== Attack: {attack_name.upper()} ===")

        # Generate adversarial examples for all images
        adv_images = attack_fn(wrapper, all_images, device)

        for sample_size in SAMPLE_SIZES:
            n = min(sample_size, len(all_images))
            indices = random.sample(range(len(all_images)), n)
            clean_batch = all_images[indices]

            # Clean coverage
            clean_cov = batch_coverage(torch_model, clean_batch, device, top_neurons)

            for error_rate in ERROR_RATES:
                n_adv = max(1, int(n * error_rate))
                # Mix clean + adversarial
                mixed = clean_batch.clone()
                adv_indices = random.sample(range(n), n_adv)
                for ai in adv_indices:
                    orig_idx = indices[ai]
                    mixed[ai] = adv_images[orig_idx]

                mixed_cov = batch_coverage(torch_model, mixed, device, top_neurons)
                norm_change = abs(mixed_cov - clean_cov) / max(clean_cov, 1e-8)

                records.append({
                    "Attack": attack_name.upper(),
                    "Sample Size": n,
                    "Error Rate": error_rate,
                    "Clean Coverage": clean_cov,
                    "Mixed Coverage": mixed_cov,
                    "Normalised Change": norm_change,
                })
                print(f"  n={n}, err={error_rate}: clean={clean_cov:.4f}, mixed={mixed_cov:.4f}, Δ={norm_change:.4f}")

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    return out_csv


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RQ3: Adversarial effectiveness for YOLOv11")
    p.add_argument("--weights", default="standalone/models/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file", default="wisdom_yolo11n_scores.csv")
    p.add_argument("--out-csv", default="rq3_yolo11n_effectiveness.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-images", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=320)
    args = p.parse_args()
    run_rq3(**vars(args))

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

SAMPLE_SIZES = [100, 500, 1000]
ERROR_RATES = [0.01, 0.05, 0.10]
ATTACKS = ["fgsm", "pgd"]


# ── Adversarial attack helpers ─────────────────────────────────────
def fgsm_attack(wrapper: nn.Module, images: torch.Tensor, device: str,
                eps: float = 0.03, batch_size: int = 4) -> torch.Tensor:
    """Fast Gradient Sign Method (batched to avoid OOM)."""
    wrapper.eval().to(device)
    adv_list = []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].to(device).requires_grad_(True)
        out = wrapper(x)
        loss = out.sum()
        loss.backward()
        adv = x + eps * x.grad.sign()
        adv_list.append(adv.clamp(0, 1).detach().cpu())
    return torch.cat(adv_list, dim=0)


def pgd_attack(
    wrapper: nn.Module, images: torch.Tensor, device: str,
    eps: float = 0.03, alpha: float = 0.01, steps: int = 5,
    batch_size: int = 4,
) -> torch.Tensor:
    """Projected Gradient Descent attack (batched to avoid OOM)."""
    wrapper.eval().to(device)
    adv_list = []
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        x_adv = chunk.clone().to(device)
        x_orig = chunk.to(device)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            out = wrapper(x_adv)
            loss = out.sum()
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x_orig, -eps, eps)
                x_adv = torch.clamp(x_orig + delta, 0, 1)
        adv_list.append(x_adv.detach().cpu())
    return torch.cat(adv_list, dim=0)


ATTACK_FNS = {
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
}


# ── Coverage computation ──────────────────────────────────────────
def batch_coverage(
    model: nn.Module, images: torch.Tensor, device: str,
    top_neurons: Dict[str, List[int]],
    batch_size: int = 8,
) -> float:
    """Compute fraction of top neurons that are 'active' (above mean).

    Processes images in batches to avoid GPU OOM on large inputs.
    """
    # Accumulate activations across batches
    all_acts: Dict[str, list] = {k: [] for k in top_neurons}
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        acts = collect_activations(model, chunk, device, top_neurons)
        for lname, act_t in acts.items():
            all_acts[lname].append(act_t.cpu())

    active = 0
    total = 0
    for lname in top_neurons:
        if not all_acts[lname]:
            continue
        combined = torch.cat(all_acts[lname], dim=0)
        mean_act = combined.abs().mean()
        active += (combined.abs() > mean_act * 0.5).sum().item()
        total += combined.numel()
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
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Line plot: Normalised Change vs Error Rate for each attack
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        attack_colors = {"FGSM": "#d62728", "PGD": "#1f77b4"}
        fig, ax = plt.subplots(figsize=(8, 5))
        for attack_name_upper in df["Attack"].unique():
            subset = df[df["Attack"] == attack_name_upper]
            err_rates = sorted(subset["Error Rate"].unique())
            norm_changes = [subset[subset["Error Rate"] == er]["Normalised Change"].mean() for er in err_rates]
            color = attack_colors.get(attack_name_upper, "#333333")
            ax.plot(err_rates, norm_changes, marker="o", linewidth=2, markersize=8,
                    label=attack_name_upper, color=color)

        ax.set_xlabel("Error Rate (fraction of adversarial inputs)", fontsize=11)
        ax.set_ylabel("Normalised Coverage Change", fontsize=11)
        ax.set_title("RQ3: Adversarial Detection via WISDOM Coverage", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = out_csv.replace(".csv", "_plot.pdf")
        fig.savefig(plot_path, format="pdf", dpi=1200, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {plot_path}")
    except Exception as e:
        print(f"Warning: could not generate plot: {e}")

    print(f"\nSaved: {out_csv}")
    return out_csv


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RQ3: Adversarial effectiveness for YOLOv11")
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file", default="neuron_eval_out/wisdom_yolo11n_scores.csv")
    p.add_argument("--out-csv", default="results/rq3_yolo11n_effectiveness.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-images", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=320)
    args = p.parse_args()
    run_rq3(**vars(args))

#!/usr/bin/env python
"""
run_rq3.py – RQ3: Effectiveness at detecting adversarial examples
==================================================================
Generates adversarial examples (FGSM, PGD, CW) and measures whether
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
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wisdom_yolo_train import COCOImageDataset, _collate
from wisdom.utils.detection_loader import load_detection_model
from wisdom.utils.yolo_wrapper import YOLOWrapper
from run_cases.run_rq2 import collect_activations

SAMPLE_SIZES = [100, 500, 1000]
ERROR_RATES = [0.01, 0.05, 0.10]
ATTACKS = ["fgsm", "pgd"]


def _parse_int_list(value: str | Sequence[int]) -> List[int]:
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


def _parse_float_list(value: str | Sequence[float]) -> List[float]:
    if isinstance(value, str):
        return [float(v.strip()) for v in value.split(",") if v.strip()]
    return [float(v) for v in value]


def _parse_attack_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        attacks = [v.strip().lower() for v in value.split(",") if v.strip()]
    else:
        attacks = [str(v).lower() for v in value]
    unsupported = [attack for attack in attacks if attack not in ATTACK_FNS]
    if unsupported:
        raise ValueError(f"Unsupported attack(s): {unsupported}. Supported: {sorted(ATTACK_FNS)}")
    return attacks


def _set_parameter_grad(module: nn.Module, requires_grad: bool) -> List[bool]:
    old = []
    for param in module.parameters():
        old.append(param.requires_grad)
        param.requires_grad_(requires_grad)
    return old


def _restore_parameter_grad(module: nn.Module, old: Sequence[bool]) -> None:
    for param, requires_grad in zip(module.parameters(), old):
        param.requires_grad_(requires_grad)


# ── Adversarial attack helpers ─────────────────────────────────────
def fgsm_attack(wrapper: nn.Module, images: torch.Tensor, device: str,
                eps: float = 0.03, batch_size: int = 4) -> torch.Tensor:
    """Fast Gradient Sign Method (batched to avoid OOM)."""
    wrapper.eval().to(device)
    old_grad = _set_parameter_grad(wrapper, False)
    adv_list = []
    try:
        for i in range(0, len(images), batch_size):
            x = images[i:i + batch_size].to(device).requires_grad_(True)
            out = wrapper(x)
            loss = out.sum()
            loss.backward()
            adv = x + eps * x.grad.sign()
            adv_list.append(adv.clamp(0, 1).detach().cpu())
    finally:
        _restore_parameter_grad(wrapper, old_grad)
    return torch.cat(adv_list, dim=0)


def pgd_attack(
    wrapper: nn.Module, images: torch.Tensor, device: str,
    eps: float = 0.03, alpha: float = 0.01, steps: int = 5,
    batch_size: int = 4,
) -> torch.Tensor:
    """Projected Gradient Descent attack (batched to avoid OOM)."""
    wrapper.eval().to(device)
    old_grad = _set_parameter_grad(wrapper, False)
    adv_list = []
    try:
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
    finally:
        _restore_parameter_grad(wrapper, old_grad)
    return torch.cat(adv_list, dim=0)


def _to_tanh_space(images: torch.Tensor) -> torch.Tensor:
    scaled = images.mul(2).sub(1).clamp(-0.999999, 0.999999)
    return torch.atanh(scaled)


def _from_tanh_space(tanh_images: torch.Tensor) -> torch.Tensor:
    return torch.tanh(tanh_images).add(1).mul(0.5)


def cw_attack(
    wrapper: nn.Module,
    images: torch.Tensor,
    device: str,
    c: float = 1.0,
    confidence: float = 0.0,
    steps: int = 20,
    lr: float = 0.01,
    batch_size: int = 2,
) -> torch.Tensor:
    """Untargeted CW-L2 attack over YOLOWrapper class-score logits.

    YOLO does not expose ground-truth labels through this runner, so the
    current top wrapper class is used as the source class to suppress.
    """
    wrapper.eval().to(device)
    old_grad = _set_parameter_grad(wrapper, False)
    adv_list = []
    try:
        for i in range(0, len(images), batch_size):
            x_orig = images[i:i + batch_size].to(device)
            with torch.no_grad():
                labels = wrapper(x_orig).argmax(dim=1)

            w = _to_tanh_space(x_orig).detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([w], lr=lr)
            best_adv = x_orig.detach().clone()
            best_l2 = torch.full((x_orig.size(0),), float("inf"), device=device)

            for _ in range(steps):
                adv = _from_tanh_space(w)
                logits = wrapper(adv)
                real = logits.gather(1, labels[:, None]).squeeze(1)
                other_logits = logits.clone()
                other_logits.scatter_(1, labels[:, None], -float("inf"))
                other = other_logits.max(dim=1).values

                f_loss = torch.clamp(real - other + confidence, min=0)
                l2 = (adv - x_orig).flatten(1).pow(2).sum(dim=1)
                loss = (l2 + c * f_loss).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    is_adv = other > real
                    improved = is_adv & (l2 < best_l2)
                    best_l2[improved] = l2[improved]
                    best_adv[improved] = adv.detach()[improved]

            with torch.no_grad():
                final_adv = _from_tanh_space(w).detach()
                never_adv = torch.isinf(best_l2)
                best_adv[never_adv] = final_adv[never_adv]
                adv_list.append(best_adv.clamp(0, 1).cpu())
    finally:
        _restore_parameter_grad(wrapper, old_grad)
    return torch.cat(adv_list, dim=0)


ATTACK_FNS = {
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
    "cw": cw_attack,
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
    sample_sizes: str | Sequence[int] = SAMPLE_SIZES,
    error_rates: str | Sequence[float] = ERROR_RATES,
    attacks: str | Sequence[str] = ATTACKS,
    num_runs: int = 1,
    seed: int = 42,
    fgsm_eps: float = 0.03,
    pgd_eps: float = 0.03,
    pgd_alpha: float = 0.01,
    pgd_steps: int = 5,
    cw_c: float = 1.0,
    cw_confidence: float = 0.0,
    cw_steps: int = 20,
    cw_lr: float = 0.01,
) -> str:
    sample_sizes = _parse_int_list(sample_sizes)
    error_rates = _parse_float_list(error_rates)
    attacks = _parse_attack_list(attacks)
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")

    bundle = load_detection_model(weights, device=device)
    torch_model = bundle.model.eval()
    wrapper = YOLOWrapper(torch_model, num_classes=bundle.num_classes)
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

    max_sample_size = max(sample_sizes)
    if len(all_images) < max_sample_size:
        print(f"Warning: requested max sample size {max_sample_size}, but only {len(all_images)} images are available.")

    records = []
    clean_cache: Dict[tuple[int, int], tuple[List[int], torch.Tensor, float]] = {}
    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        run_rng = random.Random(run_seed)
        for sample_size in sample_sizes:
            n = min(sample_size, len(all_images))
            indices = run_rng.sample(range(len(all_images)), n)
            clean_batch = all_images[indices]
            clean_cov = batch_coverage(torch_model, clean_batch, device, top_neurons, batch_size=batch_size)
            clean_cache[(run_idx, n)] = (indices, clean_batch, clean_cov)
            print(f"Run {run_idx + 1}/{num_runs}: n={n}, clean coverage={clean_cov:.4f}")

    attack_kwargs: Dict[str, dict] = {
        "fgsm": {"eps": fgsm_eps, "batch_size": batch_size},
        "pgd": {"eps": pgd_eps, "alpha": pgd_alpha, "steps": pgd_steps, "batch_size": batch_size},
        "cw": {
            "c": cw_c,
            "confidence": cw_confidence,
            "steps": cw_steps,
            "lr": cw_lr,
            "batch_size": batch_size,
        },
    }

    for attack_name in attacks:
        attack_fn: Callable[..., torch.Tensor] = ATTACK_FNS[attack_name]
        print(f"\n=== Attack: {attack_name.upper()} ===")

        # Generate adversarial examples once and reuse them across repeated
        # clean/adversarial subset draws.
        adv_images = attack_fn(wrapper, all_images, device, **attack_kwargs[attack_name])

        for run_idx in range(num_runs):
            run_seed = seed + run_idx
            for sample_size in sample_sizes:
                n = min(sample_size, len(all_images))
                indices, clean_batch, clean_cov = clean_cache[(run_idx, n)]

                for error_rate in error_rates:
                    n_adv = max(1, int(round(n * error_rate)))
                    mix_rng = random.Random(f"{run_seed}:{attack_name}:{n}:{error_rate}")
                    mixed = clean_batch.clone()
                    adv_indices = mix_rng.sample(range(n), min(n_adv, n))
                    for ai in adv_indices:
                        orig_idx = indices[ai]
                        mixed[ai] = adv_images[orig_idx]

                    mixed_cov = batch_coverage(torch_model, mixed, device, top_neurons, batch_size=batch_size)
                    norm_change = abs(mixed_cov - clean_cov) / max(clean_cov, 1e-8)

                    records.append({
                        "Run": run_idx + 1,
                        "Seed": run_seed,
                        "Attack": attack_name.upper(),
                        "Sample Size": n,
                        "Error Rate": error_rate,
                        "Adversarial Count": len(adv_indices),
                        "Clean Coverage": clean_cov,
                        "Mixed Coverage": mixed_cov,
                        "Normalised Change": norm_change,
                    })
                    print(
                        f"  run={run_idx + 1}, n={n}, err={error_rate:.2f}, "
                        f"adv={len(adv_indices)}: clean={clean_cov:.4f}, "
                        f"mixed={mixed_cov:.4f}, Δ={norm_change:.4f}"
                    )

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Line plot: Normalised Change vs Error Rate for each attack
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        attack_colors = {"FGSM": "#d62728", "PGD": "#1f77b4", "CW": "#2ca02c"}
        fig, ax = plt.subplots(figsize=(8, 5))
        for attack_name_upper in df["Attack"].unique():
            subset = df[df["Attack"] == attack_name_upper]
            err_rates = sorted(subset["Error Rate"].unique())
            norm_changes = [subset[subset["Error Rate"] == er]["Normalised Change"].mean() for er in err_rates]
            color = attack_colors.get(attack_name_upper, "#333333")
            ax.plot(err_rates, norm_changes, marker="o", linewidth=2, markersize=8,
                    label=attack_name_upper, color=color)

        ax.set_xlim(0.00, 0.10)
        x_ticks = np.arange(0.00, 0.101, 0.01)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x:.2f}" for x in x_ticks])
        
        ax.set_xlabel("Error Rate", fontsize=12)
        ax.set_ylabel("Coverage Change (Normalized)", fontsize=12)
        # ax.set_title("RQ3: Adversarial Detection via WISDOM Coverage", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = out_csv.replace(".csv", "_plot.pdf")
        fig.savefig(plot_path, format="pdf", dpi=1200, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {plot_path}")
    except Exception as e:
        print(f"Warning: could not generate plot: {e}")

    summary = df.groupby(["Attack", "Sample Size", "Error Rate"])["Normalised Change"].agg(["mean", "std"]).reset_index()
    log_dir = os.path.join(os.path.dirname(out_csv) or ".", "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "rq3_results.log")
    with open(log_path, "w") as f:
        f.write("RQ3: Adversarial effectiveness summary\n")
        f.write("=" * 60 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nFull records:\n")
        f.write(df.to_string(index=False))
        f.write("\n")
    print(f"Log saved: {log_path}")

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
    p.add_argument("--sample-sizes", default="100,500,1000")
    p.add_argument("--error-rates", default="0.01,0.05,0.10")
    p.add_argument("--attacks", default="fgsm,pgd")
    p.add_argument("--num-runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fgsm-eps", type=float, default=0.03)
    p.add_argument("--pgd-eps", type=float, default=0.03)
    p.add_argument("--pgd-alpha", type=float, default=0.01)
    p.add_argument("--pgd-steps", type=int, default=5)
    p.add_argument("--cw-c", type=float, default=1.0)
    p.add_argument("--cw-confidence", type=float, default=0.0)
    p.add_argument("--cw-steps", type=int, default=20)
    p.add_argument("--cw-lr", type=float, default=0.01)
    args = p.parse_args()
    run_rq3(**vars(args))

#!/usr/bin/env python
"""
run_rq3_opt.py – Optimised RQ3: Adversarial Effectiveness
==========================================================
Improvements over run_rq3.py:
1. Stronger attacks: ε=0.1 (FGSM), ε=0.1/10-step (PGD)
2. Feature-space attack: directly disrupts backbone features
3. Stratified coverage with calibrated thresholds
4. Layer-wise neuron selection

Reuses: fgsm_attack, pgd_attack logic from run_rq3.py
        collect_activations from run_rq2.py
"""
from __future__ import annotations
import argparse, os, sys, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wisdom_yolo_train import COCOImageDataset, _collate
from wisdom.utils.yolo_wrapper import YOLOWrapper
from optimize.coverage_utils import (
    load_layerwise_top_neurons,
    load_groupwise_top_neurons,
    ActivationCollector, calibrate_thresholds,
    compute_stratified_coverage,
    ClusterCoverageComputer,
)


# ── Stronger attacks ─────────────────────────────────────────────────
def fgsm_strong(wrapper, images, device, eps=0.1, batch_size=4):
    """FGSM with larger epsilon."""
    adv_list = []
    for i in range(0, len(images), batch_size):
        x = images[i:i+batch_size].to(device).requires_grad_(True)
        out = wrapper(x)
        out.sum().backward()
        adv = (x + eps * x.grad.sign()).clamp(0, 1)
        adv_list.append(adv.detach().cpu())
    return torch.cat(adv_list)


def pgd_strong(wrapper, images, device, eps=0.1, alpha=0.02, steps=10, batch_size=4):
    """PGD with more iterations and larger budget."""
    adv_list = []
    for i in range(0, len(images), batch_size):
        chunk = images[i:i+batch_size]
        x_adv = chunk.clone().to(device)
        x_orig = chunk.to(device)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            wrapper(x_adv).sum().backward()
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                delta = (x_adv - x_orig).clamp(-eps, eps)
                x_adv = (x_orig + delta).clamp(0, 1)
        adv_list.append(x_adv.detach().cpu())
    return torch.cat(adv_list)


def feature_attack(model, images, device, eps=0.1, alpha=0.02, steps=10,
                   batch_size=4):
    """
    Feature-space attack: maximise disruption of mid/late backbone
    activations directly, bypassing the detection head.
    """
    # Pick representative mid-to-late layers to attack
    target_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            for idx in [8, 13, 16, 19]:
                if f"model.{idx}." in name and name not in target_layers:
                    target_layers.append(name)
                    break
    if not target_layers:
        return images.clone()  # fallback: no-op

    layer_to_mod = {}
    for name, mod in model.named_modules():
        if name in target_layers:
            layer_to_mod[name] = mod

    adv_list = []
    for i in range(0, len(images), batch_size):
        chunk = images[i:i+batch_size]
        orig = chunk.to(device)

        # Collect clean activations as reference
        clean_acts = {}
        handles = []
        for n, mod in layer_to_mod.items():
            def mk_clean(name):
                def fn(m, inp, out):
                    clean_acts[name] = out.detach().clone()
                return fn
            handles.append(mod.register_forward_hook(mk_clean(n)))
        with torch.no_grad():
            model(orig)
        for h in handles:
            h.remove()

        # Initialize with small noise to avoid zero-gradient saddle point
        x_adv = (orig + torch.randn_like(orig) * 0.01).clamp(0, 1)
        for step in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            adv_acts = {}
            handles = []
            for n, mod in layer_to_mod.items():
                def mk_adv(name):
                    def fn(m, inp, out):
                        adv_acts[name] = out
                    return fn
                handles.append(mod.register_forward_hook(mk_adv(n)))

            model(x_adv)

            loss = torch.tensor(0.0, device=device)
            for n in target_layers:
                if n in adv_acts and n in clean_acts:
                    a = adv_acts[n].flatten()
                    c = clean_acts[n].flatten()
                    # Cosine similarity → we MINIMIZE it (push away from clean)
                    cos = torch.nn.functional.cosine_similarity(
                        a.unsqueeze(0), c.unsqueeze(0))
                    loss = loss + cos  # minimize cosine = maximize disruption
            loss.backward()
            for h in handles:
                h.remove()

            with torch.no_grad():
                if x_adv.grad is not None:
                    # SUBTRACT gradient to minimize cosine (push away)
                    x_adv = x_adv - alpha * x_adv.grad.sign()
                    delta = (x_adv - orig).clamp(-eps, eps)
                    x_adv = (orig + delta).clamp(0, 1)
        adv_list.append(x_adv.detach().cpu())
    return torch.cat(adv_list)


# ── Main ─────────────────────────────────────────────────────────────
def run_rq3_opt(
    weights, img_dir, csv_file, num_images=200,
    batch_size=4, imgsz=320, device="cuda:0",
    out_prefix="results/rq3_opt", per_layer_k=5,
    coverage_mode="plain", neuron_select="per-layer",
):
    from ultralytics import YOLO
    yolo = YOLO(weights)
    model = yolo.model.eval().to(device)
    wrapper = YOLOWrapper(model, num_classes=80).eval().to(device)

    if neuron_select == "per-group":
        top_neurons = load_groupwise_top_neurons(csv_file, per_group_k=per_layer_k)
    else:
        top_neurons = load_layerwise_top_neurons(csv_file, per_layer_k=per_layer_k)
    total_n = sum(len(v) for v in top_neurons.values())
    print(f"Monitoring {total_n} neurons across {len(top_neurons)} layers")
    print(f"Neuron selection: {neuron_select}")
    print(f"Coverage mode: {coverage_mode}")

    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    all_images = torch.stack([ds[i][0] for i in range(len(ds))])

    calib_imgs = all_images[:min(50, len(all_images))]

    # Setup coverage based on mode
    cluster_comp = None
    thresholds = None
    if coverage_mode == "cluster":
        cluster_comp = ClusterCoverageComputer(
            model, top_neurons, device=device,
            method="KMeans", use_silhouette=True, k_max=5,
            combo_mode=neuron_select,
        )
        print("Fitting clusters on calibration images...")
        cluster_comp.fit(calib_imgs, batch_size=batch_size)
        print(f"Clusters fitted: {len(cluster_comp.cluster_sizes)} neurons")
    else:
        thresholds = calibrate_thresholds(model, top_neurons, calib_imgs, device, percentile=50.0)

    collector = ActivationCollector(model, top_neurons, device)
    collector.attach()

    def _compute_coverage_for_images(imgs):
        """Helper: compute coverage for a set of images using the chosen mode."""
        if coverage_mode == "cluster":
            return cluster_comp.coverage(imgs, batch_size=batch_size)
        else:
            covs = []
            for i in range(0, len(imgs), batch_size):
                acts = collector.collect(imgs[i:i+batch_size])
                covs.append(compute_stratified_coverage(acts, thresholds, top_neurons))
            return {k: np.mean([c[k] for c in covs]) for k in covs[0]}

    ATTACKS = {
        "FGSM_0.1": lambda imgs: fgsm_strong(wrapper, imgs, device, eps=0.1, batch_size=batch_size),
        "PGD_0.1":  lambda imgs: pgd_strong(wrapper, imgs, device, eps=0.1, steps=10, batch_size=batch_size),
        "Feature_0.1": lambda imgs: feature_attack(model, imgs, device, eps=0.1, steps=10, batch_size=batch_size),
    }
    SAMPLE_SIZES = [50, 100, 200]
    ERROR_RATES = [0.05, 0.10, 0.20]

    records = []
    for attack_name, attack_fn in ATTACKS.items():
        print(f"\n=== {attack_name} ===")
        adv_all = attack_fn(all_images)

        for ss in SAMPLE_SIZES:
            n = min(ss, len(all_images))
            sample = all_images[:n]
            adv_sample = adv_all[:n]

            clean_cov = _compute_coverage_for_images(sample)

            for er in ERROR_RATES:
                n_adv = max(1, int(n * er))
                mixed = sample.clone()
                perm = torch.randperm(n)
                for j in range(n_adv):
                    mixed[perm[j]] = adv_sample[perm[j]]

                mixed_cov = _compute_coverage_for_images(mixed)

                row = {"attack": attack_name, "sample_size": n, "error_rate": er,
                       "coverage_mode": coverage_mode}
                for k in ("early", "middle", "late", "overall", "variability"):
                    row[f"clean_{k}"] = clean_cov[k]
                    row[f"mixed_{k}"] = mixed_cov[k]
                    row[f"delta_{k}"] = abs(mixed_cov[k] - clean_cov[k])
                    row[f"norm_delta_{k}"] = abs(mixed_cov[k] - clean_cov[k]) / max(clean_cov[k], 1e-8)
                records.append(row)
                print(f"  N={n} er={er:.0%}: Δ_all={row['delta_overall']:.4f} Δ_late={row['delta_late']:.4f} Δ_var={row['delta_variability']:.4f}")

    collector.detach()

    df = pd.DataFrame(records)
    csv_out = f"{out_prefix}_effectiveness.csv"
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    df.to_csv(csv_out, index=False)

    # Summary
    print("\n" + "=" * 80)
    print("RQ3 OPTIMISED – Adversarial Effectiveness")
    print("=" * 80)
    print(f"{'Attack':<16} {'N':>5} {'ε_rate':>6}  {'Δ_late':>8} {'Δ_all':>8} {'Δ_var':>8}")
    print("-" * 60)
    for _, r in df.iterrows():
        print(f"{r['attack']:<16} {r['sample_size']:>5} {r['error_rate']:>6.0%}  "
              f"{r['delta_late']:>8.4f} {r['delta_overall']:>8.4f} {r['delta_variability']:>8.4f}")

    # Plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, grp in zip(axes, ["overall", "late", "variability"]):
            for atk in df["attack"].unique():
                sub = df[df["attack"] == atk]
                for ss in sub["sample_size"].unique():
                    ssub = sub[sub["sample_size"] == ss]
                    ax.plot(ssub["error_rate"], ssub[f"delta_{grp}"], marker="o",
                            label=f"{atk} N={ss}")
            ax.set_xlabel("Error Rate"); ax.set_ylabel(f"Δ {grp}")
            ax.set_title(f"Coverage Change ({grp})"); ax.legend(fontsize=6)
        plt.tight_layout()
        plot_path = f"{out_prefix}_plot.pdf"
        fig.savefig(plot_path, bbox_inches="tight"); plt.close()
        print(f"  Plot → {plot_path}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    log_path = "logs/rq3_opt_results.log"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        f.write(df.to_string(index=False) + "\n")
    print(f"  CSV → {csv_out}   Log → {log_path}")
    return csv_out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file", default="neuron_eval_out/wisdom_yolo11n_scores_5000.csv")
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-prefix", default="results/rq3_opt")
    p.add_argument("--per-layer-k", type=int, default=5)
    p.add_argument("--coverage-mode", choices=["plain", "cluster"], default="plain")
    p.add_argument("--neuron-select", choices=["per-layer", "per-group"],
                   default="per-layer",
                   help="'per-layer': top-k from each layer. "
                        "'per-group': top-k from each group (early/middle/late).")
    a = p.parse_args()
    run_rq3_opt(**vars(a))

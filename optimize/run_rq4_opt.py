#!/usr/bin/env python
"""
run_rq4_opt.py – Optimised RQ4: Correlation with Diversity
==========================================================
Improvements over run_rq4.py:
1. Detection-aware diversity: spatial diversity of object sizes/positions
2. Coverage variability as additional signal
3. Layer-stratified coverage with calibrated thresholds
4. Layer-wise neuron selection

Reuses: pielou_evenness, get_yolo_predictions, neuron_coverage from run_rq4.py
        collect_activations from run_rq2.py
"""
from __future__ import annotations
import argparse, math, os, sys, random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wisdom_yolo_train import COCOImageDataset, _collate
from wisdom.utils.yolo_wrapper import YOLOWrapper
from run_rq4 import pielou_evenness, get_yolo_predictions, neuron_coverage
from optimize.coverage_utils import (
    load_layerwise_top_neurons,
    ActivationCollector, calibrate_thresholds,
    compute_stratified_coverage,
    ClusterCoverageComputer,
)
from torch.utils.data import DataLoader


# ── Detection-aware diversity ────────────────────────────────────────
def spatial_diversity(model, images, device, batch_size=16, imgsz=320):
    """
    Measure spatial diversity of detections: how spread out are object
    sizes and positions across the test suite?

    Returns a [0,1] score. Higher = more spatially diverse detections.
    Uses normalised entropy of discretised (cx, cy, area) bins.
    """
    model.eval().to(device)
    bins = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            chunk = images[i:i+batch_size].to(device)
            out = model(chunk)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            for b in range(preds.shape[0]):
                pred = preds[b]  # (84, A)
                conf = pred[4:, :].max(dim=0).values
                mask = conf > 0.25
                if mask.sum() == 0:
                    continue
                boxes = pred[:4, mask]  # (4, n_det)
                for j in range(boxes.shape[1]):
                    cx, cy, w, h = boxes[:, j].cpu().tolist()
                    # Discretise into grid (5×5 position + 3 size bins)
                    gx = min(4, int(cx / imgsz * 5))
                    gy = min(4, int(cy / imgsz * 5))
                    area = w * h / (imgsz * imgsz)
                    sz = 0 if area < 0.01 else (1 if area < 0.1 else 2)
                    bins.append((gx, gy, sz))

    if len(bins) < 2:
        return 0.0
    counts = Counter(bins)
    S = len(counts)
    if S <= 1:
        return 1.0
    N = len(bins)
    H = -sum((c / N) * math.log(c / N) for c in counts.values())
    return H / math.log(S)


def coverage_variability_metric(
    model, images, top_neurons, thresholds, device, batch_size=8
):
    """
    Compute coverage variability across individual images.
    Higher variability = coverage metric is more discriminating.
    """
    collector = ActivationCollector(model, top_neurons, device)
    collector.attach()
    per_img_covs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        for j in range(batch.shape[0]):
            single = batch[j:j+1]
            acts = collector.collect(single)
            cov = compute_stratified_coverage(acts, thresholds, top_neurons)
            per_img_covs.append(cov["overall"])
    collector.detach()
    return float(np.std(per_img_covs)) if per_img_covs else 0.0


def activation_profile_diversity(
    model, images, top_neurons, thresholds, device, batch_size=8
):
    """
    Compute how diverse the activation PATTERNS are across images.

    For each image, compute binary vector of which WISDOM neurons are active.
    Then compute mean pairwise Hamming distance across all image pairs.

    Higher value = more diverse activation patterns = suite exercises
    different neurons for different images.
    Returns dict with 'overall', 'late' diversity scores.
    """
    collector = ActivationCollector(model, top_neurons, device)
    collector.attach()

    # Build per-image binary activation vectors
    all_profiles = []
    all_late_profiles = []
    for i in range(len(images)):
        single = images[i:i+1]
        acts = collector.collect(single)
        full_vec = []
        late_vec = []
        for layer_name in sorted(top_neurons.keys()):
            if layer_name not in acts:
                continue
            act = acts[layer_name]  # (1, n_neurons)
            thr = thresholds.get(layer_name, torch.zeros(act.shape[1]))
            active = (act.abs().squeeze(0).cpu() > thr.cpu()).float()
            full_vec.append(active)
            # Check if late layer
            try:
                idx = int(layer_name.split('.')[1])
                if idx >= 13:
                    late_vec.append(active)
            except (ValueError, IndexError):
                pass
        if full_vec:
            all_profiles.append(torch.cat(full_vec))
        if late_vec:
            all_late_profiles.append(torch.cat(late_vec))
    collector.detach()

    result = {}
    for key, profiles in [("overall", all_profiles), ("late", all_late_profiles)]:
        if len(profiles) < 2:
            result[key] = 0.0
            continue
        vecs = torch.stack(profiles)  # (N, D)
        # Mean pairwise Hamming distance (normalised to [0,1])
        N, D = vecs.shape
        dists = []
        for i in range(min(N, 50)):  # cap pairwise comparisons
            for j in range(i + 1, min(N, 50)):
                dists.append((vecs[i] != vecs[j]).float().mean().item())
        result[key] = float(np.mean(dists)) if dists else 0.0
    return result


# ── Stratified WISDOM coverage ───────────────────────────────────────
def wisdom_coverage_stratified(
    model, images, top_neurons, thresholds, device, batch_size=16
):
    """Compute stratified coverage over a batch of images."""
    collector = ActivationCollector(model, top_neurons, device)
    collector.attach()
    all_covs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        acts = collector.collect(batch)
        cov = compute_stratified_coverage(acts, thresholds, top_neurons)
        all_covs.append(cov)
    collector.detach()
    return {k: np.mean([c[k] for c in all_covs]) for k in all_covs[0]}


# ── Main ─────────────────────────────────────────────────────────────
def run_rq4_opt(
    weights, img_dir, csv_file, num_images=200,
    imgsz=320, device="cuda:0",
    out_prefix="results/rq4_opt", per_layer_k=5,
    coverage_mode="plain",
):
    from ultralytics import YOLO
    yolo = YOLO(weights)
    model = yolo.model.eval().to(device)

    top_neurons = load_layerwise_top_neurons(csv_file, per_layer_k=per_layer_k)
    total_n = sum(len(v) for v in top_neurons.values())
    print(f"Monitoring {total_n} neurons across {len(top_neurons)} layers")
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
        )
        print("Fitting clusters on calibration images...")
        cluster_comp.fit(calib_imgs, batch_size=4)
        print(f"Clusters fitted: {len(cluster_comp.cluster_sizes)} neurons")
    else:
        thresholds = calibrate_thresholds(model, top_neurons, calib_imgs, device, percentile=75.0)

    def _compute_suite_coverage(suite):
        """Compute coverage for a suite of images using chosen mode."""
        if coverage_mode == "cluster":
            return cluster_comp.coverage(suite, batch_size=4)
        else:
            return wisdom_coverage_stratified(model, suite, top_neurons, thresholds, device)

    SUITE_SIZES = [5, 10, 20, 50]
    if num_images >= 200:
        SUITE_SIZES = [10, 20, 50, 100]
    records = []
    for ss in SUITE_SIZES:
        n = min(ss, len(all_images))
        for trial in range(5):
            indices = random.sample(range(len(all_images)), n)
            suite = all_images[indices]

            # Class diversity (Pielou's evenness)
            preds = get_yolo_predictions(model, suite, device)
            J_class = pielou_evenness(preds)

            # Spatial diversity
            J_spatial = spatial_diversity(model, suite, device, imgsz=imgsz)

            # WISDOM coverage (stratified or cluster)
            w_cov = _compute_suite_coverage(suite)

            # Baseline neuron coverage
            nc = neuron_coverage(model, suite, device)

            # Coverage variability (plain mode: explicit computation; cluster mode: from coverage dict)
            cov_var = coverage_variability_metric(model, suite, top_neurons, thresholds, device) if coverage_mode == "plain" else w_cov.get("variability", 0.0)

            # Activation profile diversity (only meaningful in plain mode)
            apd = activation_profile_diversity(model, suite, top_neurons, thresholds, device) if coverage_mode == "plain" else {"overall": 0.0, "late": 0.0}

            row = {
                "suite_size": n, "trial": trial,
                "coverage_mode": coverage_mode,
                "pielou_class": J_class,
                "spatial_diversity": J_spatial,
                "wisdom_overall": w_cov["overall"],
                "wisdom_early": w_cov["early"],
                "wisdom_middle": w_cov["middle"],
                "wisdom_late": w_cov["late"],
                "wisdom_variability": w_cov["variability"],
                "neuron_coverage": nc,
                "coverage_var": cov_var,
                "act_profile_div": apd["overall"],
                "act_profile_div_late": apd["late"],
            }
            records.append(row)
            print(f"  N={n} trial={trial}: J_cls={J_class:.3f} J_spat={J_spatial:.3f} "
                  f"W_all={w_cov['overall']:.6f} W_late={w_cov['late']:.6f} "
                  f"W_var={w_cov['variability']:.6f} nc={nc:.4f}")

    df = pd.DataFrame(records)
    csv_out = f"{out_prefix}_correlation.csv"
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    df.to_csv(csv_out, index=False)

    # Correlations
    print("\n" + "=" * 80)
    print("RQ4 OPTIMISED – Correlations")
    print("=" * 80)

    diversity_cols = ["pielou_class", "spatial_diversity"]
    coverage_cols = ["wisdom_overall", "wisdom_late", "wisdom_variability",
                     "act_profile_div", "act_profile_div_late",
                     "neuron_coverage", "coverage_var"]

    print(f"{'':>20s}", end="")
    for cc in coverage_cols:
        print(f"  {cc:>18s}", end="")
    print()
    print("-" * (20 + 20 * len(coverage_cols)))

    for dc in diversity_cols:
        print(f"{dc:>20s}", end="")
        for cc in coverage_cols:
            r = df[dc].corr(df[cc])
            marker = "✅" if abs(r) > 0.3 and r > 0 else ("⚠️" if abs(r) > 0.3 else "  ")
            print(f"  {r:>+8.4f} {marker:>8s}", end="")
        print()

    # Per-size summary
    print(f"\n{'Size':>5} {'J_cls':>7} {'J_spat':>7} {'W_all':>7} {'W_late':>7} {'W_var':>7} {'NC':>7}")
    print("-" * 50)
    for ss in sorted(df["suite_size"].unique()):
        sub = df[df["suite_size"] == ss]
        print(f"{ss:>5} {sub['pielou_class'].mean():>7.3f} {sub['spatial_diversity'].mean():>7.3f} "
              f"{sub['wisdom_overall'].mean():>7.3f} {sub['wisdom_late'].mean():>7.3f} "
              f"{sub['wisdom_variability'].mean():>7.3f} {sub['neuron_coverage'].mean():>7.4f}")

    log_path = "logs/rq4_opt_results.log"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        f.write(df.to_string(index=False) + "\n")
    print(f"\n  CSV → {csv_out}   Log → {log_path}")
    return csv_out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file", default="neuron_eval_out/wisdom_yolo11n_scores_5000.csv")
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-prefix", default="results/rq4_opt")
    p.add_argument("--per-layer-k", type=int, default=5)
    p.add_argument("--coverage-mode", choices=["plain", "cluster"], default="plain")
    a = p.parse_args()
    run_rq4_opt(**vars(a))

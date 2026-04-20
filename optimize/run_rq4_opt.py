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
    load_groupwise_top_neurons,
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

def _classify_images_gt(img_dir, all_images_paths):
    """Classify images by ground truth COCO annotations.

    Falls back to model prediction if annotations not found.
    Returns dict: class_id -> list of image indices, and
            dict: img_idx -> list of all class_ids in that image.
    """
    import json
    ann_path = Path(img_dir).parent.parent / "annotations" / "instances_val2017.json"
    if not ann_path.exists():
        return None, None

    with open(ann_path) as f:
        coco = json.load(f)

    # Build filename -> list of category_ids
    img_id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}
    fname_to_cats: Dict[str, list] = {}
    for ann in coco["annotations"]:
        fname = img_id_to_fname.get(ann["image_id"], "")
        fname_to_cats.setdefault(fname, []).append(ann["category_id"])

    # Map image indices to classes
    img_to_classes: Dict[int, list] = {}
    class_to_imgs: Dict[int, list] = {}
    for idx, fpath in enumerate(all_images_paths):
        fname = Path(fpath).name
        cats = fname_to_cats.get(fname, [])
        if cats:
            dominant = max(set(cats), key=cats.count)
            img_to_classes[idx] = cats
            class_to_imgs.setdefault(dominant, []).append(idx)
        else:
            img_to_classes[idx] = []
            class_to_imgs.setdefault(-1, []).append(idx)

    return class_to_imgs, img_to_classes


def _classify_images(model, all_images, device, batch_size=16):
    """Pre-classify images by dominant detected class.

    Returns a dict mapping class_id -> list of image indices.
    """
    model.eval()
    img_to_class: Dict[int, int] = {}
    with torch.no_grad():
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i + batch_size].to(device)
            out = model(batch)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            for b in range(preds.shape[0]):
                pred = preds[b]
                conf = pred[4:, :].max(dim=0).values
                mask = conf > 0.25
                if mask.sum() > 0:
                    cls_scores = pred[4:, mask]
                    cls_ids = cls_scores.argmax(dim=0)
                    dominant = int(cls_ids.mode().values.item())
                else:
                    dominant = -1
                img_to_class[i + b] = dominant

    class_to_imgs: Dict[int, list] = {}
    for idx, cls_id in img_to_class.items():
        class_to_imgs.setdefault(cls_id, []).append(idx)
    return class_to_imgs


def _make_suites(class_to_imgs, n_images_total, suite_size, n_trials=10):
    """Create suites with controlled diversity: biased, balanced, and random.

    Biased suites use as few classes as needed to fill suite_size.
    Balanced suites spread images across as many classes as possible.
    Returns list of (indices, suite_type) tuples.
    """
    all_indices = list(range(n_images_total))
    classes_with_enough = {c: imgs for c, imgs in class_to_imgs.items()
                          if c >= 0 and len(imgs) >= 2}
    class_list = sorted(classes_with_enough.keys())
    # Sort classes by size (largest first) for biased suite construction
    classes_by_size = sorted(class_list, key=lambda c: len(classes_with_enough[c]),
                             reverse=True)
    suites = []

    # --- Biased suites (few classes → low Pielou) ---
    n_biased = n_trials // 3
    for _ in range(n_biased):
        # Use minimum number of classes to fill suite_size
        pool = []
        n_cls = 0
        shuffled = list(classes_by_size)
        random.shuffle(shuffled)
        for c in shuffled:
            pool.extend(classes_with_enough[c])
            n_cls += 1
            if len(pool) >= suite_size:
                break
        if len(pool) >= suite_size:
            idx = random.sample(pool, suite_size)
            suites.append((idx, "biased"))

    # --- Balanced suites (many classes → high Pielou) ---
    n_balanced = n_trials // 3
    for _ in range(n_balanced):
        if len(class_list) >= 5:
            # Pick one image per class, cycle through classes
            shuffled_cls = list(class_list)
            random.shuffle(shuffled_cls)
            idx = []
            used = set()
            cycle = 0
            while len(idx) < suite_size and cycle < 10:
                for c in shuffled_cls:
                    candidates = [i for i in classes_with_enough[c] if i not in used]
                    if candidates:
                        pick = random.choice(candidates)
                        idx.append(pick)
                        used.add(pick)
                        if len(idx) >= suite_size:
                            break
                cycle += 1
            if len(idx) >= suite_size:
                suites.append((idx[:suite_size], "balanced"))

    # --- Random suites ---
    for _ in range(n_trials - len(suites)):
        idx = random.sample(all_indices, min(suite_size, n_images_total))
        suites.append((idx, "random"))

    return suites


def run_rq4_opt(
    weights, img_dir, csv_file, num_images=200,
    imgsz=320, device="cuda:0",
    out_prefix="results/rq4_opt", per_layer_k=5,
    coverage_mode="plain", neuron_select="per-layer",
    n_groups=3, n_clusters=3, n_trials=15,
):
    from scipy.stats import pearsonr, spearmanr
    from ultralytics import YOLO
    from optimize.coverage_utils import set_n_groups, get_group_names

    set_n_groups(n_groups)
    gnames = get_group_names()
    print(f"Layer groups ({n_groups}): {gnames}")

    yolo = YOLO(weights)
    model = yolo.model.eval().to(device)

    if neuron_select == "per-group":
        top_neurons = load_groupwise_top_neurons(csv_file, per_group_k=per_layer_k)
    else:
        top_neurons = load_layerwise_top_neurons(csv_file, per_layer_k=per_layer_k)
    total_n = sum(len(v) for v in top_neurons.values())
    print(f"Monitoring {total_n} neurons across {len(top_neurons)} layers")
    print(f"Neuron selection: {neuron_select}, coverage mode: {coverage_mode}")
    print(f"n_clusters: {n_clusters}, per_layer_k: {per_layer_k}")

    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    all_images = torch.stack([ds[i][0] for i in range(len(ds))])

    # Pre-classify images using GT annotations (cleaner) or model predictions
    print("Pre-classifying images for diversity control...")
    img_paths = [str(ds.paths[i]) for i in range(len(ds))]
    class_to_imgs_gt, img_to_classes_gt = _classify_images_gt(img_dir, img_paths)

    if class_to_imgs_gt is not None:
        class_to_imgs = class_to_imgs_gt
        use_gt_pielou = True
        print(f"Using GT annotations for classification")
    else:
        class_to_imgs = _classify_images(model, all_images, device)
        img_to_classes_gt = None
        use_gt_pielou = False
        print(f"Using model predictions for classification (no GT annotations found)")
    n_classes_found = len([c for c in class_to_imgs if c >= 0])
    print(f"Found {n_classes_found} distinct classes across {len(all_images)} images")

    calib_imgs = all_images[:min(50, len(all_images))]

    # Setup coverage
    cluster_comp = None
    thresholds = None
    if coverage_mode == "cluster":
        cluster_comp = ClusterCoverageComputer(
            model, top_neurons, device=device,
            method="KMeans", use_silhouette=True, k_max=5,
            n_clusters=n_clusters,
            combo_mode=neuron_select,
        )
        print("Fitting clusters on calibration images...")
        cluster_comp.fit(calib_imgs, batch_size=4)
        print(f"Clusters fitted: {len(cluster_comp.cluster_sizes)} neurons")
    else:
        thresholds = calibrate_thresholds(model, top_neurons, calib_imgs, device, percentile=75.0)

    def _compute_suite_coverage(suite):
        if coverage_mode == "cluster":
            return cluster_comp.coverage(suite, batch_size=4)
        else:
            return wisdom_coverage_stratified(model, suite, top_neurons, thresholds, device)

    if num_images >= 500:
        SUITE_SIZES = [10, 20, 50, 100, 200]
    elif num_images >= 200:
        SUITE_SIZES = [10, 20, 50, 100]
    else:
        SUITE_SIZES = [5, 10, 20, 50]
    records = []
    for ss in SUITE_SIZES:
        n = min(ss, len(all_images))
        suites = _make_suites(class_to_imgs, len(all_images), n, n_trials=n_trials)

        for trial_i, (indices, suite_type) in enumerate(suites):
            suite = all_images[indices]

            # Pielou from GT labels (cleaner) or model predictions
            if use_gt_pielou and img_to_classes_gt is not None:
                gt_cats = []
                for idx in indices:
                    gt_cats.extend(img_to_classes_gt.get(idx, []))
                J_class = pielou_evenness(gt_cats) if gt_cats else 0.0
            else:
                preds = get_yolo_predictions(model, suite, device)
                J_class = pielou_evenness(preds)
            J_spatial = spatial_diversity(model, suite, device, imgsz=imgsz)

            w_cov = _compute_suite_coverage(suite)

            nc = neuron_coverage(model, suite, device)
            cov_var = w_cov.get("variability", 0.0)

            row = {
                "suite_size": n, "trial": trial_i,
                "suite_type": suite_type,
                "coverage_mode": coverage_mode,
                "pielou_class": J_class,
                "spatial_diversity": J_spatial,
                "wisdom_overall": w_cov["overall"],
                "wisdom_variability": w_cov["variability"],
                "neuron_coverage": nc,
                "coverage_var": cov_var,
            }
            # Dynamic group columns
            for g in gnames:
                row[f"wisdom_{g}"] = w_cov.get(g, 0.0)

            records.append(row)
            print(f"  N={n} t={trial_i}({suite_type[:3]}): J={J_class:.3f} "
                  f"W={w_cov['overall']:.4f} nc={nc:.4f}")

    df = pd.DataFrame(records)
    csv_out = f"{out_prefix}_correlation.csv"
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    df.to_csv(csv_out, index=False)

    # ── Correlations ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RQ4 – Overall Correlations (Pearson / Spearman)")
    print("=" * 80)

    cov_cols = ["wisdom_overall"] + [f"wisdom_{g}" for g in gnames] + ["neuron_coverage"]

    hdr = f"{'':>20s}"
    for cc in cov_cols:
        hdr += f"  {cc:>20s}"
    print(hdr)
    print("-" * (20 + 22 * len(cov_cols)))

    for dc in ["pielou_class", "spatial_diversity"]:
        line = f"{dc:>20s}"
        for cc in cov_cols:
            if df[dc].std() > 1e-9 and df[cc].std() > 1e-9:
                pr, _ = pearsonr(df[dc], df[cc])
                sr, _ = spearmanr(df[dc], df[cc])
            else:
                pr, sr = 0.0, 0.0
            marker = "✅" if pr > 0.3 else ("⚠️" if pr > 0 else "  ")
            line += f"  P{pr:>+.3f}/S{sr:>+.3f}{marker}"
            print() if False else None  # no-op
        print(line)

    # ── Within-size correlations (controlling for N) ──────────────────
    print("\n" + "=" * 80)
    print("RQ4 – Within-Size Correlations (Pielou vs Coverage)")
    print("=" * 80)
    print(f"{'Size':>5} {'N_pts':>5} {'Pearson_r':>10} {'Spearman_r':>10} "
          f"{'J_range':>12} {'W_range':>12} {'Verdict':>8}")
    print("-" * 70)

    within_pearson = []
    within_spearman = []
    for ss in sorted(df["suite_size"].unique()):
        sub = df[df["suite_size"] == ss]
        j_col = sub["pielou_class"]
        w_col = sub["wisdom_overall"]
        j_range = j_col.max() - j_col.min()
        w_range = w_col.max() - w_col.min()
        if j_col.std() > 1e-9 and w_col.std() > 1e-9:
            pr, pp = pearsonr(j_col, w_col)
            sr, sp = spearmanr(j_col, w_col)
        else:
            pr, sr = 0.0, 0.0
        within_pearson.append(pr)
        within_spearman.append(sr)
        verdict = "✅" if pr > 0.2 else ("~" if pr > 0 else "❌")
        print(f"{ss:>5} {len(sub):>5} {pr:>+10.4f} {sr:>+10.4f} "
              f"{j_range:>12.4f} {w_range:>12.6f} {verdict:>8}")

    mean_pr = np.mean(within_pearson) if within_pearson else 0
    mean_sr = np.mean(within_spearman) if within_spearman else 0
    print(f"{'Mean':>5} {'':>5} {mean_pr:>+10.4f} {mean_sr:>+10.4f}")

    # ── Partial correlation (residualise out suite size) ───────────────
    print("\n── Partial Correlation (controlling for suite_size) ──")
    from numpy.polynomial.polynomial import polyfit, polyval
    for dc in ["pielou_class"]:
        for cc in ["wisdom_overall"] + [f"wisdom_{g}" for g in gnames]:
            x = df["suite_size"].values.astype(float)
            y_div = df[dc].values.astype(float)
            y_cov = df[cc].values.astype(float)
            # Regress out size from both
            c_div = polyfit(x, y_div, 1)
            c_cov = polyfit(x, y_cov, 1)
            r_div = y_div - polyval(x, c_div)
            r_cov = y_cov - polyval(x, c_cov)
            if np.std(r_div) > 1e-9 and np.std(r_cov) > 1e-9:
                pr, _ = pearsonr(r_div, r_cov)
                sr, _ = spearmanr(r_div, r_cov)
            else:
                pr, sr = 0.0, 0.0
            marker = "✅" if pr > 0.15 else ""
            print(f"  {dc:>20s} vs {cc:>20s}: Pearson={pr:>+.4f} Spearman={sr:>+.4f} {marker}")

    # Per-size summary
    print(f"\n{'Size':>5} {'N_pts':>5} {'J_cls':>7} {'W_all':>7} {'NC':>7}")
    print("-" * 40)
    for ss in sorted(df["suite_size"].unique()):
        sub = df[df["suite_size"] == ss]
        print(f"{ss:>5} {len(sub):>5} {sub['pielou_class'].mean():>7.3f} "
              f"{sub['wisdom_overall'].mean():>7.3f} {sub['neuron_coverage'].mean():>7.4f}")

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
    p.add_argument("--neuron-select", choices=["per-layer", "per-group"],
                   default="per-layer",
                   help="'per-layer': top-k from each layer. "
                        "'per-group': top-k from each group.")
    p.add_argument("--n-groups", type=int, default=3,
                   choices=[2, 3, 4, 5],
                   help="Number of layer groups (2/3/4/5)")
    p.add_argument("--n-clusters", type=int, default=3,
                   help="KMeans clusters per neuron (only for cluster mode)")
    p.add_argument("--n-trials", type=int, default=15,
                   help="Number of trials per suite size")
    a = p.parse_args()
    run_rq4_opt(**vars(a))

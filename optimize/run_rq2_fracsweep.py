#!/usr/bin/env python3
"""RQ2 frac-sweep: run multiple perturbation rates with ONE preprocessing pass.

Shares the same preprocessing (model loading, neuron selection, cluster fitting,
activation collection, importance map computation) across all frac values.
Optionally evaluates YOLO mAP on perturbed images.

Usage:
    python optimize/run_rq2_fracsweep.py \
        --weights weights/yolo11n.pt \
        --csv-file neuron_eval_out/wisdom_yolo11n_scores_5000.csv \
        --num-images 5000 --noise-std 0.3 \
        --neuron-select important-layer --top-layers 10 --per-layer-k 9 \
        --n-clusters 3 --coverage-mode cluster \
        --importance wisdom \
        --frac-list 0.02,0.04,0.06,0.08,0.10 \
        --eval-map \
        --out-dir results --log-dir logs --verbose
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Reuse everything from the existing run_rq2_opt module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimize.run_rq2_opt import (
    COCOImageDataset,
    ActivationCollector,
    ClusterCoverageComputer,
    ClusterUnionTracker,
    _collate,
    compute_magnitude_change,
    generate_variant,
    get_object_mask,
    load_groupwise_top_neurons,
    load_important_layer_neurons,
    load_layerwise_top_neurons,
    pixel_importance_gradient,
    verbose_coverage_breakdown,
    verbose_union_diff,
    verbose_combo_overlap,
    NOISE_STD,
    VARIANTS,
)
from optimize.coverage_utils import set_n_groups


# ── mAP evaluation helpers ────────────────────────────────────────

def evaluate_map_on_perturbed(
    yolo_model,
    original_images,     # list of (B, C, H, W) CPU tensors
    importance_maps,     # list of (B, H, W) CPU tensors
    obj_masks,           # list of (B, H, W) CPU tensors
    pixel_frac,
    noise_std,
    device,
    imgsz,
    img_dir,
    batch_size=16,
):
    """Evaluate YOLO detection metrics on original vs perturbed images.

    Returns dict with keys: orig_map50, orig_map, I_map50, I_map, R_map50, R_map,
    orig_precision, orig_recall, I_precision, I_recall, R_precision, R_recall,
    orig_n_det, I_n_det, R_n_det (average detections per image).
    """
    from ultralytics import YOLO

    results = {}

    def _predict_batch(imgs_list, label):
        """Run YOLO predict on batched tensor images, collect per-image stats."""
        all_confs = []
        all_n_dets = []
        for imgs in imgs_list:
            imgs_dev = imgs.to(device)
            preds = yolo_model.predict(
                imgs_dev, imgsz=imgsz, verbose=False,
                conf=0.001, iou=0.6, max_det=300,
                device=device,
            )
            for p in preds:
                n_det = len(p.boxes)
                avg_conf = float(p.boxes.conf.mean()) if n_det > 0 else 0.0
                all_confs.append(avg_conf)
                all_n_dets.append(n_det)
        return {
            f"{label}_avg_conf": np.mean(all_confs),
            f"{label}_avg_n_det": np.mean(all_n_dets),
            f"{label}_median_n_det": np.median(all_n_dets),
        }

    # Original images
    stats_orig = _predict_batch(original_images, "orig")
    results.update(stats_orig)

    # Important-perturbed
    pert_I = []
    for imgs, imp, mask in zip(original_images, importance_maps, obj_masks):
        pert_I.append(generate_variant("I", imgs, imp, mask, pixel_frac, noise_std))
    stats_I = _predict_batch(pert_I, "I")
    results.update(stats_I)

    # Random-perturbed
    pert_R = []
    for imgs, imp, mask in zip(original_images, importance_maps, obj_masks):
        pert_R.append(generate_variant("R", imgs, imp, mask, pixel_frac, noise_std))
    stats_R = _predict_batch(pert_R, "R")
    results.update(stats_R)

    return results


def evaluate_map_with_val(
    weights_path,
    original_images,     # list of (B, C, H, W) CPU tensors
    importance_maps,     # list of (B, H, W) CPU tensors
    obj_masks,           # list of (B, H, W) CPU tensors
    pixel_frac,
    noise_std,
    device,
    imgsz,
    img_dir,
    batch_size=32,
):
    """Evaluate proper mAP using ultralytics val on saved perturbed images.

    Saves perturbed images to a temp dir, copies labels, runs val, cleans up.
    Returns dict with mAP50, mAP50-95, precision, recall for orig/I/R.
    """
    import shutil
    import tempfile
    from PIL import Image
    from ultralytics import YOLO

    label_dir = img_dir.replace("images/val2017", "labels/val2017")
    if not os.path.isdir(label_dir):
        label_dir = img_dir.replace("images", "labels")
    if not os.path.isdir(label_dir):
        print(f"  [mAP] Cannot find labels dir for {img_dir}, skipping val-based mAP")
        return {}

    # Collect all original image filenames in order
    img_files = sorted(os.listdir(img_dir))[:sum(imgs.shape[0] for imgs in original_images)]

    def _save_and_val(images_list, variant_name, tmpdir_base):
        """Save images to tmpdir, create symlinks for labels, run val."""
        img_out = os.path.join(tmpdir_base, variant_name, "images", "val2017")
        lbl_out = os.path.join(tmpdir_base, variant_name, "labels", "val2017")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(os.path.dirname(lbl_out), exist_ok=True)

        # Symlink labels
        os.symlink(os.path.abspath(label_dir), lbl_out)

        # Save images
        idx = 0
        for batch in images_list:
            for i in range(batch.shape[0]):
                if idx >= len(img_files):
                    break
                img_np = (batch[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                fname = img_files[idx]
                Image.fromarray(img_np).save(os.path.join(img_out, fname))
                idx += 1

        # Create data yaml
        data_yaml = os.path.join(tmpdir_base, variant_name, "data.yaml")
        base_path = os.path.join(tmpdir_base, variant_name)
        with open(data_yaml, "w") as f:
            f.write(f"path: {base_path}\n")
            f.write(f"train: images/val2017\n")
            f.write(f"val: images/val2017\n")
            f.write(f"nc: 80\n")
            f.write(f"names: {{0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, "
                    f"5: bus, 6: train, 7: truck, 8: boat, 9: 'traffic light', "
                    f"10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: bench, "
                    f"14: bird, 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow, "
                    f"20: elephant, 21: bear, 22: zebra, 23: giraffe, 24: backpack, "
                    f"25: umbrella, 26: handbag, 27: tie, 28: suitcase, 29: frisbee, "
                    f"30: skis, 31: snowboard, 32: 'sports ball', 33: kite, "
                    f"34: 'baseball bat', 35: 'baseball glove', 36: skateboard, "
                    f"37: surfboard, 38: 'tennis racket', 39: bottle, 40: 'wine glass', "
                    f"41: cup, 42: fork, 43: knife, 44: spoon, 45: bowl, 46: banana, "
                    f"47: apple, 48: sandwich, 49: orange, 50: broccoli, 51: carrot, "
                    f"52: 'hot dog', 53: pizza, 54: donut, 55: cake, 56: chair, 57: couch, "
                    f"58: 'potted plant', 59: bed, 60: 'dining table', 61: toilet, "
                    f"62: tv, 63: laptop, 64: mouse, 65: remote, 66: keyboard, "
                    f"67: 'cell phone', 68: microwave, 69: oven, 70: toaster, 71: sink, "
                    f"72: refrigerator, 73: book, 74: clock, 75: vase, 76: scissors, "
                    f"77: 'teddy bear', 78: 'hair drier', 79: toothbrush}}\n")

        # Run val
        m = YOLO(weights_path)
        metrics = m.val(
            data=data_yaml, imgsz=imgsz, batch=batch_size,
            verbose=False, device=device, plots=False, save=False,
        )
        return {
            f"{variant_name}_map50": float(metrics.box.map50),
            f"{variant_name}_map": float(metrics.box.map),
            f"{variant_name}_precision": float(metrics.box.mp),
            f"{variant_name}_recall": float(metrics.box.mr),
        }

    tmpdir = tempfile.mkdtemp(prefix="rq2_map_")
    results = {}
    try:
        # Original
        print(f"  [mAP] Evaluating original images...")
        r = _save_and_val(original_images, "orig", tmpdir)
        results.update(r)

        # Important-perturbed
        print(f"  [mAP] Generating & evaluating I-perturbed (frac={pixel_frac})...")
        pert_I = []
        for imgs, imp, mask in zip(original_images, importance_maps, obj_masks):
            pert_I.append(generate_variant("I", imgs, imp, mask, pixel_frac, noise_std))
        r = _save_and_val(pert_I, "I", tmpdir)
        results.update(r)
        del pert_I

        # Random-perturbed
        print(f"  [mAP] Generating & evaluating R-perturbed (frac={pixel_frac})...")
        pert_R = []
        for imgs, imp, mask in zip(original_images, importance_maps, obj_masks):
            pert_R.append(generate_variant("R", imgs, imp, mask, pixel_frac, noise_std))
        r = _save_and_val(pert_R, "R", tmpdir)
        results.update(r)
        del pert_R

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return results


# ── Main frac-sweep ───────────────────────────────────────────────

def run_fracsweep(
    weights,
    img_dir,
    csv_file,
    frac_list,
    num_images=5000,
    batch_size=2,
    imgsz=320,
    device="cuda:0",
    out_dir="results",
    log_dir="logs",
    per_layer_k=9,
    num_iters=3,
    importance="wisdom",
    n_clusters=3,
    neuron_select="important-layer",
    n_groups=3,
    top_layers=10,
    layer_score_method="mean_positive",
    noise_std=NOISE_STD,
    verbose=False,
    eval_map=False,
    eval_map_full=False,
    skip_existing=True,
    cache_dir="cache",
):
    from ultralytics import YOLO

    set_n_groups(n_groups)

    yolo = YOLO(weights)
    # Fuse Conv+BN upfront so hook activations are on the same scale
    # throughout preprocessing AND per-frac loops (predict() also fuses,
    # so doing it first avoids a scale mismatch for later fracs).
    yolo.fuse()
    model = yolo.model.eval().to(device)

    # Determine naming prefix
    model_tag = "n"
    if "11s" in weights:
        model_tag = "s"
    elif "11m" in weights:
        model_tag = "m"

    mode_tag = {
        "important-layer": "implayer",
        "per-group": "pergroup",
        "per-layer": "perlayer",
    }[neuron_select]

    imp_tag = "wis" if importance == "wisdom" else importance

    # Check which fracs already have results
    fracs_to_run = []
    for frac in frac_list:
        pct = int(round(frac * 100))
        prefix = f"{model_tag}_{mode_tag}_{imp_tag}_{pct}pct"
        csv_path = os.path.join(out_dir, f"{prefix}_union_coverage.csv")
        if skip_existing and os.path.exists(csv_path):
            print(f"[SKIP] {prefix} already exists at {csv_path}")
        else:
            fracs_to_run.append(frac)

    if not fracs_to_run:
        print("All frac values already have results. Nothing to do.")
        return

    pcts_str = ",".join(f"{f*100:.0f}%" for f in fracs_to_run)
    print(f"\n{'='*72}")
    print(f"RQ2 Frac Sweep: {model_tag} / {mode_tag} / {imp_tag}")
    print(f"  Frac rates to run: {pcts_str}")
    print(f"  Images: {num_images}, iters: {num_iters}, k={per_layer_k}, nc={n_clusters}")
    print(f"{'='*72}\n")

    # ── Neuron selection (once) ──────────────────────────────────
    if neuron_select == "per-group":
        top_neurons = load_groupwise_top_neurons(csv_file, per_group_k=per_layer_k)
    elif neuron_select == "important-layer":
        top_neurons = load_important_layer_neurons(
            csv_file, top_layers=top_layers, per_layer_k=per_layer_k,
            layer_score_method=layer_score_method,
        )
    else:
        top_neurons = load_layerwise_top_neurons(csv_file, per_layer_k=per_layer_k)

    total_n = sum(len(v) for v in top_neurons.values())
    print(f"Monitoring {total_n} neurons across {len(top_neurons)} layers")
    print(f"Neuron selection: {neuron_select}")

    # ── Data loading (once) ──────────────────────────────────────
    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    # ── Cluster fitting (once) ───────────────────────────────────
    calib_imgs = torch.stack([ds[i][0] for i in range(min(50, len(ds)))])
    nc = n_clusters
    combo_mode = "per-layer" if neuron_select in ("important-layer", "hybrid") else neuron_select
    cluster_comp = ClusterCoverageComputer(
        model, top_neurons, device=device,
        method="KMeans", use_silhouette=False,
        k_max=max(nc + 2, 5), n_clusters=nc,
        combo_mode=combo_mode,
    )
    print(f"Fitting clusters (n_clusters={nc})...")
    cluster_comp.fit(calib_imgs, batch_size=batch_size)
    sizes = list(cluster_comp.cluster_sizes.values())
    print(f"Clusters fitted: {len(sizes)} neurons, k range [{min(sizes)}-{max(sizes)}]")

    # ── Phase 1: Preprocessing (once, with caching) ────────────────
    import hashlib
    neuron_hash = hashlib.md5(
        str(sorted((k, v) for k, v in top_neurons.items()
                    for v in (v if isinstance(v, list) else [v]))).encode()
    ).hexdigest()[:8]
    cache_tag = f"{model_tag}_{mode_tag}_{imp_tag}_{num_images}_{neuron_hash}"
    cache_path = os.path.join(cache_dir, f"preprocess_{cache_tag}.pt") if cache_dir else None

    all_images = []
    all_importance = []
    all_obj_masks = []
    all_clean_acts = []

    if cache_path and os.path.exists(cache_path):
        print(f"\nPhase 1: Loading cached preprocessing from {cache_path}...")
        t0 = time.time()
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        all_images = cached["images"]
        all_importance = cached["importance"]
        all_obj_masks = cached["obj_masks"]
        all_clean_acts = cached["clean_acts"]
        print(f"  Loaded {len(all_images)} batches from cache in {time.time()-t0:.0f}s")
    else:
        print("\nPhase 1: Preprocessing images (shared across all frac rates)...")
        collector = ActivationCollector(model, top_neurons, device)
        collector.attach()

        t0 = time.time()
        for batch_idx, batch in enumerate(loader):
            images = batch[0]
            acts = collector.collect(images)
            all_clean_acts.append({k: v.cpu() for k, v in acts.items()})

            imp = pixel_importance_gradient(
                model, images, device, top_neurons=top_neurons, mode=importance,
                cluster_comp=cluster_comp,
            )
            obj_mask = get_object_mask(model, images, device, imgsz=imgsz)

            all_images.append(images.cpu())
            all_importance.append(imp)
            all_obj_masks.append(obj_mask.cpu())

            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {(batch_idx+1)*batch_size}/{len(ds)} images "
                      f"({time.time()-t0:.0f}s)")

        print(f"  Preprocessed {len(ds)} images in {time.time()-t0:.0f}s")
        collector.detach()

        # Save to cache for future runs
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"  Saving preprocessing cache to {cache_path}...")
            torch.save({
                "images": all_images,
                "importance": all_importance,
                "obj_masks": all_obj_masks,
                "clean_acts": all_clean_acts,
            }, cache_path)
            cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"  Cache saved ({cache_size_mb:.0f} MB)")

    # Create collector for per-frac activation collection (always needed)
    collector = ActivationCollector(model, top_neurons, device)

    # ── Phase 2: Per-frac coverage experiments ───────────────────
    map_results_all = []

    for frac in fracs_to_run:
        pct = int(round(frac * 100))
        prefix = f"{model_tag}_{mode_tag}_{imp_tag}_{pct}pct"
        csv_out = os.path.join(out_dir, f"{prefix}_union_coverage.csv")
        log_out = os.path.join(log_dir, f"{prefix}.log")

        print(f"\n{'─'*60}")
        print(f"  Frac = {pct}% ({frac})")
        print(f"  Output: {csv_out}")
        print(f"{'─'*60}")

        # Setup logging for this frac
        log = logging.getLogger("wisdom_opt")
        log.handlers.clear()
        if verbose:
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(log_out, mode="w")
            fh.setFormatter(logging.Formatter("%(name)s %(message)s"))
            log.addHandler(fh)
            log.setLevel(logging.INFO)

        records = []

        for it in range(num_iters):
            print(f"  --- Iteration {it+1}/{num_iters} ---")
            collector.attach()

            # Baseline
            baseline = ClusterUnionTracker(cluster_comp)
            for acts in all_clean_acts:
                baseline.update_from_activations(acts)
            C_O = baseline.coverage()
            gnames = [g for g in C_O.keys() if g not in ("overall", "variability")]
            grp_str = " ".join(f"{g[0].upper()}={C_O[g]:.4f}" for g in gnames)
            print(f"    C(D_O): overall={C_O['overall']:.4f} [{grp_str}]")
            if verbose:
                verbose_coverage_breakdown(baseline, label="D_O")

            variant_trackers = {}
            for vname in VARIANTS:
                tracker = baseline.clone()
                batch_mags = []
                for bi, (images, imp, mask, cacts) in enumerate(zip(
                    all_images, all_importance, all_obj_masks, all_clean_acts,
                )):
                    perturbed = generate_variant(
                        vname, images, imp, mask, frac, noise_std,
                    )
                    acts = collector.collect(perturbed)
                    acts_cpu = {k: v.cpu() for k, v in acts.items()}
                    tracker.update_from_activations(acts_cpu)
                    batch_mags.append(
                        compute_magnitude_change(cacts, acts_cpu, top_neurons)
                    )
                C_union = tracker.coverage()
                variant_trackers[vname] = tracker
                if verbose:
                    verbose_union_diff(baseline, tracker, label=f"D_{vname}")
                avg_mag = {
                    k: float(np.mean([m[k] for m in batch_mags]))
                    for k in list(gnames) + ["overall"]
                }
                row = {
                    "iteration": it, "scope": "dataset",
                    "variant": vname, "coverage_mode": "cluster",
                }
                for k in list(gnames) + ["overall"]:
                    row[f"C_O_{k}"] = C_O[k]
                    row[f"C_union_{k}"] = C_union[k]
                    row[f"delta_{k}"] = C_union[k] - C_O[k]
                    row[f"mag_{k}"] = avg_mag[k]
                records.append(row)
                if vname in ("I", "R"):
                    print(f"    C(D_O∪D_{vname}): overall={C_union['overall']:.4f}"
                          f"  Δ={row['delta_overall']:+.6f}")

            if verbose:
                verbose_combo_overlap(baseline, variant_trackers, combo_mode=combo_mode)

            collector.detach()

        # Save coverage CSV
        df = pd.DataFrame(records)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(csv_out, index=False)

        # Quick I/R summary
        I_delta = df[(df["variant"] == "I") & (df["scope"] == "dataset")]["delta_overall"].mean()
        R_delta = df[(df["variant"] == "R") & (df["scope"] == "dataset")]["delta_overall"].mean()
        ir_ratio = I_delta / max(R_delta, 1e-10)
        print(f"  → I/R = {ir_ratio:.3f} (Δ_I={I_delta:+.6f}, Δ_R={R_delta:+.6f})")

        # ── mAP evaluation ────────────────────────────────────────
        if eval_map:
            print(f"\n  [mAP] Quick detection quality evaluation (predict-based)...")
            map_stats = evaluate_map_on_perturbed(
                yolo, all_images, all_importance, all_obj_masks,
                frac, noise_std, device, imgsz, img_dir,
            )
            map_stats["model"] = model_tag
            map_stats["mode"] = mode_tag
            map_stats["importance"] = imp_tag
            map_stats["frac"] = frac
            map_stats["frac_pct"] = pct
            map_stats["I_R_ratio"] = ir_ratio
            map_stats["delta_I"] = I_delta
            map_stats["delta_R"] = R_delta
            map_results_all.append(map_stats)

            print(f"    Orig: avg_conf={map_stats['orig_avg_conf']:.4f}, "
                  f"avg_det={map_stats['orig_avg_n_det']:.1f}")
            print(f"    I:    avg_conf={map_stats['I_avg_conf']:.4f}, "
                  f"avg_det={map_stats['I_avg_n_det']:.1f}")
            print(f"    R:    avg_conf={map_stats['R_avg_conf']:.4f}, "
                  f"avg_det={map_stats['R_avg_n_det']:.1f}")
            # Ensure model stays on correct device after predict()
            yolo.to(device)
            model = yolo.model

        if eval_map_full:
            print(f"\n  [mAP] Full val-based mAP evaluation...")
            full_map = evaluate_map_with_val(
                weights, all_images, all_importance, all_obj_masks,
                frac, noise_std, device, imgsz, img_dir,
                batch_size=32,
            )
            if full_map:
                full_map["model"] = model_tag
                full_map["mode"] = mode_tag
                full_map["importance"] = imp_tag
                full_map["frac"] = frac
                full_map["frac_pct"] = pct
                full_map["I_R_ratio"] = ir_ratio
                map_results_all.append(full_map)

                print(f"    Orig: mAP50={full_map.get('orig_map50',0):.4f}, "
                      f"mAP={full_map.get('orig_map',0):.4f}")
                print(f"    I:    mAP50={full_map.get('I_map50',0):.4f}, "
                      f"mAP={full_map.get('I_map',0):.4f}")
                print(f"    R:    mAP50={full_map.get('R_map50',0):.4f}, "
                      f"mAP={full_map.get('R_map',0):.4f}")
            # Ensure model stays on correct device after val()
            yolo.to(device)
            model = yolo.model

    # ── Save aggregated mAP results ──────────────────────────────
    if map_results_all:
        map_csv = os.path.join(out_dir,
                               f"{model_tag}_{mode_tag}_{imp_tag}_map_results.csv")
        pd.DataFrame(map_results_all).to_csv(map_csv, index=False)
        print(f"\n  mAP results → {map_csv}")

    print(f"\n{'='*72}")
    print("Frac sweep complete!")
    print(f"{'='*72}")


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="RQ2 frac-sweep: multiple perturbation rates, one preprocessing pass",
    )
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file",
                   default="neuron_eval_out/wisdom_yolo11n_scores_5000.csv")
    p.add_argument("--num-images", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--per-layer-k", type=int, default=9)
    p.add_argument("--num-iters", type=int, default=3)
    p.add_argument("--noise-std", type=float, default=NOISE_STD)
    p.add_argument("--importance", choices=["wisdom", "bga"],
                   default="wisdom")
    p.add_argument("--n-clusters", type=int, default=3)
    p.add_argument("--neuron-select",
                   choices=["per-layer", "per-group", "important-layer"],
                   default="important-layer")
    p.add_argument("--n-groups", type=int, default=3)
    p.add_argument("--top-layers", type=int, default=10)
    p.add_argument("--layer-score-method",
                   choices=["mean_positive", "sum", "max"],
                   default="mean_positive")
    p.add_argument("--frac-list", type=str, default="0.02,0.04,0.06,0.08,0.10",
                   help="Comma-separated list of pixel frac rates")
    p.add_argument("--eval-map", action="store_true",
                   help="Evaluate detection quality (predict-based: avg conf, n_det)")
    p.add_argument("--eval-map-full", action="store_true",
                   help="Evaluate full mAP using ultralytics val (slow, saves images)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-skip", action="store_true",
                   help="Re-run even if results already exist")
    p.add_argument("--cache-dir", type=str, default="cache",
                   help="Directory for preprocessing cache (set to '' to disable)")

    a = p.parse_args()

    frac_list = [float(x.strip()) for x in a.frac_list.split(",")]

    if a.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(name)s %(message)s",
        )

    run_fracsweep(
        weights=a.weights,
        img_dir=a.img_dir,
        csv_file=a.csv_file,
        frac_list=frac_list,
        num_images=a.num_images,
        batch_size=a.batch_size,
        imgsz=a.imgsz,
        device=a.device,
        out_dir=a.out_dir,
        log_dir=a.log_dir,
        per_layer_k=a.per_layer_k,
        num_iters=a.num_iters,
        importance=a.importance,
        n_clusters=a.n_clusters,
        neuron_select=a.neuron_select,
        n_groups=a.n_groups,
        top_layers=a.top_layers,
        layer_score_method=a.layer_score_method,
        noise_std=a.noise_std,
        verbose=a.verbose,
        eval_map=a.eval_map,
        eval_map_full=a.eval_map_full,
        skip_existing=not a.no_skip,
        cache_dir=a.cache_dir or None,
    )

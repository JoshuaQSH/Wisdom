#!/usr/bin/env python
"""
run_rq2_opt.py – Optimised RQ2: Diversity via Perturbation
==========================================================
Tests whether WISDOM-identified neurons track object-relevant features.

Approach:
  U_obj: perturb ONLY pixels inside detected object bounding boxes
  U_bg:  perturb ONLY pixels outside detected objects (background)
  Same noise magnitude per pixel → isolates SPATIAL relevance.

Expected result: WISDOM neurons show MORE activation change for U_obj
than U_bg (ratio > 1), confirming they encode object-relevant features.

Also measures detection output change (confidence drop) to validate.

Reuses: perturb_random_pixels from run_rq2.py
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wisdom_yolo_train import COCOImageDataset, _collate
from optimize.coverage_utils import (
    load_layerwise_top_neurons,
    ActivationCollector, calibrate_thresholds,
    compute_stratified_coverage, compute_magnitude_change,
)
from torch.utils.data import DataLoader

NOISE_STD = 0.15  # moderate noise per pixel


# ── Object mask ──────────────────────────────────────────────────────
def get_object_mask(model, images, device, conf_thresh=0.25, imgsz=320):
    """Binary mask (B,1,H,W): 1 inside detected objects, 0 background."""
    with torch.no_grad():
        out = model(images.to(device))
        preds = out[0] if isinstance(out, (tuple, list)) else out
    B = preds.shape[0]
    masks = torch.zeros(B, 1, imgsz, imgsz, device=device)
    for b in range(B):
        cls_max = preds[b, 4:, :].max(dim=0).values
        confident = cls_max > conf_thresh
        if confident.sum() == 0:
            continue
        boxes = preds[b, :4, confident]
        for j in range(boxes.shape[1]):
            cx, cy, w, h = boxes[:, j]
            w, h = w * 1.3, h * 1.3  # slight expansion
            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = min(imgsz, int(cx + w / 2))
            y2 = min(imgsz, int(cy + h / 2))
            masks[b, 0, y1:y2, x1:x2] = 1.0
    return masks


def perturb_region(images, mask, std=0.15):
    """Add Gaussian noise ONLY where mask==1, zero elsewhere."""
    noise = torch.randn_like(images) * std
    return (images + noise * mask.cpu()).clamp(0, 1)


def detection_confidence(model, images, device):
    """Mean max-class confidence across all anchors."""
    with torch.no_grad():
        out = model(images.to(device))
        preds = out[0] if isinstance(out, (tuple, list)) else out
    return preds[:, 4:, :].max(dim=1).values.mean(dim=1).cpu().numpy()


# ── Main ─────────────────────────────────────────────────────────────
def run_rq2_opt(
    weights, img_dir, csv_file, num_images=200,
    batch_size=2, imgsz=320, device="cuda:0",
    out_prefix="results/rq2_opt", per_layer_k=5,
    num_iters=3,
):
    from ultralytics import YOLO
    yolo = YOLO(weights)
    model = yolo.model.eval().to(device)

    top_neurons = load_layerwise_top_neurons(csv_file, per_layer_k=per_layer_k)
    total_n = sum(len(v) for v in top_neurons.values())
    print(f"Monitoring {total_n} neurons across {len(top_neurons)} layers")

    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    calib_imgs = torch.stack([ds[i][0] for i in range(min(50, len(ds)))])
    thresholds = calibrate_thresholds(model, top_neurons, calib_imgs, device, percentile=50.0)
    print("Thresholds calibrated (p50)")

    collector = ActivationCollector(model, top_neurons, device)
    collector.attach()

    records = []
    for it in range(num_iters):
        all_obj_mag, all_bg_mag = [], []
        all_obj_cov, all_bg_cov = [], []
        conf_drops_obj, conf_drops_bg = [], []
        obj_fracs = []

        for batch in loader:
            images = batch[0]
            obj_mask = get_object_mask(model, images, device, imgsz=imgsz)
            bg_mask = 1.0 - obj_mask

            # Fraction of image covered by objects
            frac = obj_mask.mean(dim=[1, 2, 3]).cpu().numpy()
            obj_fracs.extend(frac.tolist())

            # Clean baseline
            acts_clean = collector.collect(images)
            cov_clean = compute_stratified_coverage(acts_clean, thresholds, top_neurons)
            conf_clean = detection_confidence(model, images, device)

            # Object-region perturbation
            u_obj = perturb_region(images, obj_mask, NOISE_STD)
            acts_obj = collector.collect(u_obj)
            cov_obj = compute_stratified_coverage(acts_obj, thresholds, top_neurons)
            mag_obj = compute_magnitude_change(acts_clean, acts_obj, top_neurons)
            conf_obj = detection_confidence(model, u_obj, device)

            # Background-region perturbation
            u_bg = perturb_region(images, bg_mask, NOISE_STD)
            acts_bg = collector.collect(u_bg)
            cov_bg = compute_stratified_coverage(acts_bg, thresholds, top_neurons)
            mag_bg = compute_magnitude_change(acts_clean, acts_bg, top_neurons)
            conf_bg = detection_confidence(model, u_bg, device)

            all_obj_cov.append({k: abs(cov_obj[k] - cov_clean[k]) for k in cov_clean})
            all_bg_cov.append({k: abs(cov_bg[k] - cov_clean[k]) for k in cov_clean})
            all_obj_mag.append(mag_obj)
            all_bg_mag.append(mag_bg)
            conf_drops_obj.extend((conf_clean - conf_obj).tolist())
            conf_drops_bg.extend((conf_clean - conf_bg).tolist())

        row = {"iteration": it}
        for k in ("early", "middle", "late", "overall", "variability"):
            row[f"obj_cov_{k}"] = float(np.mean([d[k] for d in all_obj_cov]))
            row[f"bg_cov_{k}"] = float(np.mean([d[k] for d in all_bg_cov]))
            row[f"obj_mag_{k}"] = float(np.mean([d[k] for d in all_obj_mag]))
            row[f"bg_mag_{k}"] = float(np.mean([d[k] for d in all_bg_mag]))
        row["conf_drop_obj"] = float(np.mean(conf_drops_obj))
        row["conf_drop_bg"] = float(np.mean(conf_drops_bg))
        row["mean_obj_frac"] = float(np.mean(obj_fracs))
        records.append(row)

        # Per-pixel normalised ratio: magnitude per perturbed pixel
        of = max(row["mean_obj_frac"], 1e-3)
        bf = max(1.0 - of, 1e-3)
        norm_obj = row["obj_mag_overall"] / of
        norm_bg = row["bg_mag_overall"] / bf
        r_norm = norm_obj / max(norm_bg, 1e-8)
        print(f"  Iter {it}: ObjMag={row['obj_mag_overall']:.4f} BgMag={row['bg_mag_overall']:.4f} "
              f"ObjFrac={of:.2f} NormRatio={r_norm:.3f} "
              f"ConfDrop: obj={row['conf_drop_obj']:.4f} bg={row['conf_drop_bg']:.4f}")

    collector.detach()

    df = pd.DataFrame(records)
    csv_out = f"{out_prefix}_coverage.csv"
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    df.to_csv(csv_out, index=False)

    # Summary
    print("\n" + "=" * 70)
    print("RQ2 OPTIMISED – Object vs Background Perturbation")
    print("=" * 70)
    mean_of = df["mean_obj_frac"].mean()
    mean_bf = max(1.0 - mean_of, 1e-3)

    print(f"  Mean object fraction: {mean_of:.2%}")

    # PRIMARY METRIC: Coverage change (follows original WISDOM methodology)
    print()
    print("  [Coverage change — PRIMARY METRIC (threshold-based neuron activation)]")
    print("  (Fraction of WISDOM neurons that cross calibrated activation threshold)")
    for grp in ("early", "middle", "late", "overall"):
        co = df[f"obj_cov_{grp}"].mean()
        cb = df[f"bg_cov_{grp}"].mean()
        r = co / max(cb, 1e-8)
        print(f"    {grp:>8s}:  Obj_Δcov={co:.4f}  Bg_Δcov={cb:.4f}  Ratio={r:.3f} {'✅' if r > 1.0 else '⚠️'}")
    # Coverage variability
    cv_o = df["obj_cov_variability"].mean()
    cv_b = df["bg_cov_variability"].mean()
    print(f"    {'var':>8s}:  Obj_Δvar={cv_o:.4f}  Bg_Δvar={cv_b:.4f}  Ratio={cv_o / max(cv_b, 1e-8):.3f}")

    # SUPPLEMENTARY: Magnitude change (continuous, captures sub-threshold shifts)
    print()
    print("  [Magnitude change — SUPPLEMENTARY (continuous activation shift)]")
    for grp in ("early", "middle", "late", "overall"):
        mo = df[f"obj_mag_{grp}"].mean()
        mb = df[f"bg_mag_{grp}"].mean()
        r = mo / max(mb, 1e-8)
        print(f"    {grp:>8s}:  Obj={mo:.4f}  Bg={mb:.4f}  Ratio={r:.3f} {'✅' if r > 1.0 else '⚠️'}")

    # SUPPLEMENTARY: Per-pixel normalised magnitude
    print()
    print("  [Per-pixel normalised magnitude — SUPPLEMENTARY]")
    for grp in ("early", "middle", "late", "overall"):
        mo = df[f"obj_mag_{grp}"].mean() / max(mean_of, 1e-3)
        mb = df[f"bg_mag_{grp}"].mean() / mean_bf
        r = mo / max(mb, 1e-8)
        print(f"    {grp:>8s}:  Obj/px={mo:.4f}  Bg/px={mb:.4f}  Ratio={r:.3f} {'✅' if r > 1.0 else '⚠️'}")

    # VALIDATION: Detection confidence drop (output-level confirmation)
    print()
    cd_o = df["conf_drop_obj"].mean()
    cd_b = df["conf_drop_bg"].mean()
    print(f"  [Detection confidence drop — VALIDATION (model output)]")
    print(f"    Obj perturbation: {cd_o:.4f}")
    print(f"    Bg  perturbation: {cd_b:.4f}")
    print(f"    Ratio (Obj/Bg):   {cd_o / max(cd_b, 1e-8):.3f} {'✅' if cd_o > cd_b else '⚠️'}")

    log_path = "logs/rq2_opt_results.log"
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
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-prefix", default="results/rq2_opt")
    p.add_argument("--per-layer-k", type=int, default=5)
    p.add_argument("--num-iters", type=int, default=3)
    a = p.parse_args()
    run_rq2_opt(**vars(a))

#!/usr/bin/env python
"""
sanity_check_rq2_opt.py – Visual sanity check matching run_rq2_opt.py settings
===============================================================================
Produces a multi-panel plot from a single COCO image showing:

  Row 1: Original | Object mask | Importance heatmap
  Row 2: Importance-guided top-2% pixels (per-group colors) on object
  Row 3: Importance-guided top-2% pixels (per-group colors) on background
  Row 4: Random 2% pixels (per-group colors) on object
  Row 5: Random 2% pixels (per-group colors) on background

Per-group color coding for perturbed pixels:
  ● Red   = early layers   (model.0–5)
  ● Blue  = middle layers  (model.6–12)
  ● Green = late layers    (model.13–22)

Perturbation settings match run_rq2_opt.py defaults:
  PIXEL_FRAC = 0.02  (top 2% of pixels)
  NOISE_STD  = 0.30  (Gaussian noise σ)

Supports both --neuron-select per-layer and per-group.

Usage:
    python optimize/sanity_check_rq2_opt.py \\
        --weights weights/yolo11n.pt \\
        --out-dir results/rq2_sanity_opt
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimize.coverage_utils import (
    load_layerwise_top_neurons,
    load_groupwise_top_neurons,
    _layer_group,
)

PIXEL_FRAC = 0.02
NOISE_STD = 0.30

GROUP_COLORS = {
    "early":  (255, 50,  50),   # red
    "middle": (50,  100, 255),  # blue
    "late":   (50,  220, 50),   # green
}


# ── Image loading ────────────────────────────────────────────────────

def load_single_image(img_dir: str, imgsz: int = 320, index: int | None = None):
    paths = sorted(Path(img_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg files in {img_dir}")
    if index is None:
        index = random.randint(0, len(paths) - 1)
    p = paths[index % len(paths)]
    pil = Image.open(p).convert("RGB")
    tensor = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])(pil).unsqueeze(0)
    return tensor, pil, p.name


# ── Object mask via YOLO detection ───────────────────────────────────

def get_object_mask(model, image_tensor, device, imgsz, conf_thresh=0.25):
    with torch.no_grad():
        out = model(image_tensor.to(device))
        preds = out[0] if isinstance(out, (tuple, list)) else out
    mask = torch.zeros(1, 1, imgsz, imgsz, device=device)
    cls_max = preds[0, 4:, :].max(dim=0).values
    confident = cls_max > conf_thresh
    if confident.sum() > 0:
        boxes = preds[0, :4, confident]
        for j in range(boxes.shape[1]):
            cx, cy, w, h = boxes[:, j]
            w, h = w * 1.3, h * 1.3
            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = min(imgsz, int(cx + w / 2))
            y2 = min(imgsz, int(cy + h / 2))
            mask[0, 0, y1:y2, x1:x2] = 1.0
    return mask


# ── Per-group importance gradient ────────────────────────────────────

def importance_gradient_per_group(model, image_tensor, top_neurons, device):
    """Compute separate gradient maps for each layer group (early/middle/late).

    Returns {group_name: gradient_tensor (1,3,H,W)}.
    """
    group_neurons = {"early": {}, "middle": {}, "late": {}}
    for lname, idxs in top_neurons.items():
        grp = _layer_group(lname)
        group_neurons[grp][lname] = idxs

    gradients = {}
    for grp_name, neurons in group_neurons.items():
        if not neurons:
            continue
        x = image_tensor.clone().to(device).requires_grad_(True)
        acts = {}
        handles = []
        for name, mod in model.named_modules():
            if name in neurons:
                indices = neurons[name]
                def _hook(layer_name, idxs):
                    def fn(module, inp, out):
                        if out.dim() == 4:
                            acts[layer_name] = out[:, idxs, :, :].mean(dim=(2, 3)).sum()
                        elif out.dim() == 2:
                            acts[layer_name] = out[:, idxs].sum()
                    return fn
                handles.append(mod.register_forward_hook(_hook(name, indices)))

        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        model(x)
        total = sum(acts.values())
        total.backward()
        for h in handles:
            h.remove()
        gradients[grp_name] = x.grad.detach().cpu().abs()

    return gradients


# ── Perturbation ─────────────────────────────────────────────────────

def select_top_pixels(importance_map, region_mask, frac=PIXEL_FRAC):
    """Select top-frac% pixels within region_mask by importance.

    Returns flat indices into the (H,W) space.
    """
    H, W = importance_map.shape
    region_idx = torch.nonzero(region_mask.view(-1) > 0.5, as_tuple=True)[0]
    if len(region_idx) == 0:
        return torch.tensor([], dtype=torch.long)
    k = max(1, int(len(region_idx) * frac))
    imp_in_region = importance_map.view(-1)[region_idx]
    _, topk = imp_in_region.topk(min(k, len(region_idx)))
    return region_idx[topk]


def select_random_pixels(region_mask, frac=PIXEL_FRAC):
    """Select random frac% pixels within region_mask.

    Returns flat indices into the (H,W) space.
    """
    region_idx = torch.nonzero(region_mask.view(-1) > 0.5, as_tuple=True)[0]
    if len(region_idx) == 0:
        return torch.tensor([], dtype=torch.long)
    k = max(1, int(len(region_idx) * frac))
    perm = torch.randperm(len(region_idx))[:k]
    return region_idx[perm]


def apply_noise(image_tensor, pixel_indices, W, std=NOISE_STD):
    """Apply Gaussian noise to specific pixels. Returns perturbed image."""
    perturbed = image_tensor.clone()
    if len(pixel_indices) == 0:
        return perturbed
    rows = pixel_indices // W
    cols = pixel_indices % W
    C = perturbed.shape[1]
    perturbed[0, :, rows, cols] += torch.randn(C, len(pixel_indices)) * std
    return perturbed.clamp(0, 1)


# ── Visualization ────────────────────────────────────────────────────

def tensor_to_numpy(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay_pixels_by_group(image_tensor, group_pixel_indices, W, alpha=0.85):
    """Draw colored dots on image for each group's selected pixels.

    Returns numpy array (H,W,3).
    """
    arr = tensor_to_numpy(image_tensor).astype(np.float32)
    for grp_name, indices in group_pixel_indices.items():
        if len(indices) == 0:
            continue
        color = np.array(GROUP_COLORS[grp_name], dtype=np.float32)
        rows = (indices // W).numpy()
        cols = (indices % W).numpy()
        arr[rows, cols, :] = arr[rows, cols, :] * (1 - alpha) + color * alpha
    return arr.astype(np.uint8)


def overlay_perturbed_diff(original, perturbed, group_pixel_indices, W):
    """Show perturbed image with colored outlines indicating which group
    contributed each pixel perturbation.

    Returns numpy array (H,W,3).
    """
    arr = tensor_to_numpy(perturbed).astype(np.float32)
    for grp_name, indices in group_pixel_indices.items():
        if len(indices) == 0:
            continue
        color = np.array(GROUP_COLORS[grp_name], dtype=np.float32)
        rows = (indices // W).numpy()
        cols = (indices % W).numpy()
        # Draw a subtle colored border around perturbed pixels
        arr[rows, cols, :] = arr[rows, cols, :] * 0.4 + color * 0.6
    return arr.astype(np.uint8)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Visual sanity check for RQ2 (matching run_rq2_opt.py settings)")
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file",
                   default="neuron_eval_out/wisdom_yolo11n_scores_5000.csv")
    p.add_argument("--out-dir", default="results/rq2_sanity_opt")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--per-layer-k", type=int, default=5)
    p.add_argument("--pixel-frac", type=float, default=PIXEL_FRAC)
    p.add_argument("--noise-std", type=float, default=NOISE_STD)
    p.add_argument("--neuron-select", choices=["per-layer", "per-group"],
                   default="per-layer")
    p.add_argument("--image-index", type=int, default=None)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    # Lazy import to avoid slow load when just checking --help
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from ultralytics import YOLO
    yolo = YOLO(a.weights)
    model = yolo.model.eval().to(a.device)

    if a.neuron_select == "per-group":
        top_neurons = load_groupwise_top_neurons(a.csv_file, per_group_k=a.per_layer_k)
    else:
        top_neurons = load_layerwise_top_neurons(a.csv_file, per_layer_k=a.per_layer_k)
    total_n = sum(len(v) for v in top_neurons.values())
    print(f"Neuron selection: {a.neuron_select}")
    print(f"Monitoring {total_n} neurons across {len(top_neurons)} layers")
    print(f"Perturbation: top {a.pixel_frac*100:.0f}% pixels, noise σ={a.noise_std}")

    # Find image with detections
    for attempt in range(20):
        idx = a.image_index if a.image_index is not None else random.randint(0, 4999)
        img_tensor, pil_img, fname = load_single_image(a.img_dir, a.imgsz, idx)
        obj_mask = get_object_mask(model, img_tensor, a.device, a.imgsz)
        obj_frac = obj_mask.mean().item()
        if obj_frac > 0.05:
            break
        if a.image_index is not None:
            print(f"Warning: image {fname} has only {obj_frac:.1%} object coverage")
            break
        print(f"  Skipping {fname} (obj {obj_frac:.1%}), trying another...")
    else:
        print("Could not find image with sufficient detections")
        return

    bg_mask = 1.0 - obj_mask
    H, W = a.imgsz, a.imgsz
    print(f"Image: {fname} (object fraction: {obj_frac:.1%})")

    # ── Compute per-group gradients ──────────────────────────────────
    print("Computing per-group importance gradients...")
    group_grads = importance_gradient_per_group(
        model, img_tensor, top_neurons, a.device)

    # ── Select pixels per group ──────────────────────────────────────
    # For importance: each group contributes its own top-frac% pixels
    # For random: each group gets a random frac% (different random sets)

    def select_importance_pixels_per_group(region_mask_2d):
        result = {}
        for grp_name, grad in group_grads.items():
            imp_map = grad[0].mean(dim=0)  # (H, W)
            result[grp_name] = select_top_pixels(
                imp_map, region_mask_2d, frac=a.pixel_frac)
        return result

    def select_random_pixels_per_group(region_mask_2d):
        result = {}
        for grp_name in group_grads:
            result[grp_name] = select_random_pixels(
                region_mask_2d, frac=a.pixel_frac)
        return result

    obj_mask_2d = obj_mask[0, 0].cpu()
    bg_mask_2d = bg_mask[0, 0].cpu()

    imp_obj_pixels = select_importance_pixels_per_group(obj_mask_2d)
    imp_bg_pixels = select_importance_pixels_per_group(bg_mask_2d)
    rand_obj_pixels = select_random_pixels_per_group(obj_mask_2d)
    rand_bg_pixels = select_random_pixels_per_group(bg_mask_2d)

    # ── Apply perturbations ──────────────────────────────────────────
    def apply_all_groups(img, group_pixels):
        result = img.clone()
        for grp_name, indices in group_pixels.items():
            result = apply_noise(result, indices, W, std=a.noise_std)
        return result

    imp_obj_img = apply_all_groups(img_tensor, imp_obj_pixels)
    imp_bg_img = apply_all_groups(img_tensor, imp_bg_pixels)
    rand_obj_img = apply_all_groups(img_tensor, rand_obj_pixels)
    rand_bg_img = apply_all_groups(img_tensor, rand_bg_pixels)

    # ── Build figure ─────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 3, figsize=(15, 24))
    fig.suptitle(
        f"RQ2 Sanity Check — {fname}\n"
        f"Selection: {a.neuron_select} | {total_n} neurons | "
        f"top {a.pixel_frac*100:.0f}% pixels | σ={a.noise_std}",
        fontsize=14, fontweight="bold", y=0.98)

    legend_handles = [
        Patch(facecolor=np.array(GROUP_COLORS["early"]) / 255, label="Early (model.0–5)"),
        Patch(facecolor=np.array(GROUP_COLORS["middle"]) / 255, label="Middle (model.6–12)"),
        Patch(facecolor=np.array(GROUP_COLORS["late"]) / 255, label="Late (model.13–22)"),
    ]

    def show(ax, img_arr, title):
        ax.imshow(img_arr)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # Row 0: Original, Object Mask, Combined Importance Heatmap
    show(axes[0, 0], tensor_to_numpy(img_tensor), "Original")

    mask_vis = tensor_to_numpy(img_tensor).astype(np.float32)
    m = obj_mask[0, 0].cpu().numpy()
    mask_vis[m > 0.5] = mask_vis[m > 0.5] * 0.5 + np.array([255, 100, 100]) * 0.5
    show(axes[0, 1], mask_vis.astype(np.uint8), "Object Mask (red)")

    # Combined gradient heatmap colored by group
    heatmap = np.zeros((H, W, 3), dtype=np.float32)
    for grp_name, grad in group_grads.items():
        gm = grad[0].mean(dim=0).numpy()
        gm = (gm - gm.min()) / (gm.max() - gm.min() + 1e-8)
        color = np.array(GROUP_COLORS[grp_name], dtype=np.float32) / 255.0
        for c in range(3):
            heatmap[:, :, c] += gm * color[c]
    heatmap = np.clip(heatmap / max(heatmap.max(), 1e-8), 0, 1)
    show(axes[0, 2], (heatmap * 255).astype(np.uint8),
         "Importance Heatmap (per-group)")

    # Row 1: Importance on Object
    show(axes[1, 0],
         overlay_pixels_by_group(img_tensor, imp_obj_pixels, W),
         "Imp. Pixels — Object")
    show(axes[1, 1],
         tensor_to_numpy(imp_obj_img),
         "Perturbed — Imp. Object")
    diff = (imp_obj_img - img_tensor).abs()
    diff_vis = (diff[0].permute(1, 2, 0).numpy() * 10).clip(0, 1)
    show(axes[1, 2], (diff_vis * 255).astype(np.uint8),
         "Diff ×10 — Imp. Object")

    # Row 2: Importance on Background
    show(axes[2, 0],
         overlay_pixels_by_group(img_tensor, imp_bg_pixels, W),
         "Imp. Pixels — Background")
    show(axes[2, 1],
         tensor_to_numpy(imp_bg_img),
         "Perturbed — Imp. Background")
    diff = (imp_bg_img - img_tensor).abs()
    diff_vis = (diff[0].permute(1, 2, 0).numpy() * 10).clip(0, 1)
    show(axes[2, 2], (diff_vis * 255).astype(np.uint8),
         "Diff ×10 — Imp. Background")

    # Row 3: Random on Object
    show(axes[3, 0],
         overlay_pixels_by_group(img_tensor, rand_obj_pixels, W),
         "Random Pixels — Object")
    show(axes[3, 1],
         tensor_to_numpy(rand_obj_img),
         "Perturbed — Rand. Object")
    diff = (rand_obj_img - img_tensor).abs()
    diff_vis = (diff[0].permute(1, 2, 0).numpy() * 10).clip(0, 1)
    show(axes[3, 2], (diff_vis * 255).astype(np.uint8),
         "Diff ×10 — Rand. Object")

    # Row 4: Random on Background
    show(axes[4, 0],
         overlay_pixels_by_group(img_tensor, rand_bg_pixels, W),
         "Random Pixels — Background")
    show(axes[4, 1],
         tensor_to_numpy(rand_bg_img),
         "Perturbed — Rand. Background")
    diff = (rand_bg_img - img_tensor).abs()
    diff_vis = (diff[0].permute(1, 2, 0).numpy() * 10).clip(0, 1)
    show(axes[4, 2], (diff_vis * 255).astype(np.uint8),
         "Diff ×10 — Rand. Background")

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, fontsize=12, frameon=True,
               bbox_to_anchor=(0.5, 0.005))
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plot_path = os.path.join(a.out_dir, "sanity_check_rq2_opt.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {plot_path}")

    # ── Print per-group pixel stats ──────────────────────────────────
    print(f"\n{'Setting':<30s}  {'Early':>8s} {'Middle':>8s} {'Late':>8s} {'Total':>8s}")
    print("-" * 70)
    for label, gpx in [
        ("Imp. Object", imp_obj_pixels),
        ("Imp. Background", imp_bg_pixels),
        ("Rand. Object", rand_obj_pixels),
        ("Rand. Background", rand_bg_pixels),
    ]:
        counts = {g: len(gpx.get(g, [])) for g in ("early", "middle", "late")}
        total = sum(counts.values())
        print(f"  {label:<28s}  {counts['early']:>8d} {counts['middle']:>8d} "
              f"{counts['late']:>8d} {total:>8d}")

    # Per-group L2 diffs
    print(f"\n{'Perturbation':<30s}  {'L2 diff':>10s} {'Max diff':>10s}")
    print("-" * 52)
    for label, pert_img in [
        ("Importance Object", imp_obj_img),
        ("Importance Background", imp_bg_img),
        ("Random Object", rand_obj_img),
        ("Random Background", rand_bg_img),
    ]:
        diff = (pert_img - img_tensor).abs()
        print(f"  {label:<28s}  {diff.norm().item():>10.4f} {diff.max().item():>10.4f}")

    print(f"\nAll outputs saved to {a.out_dir}/")


if __name__ == "__main__":
    main()

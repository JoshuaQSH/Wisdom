#!/usr/bin/env python
"""
run_rq2_opt.py – RQ2: Input Diversity via Importance-Guided Perturbation
=========================================================================

Follows the original WISDOM RQ2 methodology with union-based coverage:

For each image in the test set, two perturbed variants are generated:
  U_I — Gaussian noise on the **top 2 %** most important pixels (gradient)
  U_R — Gaussian noise on a **random 2 %** of pixels

Coverage is measured as **per-image union** coverage:
  C(img_i)              — coverage from the single clean image
  C(img_i ∪ img_i^I)    — union coverage from clean + importance-perturbed
  C(img_i ∪ img_i^R)    — union coverage from clean + random-perturbed

Per-image coverage gain:
  ΔC_I = C(img_i ∪ img_i^I) − C(img_i)  (averaged over all images)
  ΔC_R = C(img_i ∪ img_i^R) − C(img_i)

Success criterion:
  ΔC_I > ΔC_R  ⇒  importance-guided perturbation contributes **more** novel
  neuron coverage, confirming WISDOM identifies behaviourally meaningful pixels.

Enhanced spatial decomposition (object vs background):
  D_I_obj / D_R_obj — perturbation restricted to object bounding-box pixels
  D_I_bg  / D_R_bg  — perturbation restricted to background pixels

Supports both "plain" (threshold) and "cluster" (combinatorial) coverage.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wisdom_yolo_train import COCOImageDataset, _collate
from optimize.coverage_utils import (
    load_layerwise_top_neurons,
    load_groupwise_top_neurons,
    load_important_layer_neurons,
    load_hybrid_layer_neurons,
    ActivationCollector,
    calibrate_thresholds,
    compute_magnitude_change,
    ClusterCoverageComputer,
    UnionCoverageTracker,
    ClusterUnionTracker,
    verbose_coverage_breakdown,
    verbose_union_diff,
    _build_group_combo_sets,
    get_group_names,
    _layer_group,
)

# ── Constants (matching original WISDOM RQ2) ───────────────────────
PIXEL_FRAC = 0.02    # 2 % of pixels perturbed
NOISE_STD  = 0.30    # Gaussian noise std


# ── Pixel importance ───────────────────────────────────────────────
def pixel_importance_output(model, images, device):
    """Per-pixel importance via gradient of model output (detection preds).

    Spread across all detection anchors — may dilute signal for YOLO.
    """
    model.eval()
    x = images.to(device).clone().requires_grad_(True)
    out = model(x)
    target = out[0] if isinstance(out, (tuple, list)) else out
    target.sum().backward()
    grad = x.grad.abs()
    return grad.mean(dim=1).detach().cpu()


def pixel_importance_wisdom(model, images, device, top_neurons, _batch_size=2):
    """Per-pixel importance via gradient of WISDOM neuron activations.

    Computes which pixels most influence the *monitored* neurons —
    a direct measure of WISDOM-relevant input sensitivity.
    Processes in mini-batches to avoid OOM on large models.
    """
    model.eval()
    all_grads = []
    for start in range(0, len(images), _batch_size):
        batch = images[start:start + _batch_size]
        acts_live = {}
        hooks = []
        for lname, idxs in top_neurons.items():
            mod = dict(model.named_modules()).get(lname)
            if mod is None:
                continue
            idx_t = torch.tensor(idxs, dtype=torch.long, device=device)

            def _hook(name, idx):
                def fn(_, __, out):
                    if out.dim() == 4:
                        acts_live[name] = out[:, idx].mean(dim=(2, 3))
                    elif out.dim() == 2:
                        acts_live[name] = out[:, idx]
                return fn

            h = mod.register_forward_hook(_hook(lname, idx_t))
            hooks.append(h)

        x = batch.to(device).clone().requires_grad_(True)
        model(x)

        total = sum(a.sum() for a in acts_live.values())
        total.backward()

        for h in hooks:
            h.remove()

        all_grads.append(x.grad.abs().mean(dim=1).detach().cpu())
        del x, total, acts_live
        torch.cuda.empty_cache()

    return torch.cat(all_grads, dim=0)


def pixel_importance_bpw(model, images, device, top_neurons, cluster_comp):
    """Boundary-Proximity Weighted: gradient weighted by inverse boundary distance.

    For each monitored neuron, we compute its boundary distance (distance from
    current activation to nearest cluster boundary). Neurons near boundaries
    get higher weight, so the importance map focuses perturbation on pixels
    that influence near-boundary neurons — the ones most likely to produce
    new coverage combinations.
    """
    model.eval()
    acts_live = {}
    hooks = []
    for lname, idxs in top_neurons.items():
        mod = dict(model.named_modules()).get(lname)
        if mod is None:
            continue
        idx_t = torch.tensor(idxs, dtype=torch.long, device=device)

        def _hook(name, idx):
            def fn(_, __, out):
                if out.dim() == 4:
                    acts_live[name] = out[:, idx].mean(dim=(2, 3))
                elif out.dim() == 2:
                    acts_live[name] = out[:, idx]
            return fn

        h = mod.register_forward_hook(_hook(lname, idx_t))
        hooks.append(h)

    x = images.to(device).clone().requires_grad_(True)
    model(x)

    # Compute boundary-proximity weights and weighted activation sum
    total = torch.zeros(1, device=device, requires_grad=True)
    for lname, idxs in top_neurons.items():
        if lname not in acts_live:
            continue
        acts = acts_live[lname]  # (B, n_neurons)
        for col, idx in enumerate(idxs):
            if lname not in cluster_comp.groups or idx not in cluster_comp.groups[lname]:
                continue
            centers = cluster_comp.groups[lname][idx]["centers"]  # (C, 1)
            centers_sorted = np.sort(centers.squeeze())
            if len(centers_sorted) < 2:
                continue
            # Boundaries are midpoints between consecutive sorted centroids
            boundaries = torch.tensor(
                [(centers_sorted[i] + centers_sorted[i + 1]) / 2
                 for i in range(len(centers_sorted) - 1)],
                device=device, dtype=acts.dtype,
            )
            act_vals = acts[:, col]  # (B,)
            # Distance to nearest boundary (detached — used only as weight)
            dists = (act_vals.detach().unsqueeze(1) - boundaries.unsqueeze(0)).abs()
            min_dist = dists.min(dim=1).values  # (B,)
            weight = 1.0 / (min_dist + 0.01)  # inverse boundary distance
            total = total + (act_vals * weight).sum()

    total.backward()

    for h in hooks:
        h.remove()

    grad = x.grad.abs()
    return grad.mean(dim=1).detach().cpu()


def pixel_importance_bga(model, images, device, top_neurons, cluster_comp,
                        _batch_size=2):
    """Boundary-Gradient Attribution: gradient of cluster boundary distance.

    Uses the sum of per-neuron boundary distances as the loss.
    The gradient tells us which pixels most influence how far each neuron's
    activation is from its nearest cluster boundary. Processes in mini-batches
    to avoid OOM on large models.
    """
    model.eval()
    all_grads = []
    for start in range(0, len(images), _batch_size):
        batch = images[start:start + _batch_size]
        acts_live = {}
        hooks = []
        for lname, idxs in top_neurons.items():
            mod = dict(model.named_modules()).get(lname)
            if mod is None:
                continue
            idx_t = torch.tensor(idxs, dtype=torch.long, device=device)

            def _hook(name, idx):
                def fn(_, __, out):
                    if out.dim() == 4:
                        acts_live[name] = out[:, idx].mean(dim=(2, 3))
                    elif out.dim() == 2:
                        acts_live[name] = out[:, idx]
                return fn

            h = mod.register_forward_hook(_hook(lname, idx_t))
            hooks.append(h)

        x = batch.to(device).clone().requires_grad_(True)
        model(x)

        boundary_loss = torch.zeros(1, device=device, requires_grad=True)
        for lname, idxs in top_neurons.items():
            if lname not in acts_live:
                continue
            acts = acts_live[lname]
            for col, idx in enumerate(idxs):
                if lname not in cluster_comp.groups or idx not in cluster_comp.groups[lname]:
                    continue
                centers = cluster_comp.groups[lname][idx]["centers"]
                centers_sorted = np.sort(centers.squeeze())
                if len(centers_sorted) < 2:
                    continue
                boundaries = torch.tensor(
                    [(centers_sorted[i] + centers_sorted[i + 1]) / 2
                     for i in range(len(centers_sorted) - 1)],
                    device=device, dtype=acts.dtype,
                )
                act_vals = acts[:, col]
                dists = (act_vals.unsqueeze(1) - boundaries.unsqueeze(0)).abs()
                min_dist = dists.min(dim=1).values
                boundary_loss = boundary_loss + min_dist.sum()

        boundary_loss.backward()

        for h in hooks:
            h.remove()

        all_grads.append(x.grad.abs().mean(dim=1).detach().cpu())
        del x, boundary_loss, acts_live
        torch.cuda.empty_cache()

    return torch.cat(all_grads, dim=0)


def pixel_importance_lrp(model, images, device, top_neurons):
    """Pixel importance via LRP relevance propagation from monitored neurons."""
    from optimize.lrp_scorer import compute_lrp_pixel_importance
    return compute_lrp_pixel_importance(model, images, device, top_neurons)


def pixel_importance_gradient(model, images, device, top_neurons=None,
                              mode="wisdom", cluster_comp=None):
    """Dispatcher: choose importance method."""
    if mode in ("bpw", "bga") and cluster_comp is None:
        raise ValueError(f"importance mode '{mode}' requires cluster_comp (use --coverage-mode cluster)")
    if mode == "bpw" and top_neurons is not None:
        return pixel_importance_bpw(model, images, device, top_neurons, cluster_comp)
    if mode == "bga" and top_neurons is not None:
        return pixel_importance_bga(model, images, device, top_neurons, cluster_comp)
    if mode == "lrp" and top_neurons is not None:
        return pixel_importance_lrp(model, images, device, top_neurons)
    if mode == "wisdom" and top_neurons is not None:
        return pixel_importance_wisdom(model, images, device, top_neurons)
    return pixel_importance_output(model, images, device)


# ── Object mask ────────────────────────────────────────────────────
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
            w, h = w * 1.3, h * 1.3
            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = min(imgsz, int(cx + w / 2))
            y2 = min(imgsz, int(cy + h / 2))
            masks[b, 0, y1:y2, x1:x2] = 1.0
    return masks


# ── Perturbation functions ─────────────────────────────────────────
def perturb_important_pixels(images, importance, frac=PIXEL_FRAC, std=NOISE_STD):
    """Gaussian noise on top-frac% pixels by gradient importance."""
    B, C, H, W = images.shape
    k = max(1, int(H * W * frac))
    perturbed = images.clone()
    for i in range(B):
        _, topk_idx = importance[i].view(-1).topk(k)
        rows, cols = topk_idx // W, topk_idx % W
        perturbed[i, :, rows, cols] += torch.randn(C, k) * std
    return perturbed.clamp(0, 1)


def perturb_random_pixels(images, importance, frac=PIXEL_FRAC, std=NOISE_STD):
    """Gaussian noise on random frac% pixels, excluding importance-chosen ones."""
    B, C, H, W = images.shape
    k = max(1, int(H * W * frac))
    perturbed = images.clone()
    for i in range(B):
        # Exclude the top-k important pixels so random never overlaps importance
        _, topk_imp = importance[i].view(-1).topk(k)
        exclude = set(topk_imp.tolist())
        all_idx = [j for j in range(H * W) if j not in exclude]
        perm = torch.randperm(len(all_idx))[:k]
        idx = torch.tensor([all_idx[p] for p in perm], dtype=torch.long)
        rows, cols = idx // W, idx % W
        perturbed[i, :, rows, cols] += torch.randn(C, k) * std
    return perturbed.clamp(0, 1)


def perturb_important_in_region(images, importance, mask, frac=PIXEL_FRAC, std=NOISE_STD):
    """Gaussian noise on top-frac% important pixels WITHIN mask region.

    ``frac`` is relative to the *region* size so that perturbation density
    is comparable across regions of different sizes.
    """
    B, C, H, W = images.shape
    perturbed = images.clone()
    for i in range(B):
        region_idx = torch.nonzero(mask[i, 0].view(-1) > 0.5, as_tuple=True)[0]
        if len(region_idx) == 0:
            continue
        k = max(1, int(len(region_idx) * frac))
        imp_in_region = importance[i].view(-1)[region_idx]
        _, topk = imp_in_region.topk(min(k, len(region_idx)))
        sel = region_idx[topk]
        rows, cols = sel // W, sel % W
        perturbed[i, :, rows, cols] += torch.randn(C, len(sel)) * std
    return perturbed.clamp(0, 1)


def perturb_random_in_region(images, importance, mask, frac=PIXEL_FRAC, std=NOISE_STD):
    """Gaussian noise on random frac% pixels WITHIN mask region,
    excluding importance-chosen pixels in that region."""
    B, C, H, W = images.shape
    perturbed = images.clone()
    for i in range(B):
        region_idx = torch.nonzero(mask[i, 0].view(-1) > 0.5, as_tuple=True)[0]
        if len(region_idx) == 0:
            continue
        k = max(1, int(len(region_idx) * frac))
        # Exclude the top-k important pixels within this region
        imp_in_region = importance[i].view(-1)[region_idx]
        _, topk = imp_in_region.topk(min(k, len(region_idx)))
        exclude = set(topk.tolist())  # indices into region_idx
        eligible = [j for j in range(len(region_idx)) if j not in exclude]
        if len(eligible) == 0:
            continue
        perm = torch.randperm(len(eligible))[:k]
        sel = region_idx[torch.tensor([eligible[p] for p in perm], dtype=torch.long)]
        rows, cols = sel // W, sel % W
        perturbed[i, :, rows, cols] += torch.randn(C, len(sel)) * std
    return perturbed.clamp(0, 1)


# ── Variant helpers ────────────────────────────────────────────────
VARIANTS = {
    "I":     "Full-image importance-guided",
    "R":     "Full-image random",
    "I_obj": "Object-region importance-guided",
    "R_obj": "Object-region random",
    "I_bg":  "Background-region importance-guided",
    "R_bg":  "Background-region random",
}


def generate_variant(name, images, importance, obj_mask, frac, std):
    """Generate a perturbed image batch for the given variant."""
    bg_mask = 1.0 - obj_mask
    if name == "I":
        return perturb_important_pixels(images, importance, frac, std)
    elif name == "R":
        return perturb_random_pixels(images, importance, frac, std)
    elif name == "I_obj":
        return perturb_important_in_region(images, importance, obj_mask, frac, std)
    elif name == "R_obj":
        return perturb_random_in_region(images, importance, obj_mask, frac, std)
    elif name == "I_bg":
        return perturb_important_in_region(images, importance, bg_mask, frac, std)
    elif name == "R_bg":
        return perturb_random_in_region(images, importance, bg_mask, frac, std)
    raise ValueError(f"Unknown variant: {name}")


# ── Signal dilution diagnostic ─────────────────────────────────────

def compute_signal_dilution_diagnostic(
    model,
    sample_images,
    top_neurons,
    cluster_comp,
    device,
    pixel_frac=0.02,
    noise_std=0.30,
):
    """Measure signal dilution: pixel perturbation → activation change → cluster crossing.

    For each monitored neuron quantifies:
      - mean |Δact_imp|  : activation shift when *important* pixels are perturbed
      - mean |Δact_rnd|  : activation shift when *random* pixels are perturbed
      - mean_boundary_dist: distance from clean activation to nearest cluster boundary
      - crossing_rate_imp : fraction of images where important perturbation crosses a boundary
      - crossing_rate_rnd : fraction of images where random perturbation crosses a boundary

    A crossing_rate << 1 combined with boundary_dist >> mean_delta is direct evidence
    of signal dilution: pixel perturbations are too small relative to the cluster boundary
    distances to reliably trigger state changes.
    """
    import numpy as np
    import logging
    log = logging.getLogger("wisdom_opt")

    model.eval()

    # Build per-neuron boundary midpoints from fitted cluster centers
    boundaries: dict = {}
    for lname, neuron_dict in cluster_comp.groups.items():
        boundaries[lname] = {}
        for idx, info in neuron_dict.items():
            centers = info["centers"].flatten()
            centers_sorted = np.sort(centers)
            midpoints = (centers_sorted[:-1] + centers_sorted[1:]) / 2.0
            boundaries[lname][idx] = midpoints

    collector = ActivationCollector(model, top_neurons, device)
    collector.attach()

    # Clean activations
    acts_clean: dict = {}
    for i in range(0, len(sample_images), 4):
        batch = sample_images[i:i + 4]
        a = collector.collect(batch)
        for k, v in a.items():
            acts_clean.setdefault(k, []).append(v.cpu())
    for k in acts_clean:
        acts_clean[k] = torch.cat(acts_clean[k], dim=0)  # (N, n_neurons)
    collector.detach()

    # Pixel importance on the sample
    imp = pixel_importance_gradient(model, sample_images, device, top_neurons)

    # Perturbed images
    pert_imp = perturb_important_pixels(sample_images, imp, pixel_frac, noise_std)
    pert_rnd = perturb_random_pixels(sample_images, imp, pixel_frac, noise_std)

    collector.attach()
    acts_imp: dict = {}
    acts_rnd: dict = {}
    for i in range(0, len(sample_images), 4):
        sl = slice(i, i + 4)
        ai = collector.collect(pert_imp[sl])
        for k, v in ai.items():
            acts_imp.setdefault(k, []).append(v.cpu())
        ar = collector.collect(pert_rnd[sl])
        for k, v in ar.items():
            acts_rnd.setdefault(k, []).append(v.cpu())
    for k in acts_imp:
        acts_imp[k] = torch.cat(acts_imp[k], dim=0)
    for k in acts_rnd:
        acts_rnd[k] = torch.cat(acts_rnd[k], dim=0)
    collector.detach()

    # Per-neuron stats
    rows = []
    for lname in top_neurons:
        if lname not in acts_clean or lname not in acts_imp:
            continue
        ac = acts_clean[lname].numpy()   # (N, n_neurons)
        ai = acts_imp[lname].numpy()
        ar = acts_rnd[lname].numpy()
        idxs = top_neurons[lname]
        for col, nidx in enumerate(idxs):
            cv = ac[:, col]
            di = np.abs(ai[:, col] - cv)
            dr = np.abs(ar[:, col] - cv)
            bds = boundaries.get(lname, {}).get(nidx, np.array([]))
            if len(bds) > 0:
                bdist = np.min(np.abs(cv[:, None] - bds[None, :]), axis=1)
                cross_i = float(np.mean(di > bdist))
                cross_r = float(np.mean(dr > bdist))
                mean_bd = float(np.mean(bdist))
            else:
                cross_i = cross_r = mean_bd = float("nan")
            rows.append({
                "layer": lname, "neuron": nidx,
                "mean_delta_imp": float(np.mean(di)),
                "mean_delta_rnd": float(np.mean(dr)),
                "mean_boundary_dist": mean_bd,
                "crossing_rate_imp": cross_i,
                "crossing_rate_rnd": cross_r,
            })

    if not rows:
        log.info("  [DIAG] No diagnostic data collected.")
        return rows

    deltas_i = np.array([r["mean_delta_imp"] for r in rows])
    deltas_r = np.array([r["mean_delta_rnd"] for r in rows])
    bdists   = np.array([r["mean_boundary_dist"] for r in rows if np.isfinite(r["mean_boundary_dist"])])
    cr_i     = np.array([r["crossing_rate_imp"] for r in rows if np.isfinite(r["crossing_rate_imp"])])
    cr_r     = np.array([r["crossing_rate_rnd"] for r in rows if np.isfinite(r["crossing_rate_rnd"])])

    ratio_bd_delta = np.mean(bdists) / max(np.mean(deltas_i), 1e-9)

    log.info("\n  ╔══════════════════════════════════════════════════════╗")
    log.info("  ║        SIGNAL DILUTION DIAGNOSTIC                    ║")
    log.info(f"  ║  Sample: {len(sample_images)} images | {len(rows)} neurons monitored")
    log.info(f"  ║  Pixel frac: {pixel_frac*100:.1f}%  Noise std: {noise_std}")
    log.info("  ╠══════════════════════════════════════════════════════╣")
    log.info(f"  ║  Mean |Δact| important pixels: {np.mean(deltas_i):.6f}  (std={np.std(deltas_i):.6f})")
    log.info(f"  ║  Mean |Δact| random pixels:    {np.mean(deltas_r):.6f}  (std={np.std(deltas_r):.6f})")
    log.info(f"  ║  Mean boundary distance:        {np.mean(bdists):.6f}  (std={np.std(bdists):.6f})")
    log.info(f"  ║  Boundary / delta ratio (imp):  {ratio_bd_delta:.1f}x")
    log.info(f"  ║  → Boundaries are {ratio_bd_delta:.1f}x farther than perturbation effect")
    log.info(f"  ║  Boundary crossing rate (imp):  {np.mean(cr_i)*100:.2f}%  of (image,neuron) pairs")
    log.info(f"  ║  Boundary crossing rate (rnd):  {np.mean(cr_r)*100:.2f}%")
    log.info(f"  ║  imp/rnd crossing ratio:         {np.mean(cr_i)/max(np.mean(cr_r),1e-9):.3f}")

    # Percentile breakdown
    pcts = np.percentile(deltas_i, [25, 50, 75, 90, 99])
    log.info(f"  ║  |Δact_imp| percentiles (p25/50/75/90/99): "
             f"{pcts[0]:.5f} / {pcts[1]:.5f} / {pcts[2]:.5f} / {pcts[3]:.5f} / {pcts[4]:.5f}")
    pcts_bd = np.percentile(bdists, [25, 50, 75])
    log.info(f"  ║  boundary_dist percentiles (p25/50/75):    "
             f"{pcts_bd[0]:.5f} / {pcts_bd[1]:.5f} / {pcts_bd[2]:.5f}")

    # Neurons that actually cross boundaries
    n_cross_any_i = sum(1 for r in rows if r["crossing_rate_imp"] > 0)
    n_cross_any_r = sum(1 for r in rows if r["crossing_rate_rnd"] > 0)
    log.info(f"  ║  Neurons ever crossing a boundary (imp):   {n_cross_any_i}/{len(rows)}")
    log.info(f"  ║  Neurons ever crossing a boundary (rnd):   {n_cross_any_r}/{len(rows)}")
    log.info("  ╚══════════════════════════════════════════════════════╝\n")

    return rows


# ── Combo overlap analysis ─────────────────────────────────────────

def verbose_combo_overlap(
    baseline_tracker,
    trackers_by_variant: dict,
    combo_mode: str = "per-layer",
):
    """Log the overlap between new combos contributed by importance (I) vs random (R).

    Shows for each group/layer:
      - new_I_only : combos new in I but not in R
      - new_R_only : combos new in R but not in I
      - new_both   : combos new in BOTH I and R
      - jaccard    : |I_new ∩ R_new| / |I_new ∪ R_new|

    A high Jaccard index means I and R are triggering largely the same new combos,
    suggesting no semantic advantage of importance selection.
    """
    from optimize.coverage_utils import _build_group_combo_sets, get_group_names
    from wisdom.core.compute import combinations_coverage
    import logging
    log = logging.getLogger("wisdom_opt")

    I_tracker = trackers_by_variant.get("I")
    R_tracker = trackers_by_variant.get("R")
    if I_tracker is None or R_tracker is None:
        return

    log.info("\n  ── Combo Overlap Analysis: I vs R ──────────────────────")

    if combo_mode == "per-group":
        base_data  = _build_group_combo_sets(baseline_tracker)
        I_data     = _build_group_combo_sets(I_tracker)
        R_data     = _build_group_combo_sets(R_tracker)
        total_I_only = total_R_only = total_both = 0
        for g in get_group_names():
            base_set = base_data[g]["seen_set"]
            I_new = I_data[g]["seen_set"] - base_set
            R_new = R_data[g]["seen_set"] - base_set
            only_I = I_new - R_new
            only_R = R_new - I_new
            both   = I_new & R_new
            union_ = I_new | R_new
            jac = len(both) / max(len(union_), 1)
            total_I_only += len(only_I)
            total_R_only += len(only_R)
            total_both   += len(both)
            log.info(
                f"  [OVERLAP] group={g}: I_new={len(I_new)} R_new={len(R_new)} "
                f"| I_only={len(only_I)} R_only={len(only_R)} "
                f"shared={len(both)} | Jaccard={jac:.3f}"
            )
        total_union = total_I_only + total_R_only + total_both
        jac_total = total_both / max(total_union, 1)
        log.info(
            f"  [OVERLAP] TOTAL: I_only={total_I_only} R_only={total_R_only} "
            f"shared={total_both} | Jaccard={jac_total:.3f}"
        )
    else:
        # Per-layer
        from optimize.coverage_utils import _layer_group
        total_I_only = total_R_only = total_both = 0
        layer_stats = []
        for lname in baseline_tracker.comp.target_neurons:
            b_assigns = baseline_tracker.layer_assignments.get(lname, [])
            b_sizes   = baseline_tracker.layer_sizes.get(lname, {})
            if not (b_assigns and b_sizes):
                continue
            keys = sorted(b_sizes.keys())
            base_set = set(
                tuple(a.get(k, -1) for k in keys)
                for a in b_assigns
                if -1 not in [a.get(k, -1) for k in keys]
            )
            def _make_set(tracker):
                assigns = tracker.layer_assignments.get(lname, [])
                return set(
                    tuple(a.get(k, -1) for k in keys)
                    for a in assigns
                    if -1 not in [a.get(k, -1) for k in keys]
                )
            I_set  = _make_set(I_tracker)
            R_set  = _make_set(R_tracker)
            I_new  = I_set - base_set
            R_new  = R_set - base_set
            only_I = I_new - R_new
            only_R = R_new - I_new
            both   = I_new & R_new
            union_ = I_new | R_new
            jac    = len(both) / max(len(union_), 1)
            total_I_only += len(only_I)
            total_R_only += len(only_R)
            total_both   += len(both)
            layer_stats.append((lname, len(I_new), len(R_new), len(only_I), len(only_R), len(both), jac))
        # Summary only (per-layer is too verbose individually)
        log.info(
            f"  [OVERLAP] PER-LAYER TOTAL: I_only={total_I_only} "
            f"R_only={total_R_only} shared={total_both}"
        )
        total_union = total_I_only + total_R_only + total_both
        jac_total = total_both / max(total_union, 1)
        log.info(f"  [OVERLAP] Overall Jaccard={jac_total:.3f}")
        # Top layers where I uniquely wins
        I_unique_layers = sorted(layer_stats, key=lambda x: x[3], reverse=True)[:10]
        log.info("  [OVERLAP] Top-10 layers where I adds unique combos:")
        for lname, ni, nr, oi, or_, b, j in I_unique_layers:
            log.info(f"    {lname}: I_new={ni} R_new={nr} I_only={oi} R_only={or_} shared={b}")


# ── Main experiment ────────────────────────────────────────────────
def run_rq2_opt(
    weights,
    img_dir,
    csv_file,
    num_images=200,
    batch_size=2,
    imgsz=320,
    device="cuda:0",
    out_prefix="results/rq2_opt",
    per_layer_k=5,
    num_iters=3,
    coverage_mode="plain",
    pixel_frac=PIXEL_FRAC,
    noise_std=NOISE_STD,
    importance="wisdom",
    n_clusters=3,
    neuron_select="per-layer",
    n_groups=3,
    verbose=False,
    log_file=None,
    diag_n_sample=50,
    top_layers=10,
    layer_score_method="mean_positive",
    neuron_csv=None,
    custom_groups=None,
):
    from ultralytics import YOLO
    from optimize.coverage_utils import set_n_groups, set_custom_groups

    if custom_groups:
        set_custom_groups(custom_groups)
    else:
        set_n_groups(n_groups)

    yolo = YOLO(weights)
    model = yolo.model.eval().to(device)

    if neuron_select == "per-group":
        top_neurons = load_groupwise_top_neurons(csv_file, per_group_k=per_layer_k)
    elif neuron_select == "important-layer":
        top_neurons = load_important_layer_neurons(
            csv_file,
            top_layers=top_layers,
            per_layer_k=per_layer_k,
            layer_score_method=layer_score_method,
        )
    elif neuron_select == "hybrid":
        if neuron_csv is None:
            raise ValueError("--neuron-csv required for hybrid mode")
        top_neurons = load_hybrid_layer_neurons(
            layer_csv=csv_file,
            neuron_csv=neuron_csv,
            top_layers=top_layers,
            per_layer_k=per_layer_k,
            layer_score_method=layer_score_method,
        )
    else:
        top_neurons = load_layerwise_top_neurons(csv_file, per_layer_k=per_layer_k)
    total_n = sum(len(v) for v in top_neurons.values())
    print(f"Monitoring {total_n} neurons across {len(top_neurons)} layers")
    print(f"Neuron selection: {neuron_select}")
    if neuron_select in ("important-layer", "hybrid"):
        print(f"Layer scoring: {layer_score_method}, top {top_layers} layers")
    if neuron_select == "hybrid":
        print(f"Neuron CSV: {neuron_csv}")
    print(f"Coverage mode: {coverage_mode}")
    print(f"Perturbation: {pixel_frac*100:.1f}% pixels, noise std={noise_std}")
    print(f"Importance mode: {importance}")

    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    # Calibrate coverage infrastructure
    calib_imgs = torch.stack([ds[i][0] for i in range(min(50, len(ds)))])

    cluster_comp = None
    thresholds = None
    if coverage_mode == "cluster":
        # When user specifies n_clusters, disable silhouette auto-selection
        # so every neuron gets exactly n_clusters clusters.
        use_sil = (n_clusters is None)
        nc = n_clusters or 3
        cluster_comp = ClusterCoverageComputer(
            model, top_neurons, device=device,
            method="KMeans", use_silhouette=use_sil,
            k_max=max(nc + 2, 5), n_clusters=nc,
            combo_mode="per-layer" if neuron_select in ("important-layer", "hybrid") else neuron_select,
        )
        print(f"Fitting clusters (n_clusters={nc}, silhouette={'on' if use_sil else 'off'})...")
        cluster_comp.fit(calib_imgs, batch_size=batch_size)
        # Report actual cluster sizes
        sizes = list(cluster_comp.cluster_sizes.values())
        print(f"Clusters fitted: {len(sizes)} neurons, "
              f"k range [{min(sizes)}-{max(sizes)}], "
              f"mean k={np.mean(sizes):.1f}")
    else:
        # p90 threshold: ~10% neurons active per image → 90% headroom
        # High threshold ensures only strong perturbations cross →
        # concentrated importance noise can outperform diffuse random noise.
        thresholds = calibrate_thresholds(
            model, top_neurons, calib_imgs, device, percentile=90.0,
        )
        print("Thresholds calibrated (p90 — high headroom for per-image union)")

    # ── Phase 1: Preprocess — activations, importance, masks ───────
    print("\nPhase 1: Preprocessing images...")
    collector = ActivationCollector(model, top_neurons, device)
    collector.attach()

    all_images = []
    all_importance = []
    all_obj_masks = []
    all_clean_acts = []

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

        if (batch_idx + 1) % 25 == 0:
            print(f"  Processed {(batch_idx+1)*batch_size}/{len(ds)} images")

    print(f"  Preprocessed {len(ds)} images total")
    collector.detach()

    # ── Phase 2: Union coverage ───────────────────────────────────────
    # Plain mode  → per-image union  (dataset-level saturates at 100 %)
    # Cluster mode → dataset-level union (combinatorial space has headroom)
    records = []
    mag_records = []  # activation magnitude change (supplementary)
    for it in range(num_iters):
        print(f"\n--- Iteration {it+1}/{num_iters} ---")

        collector.attach()

        if coverage_mode == "cluster":
            # ── Dataset-level union (cluster) ──────────────────────
            baseline = ClusterUnionTracker(cluster_comp)
            for acts in all_clean_acts:
                baseline.update_from_activations(acts)
            C_O = baseline.coverage()
            gnames = list(C_O.keys())
            gnames = [g for g in gnames if g not in ("overall", "variability")]
            grp_str = " ".join(f"{g[0].upper()}={C_O[g]:.4f}" for g in gnames)
            print(f"  C(D_O): overall={C_O['overall']:.4f} [{grp_str}]")
            if verbose:
                verbose_coverage_breakdown(baseline, label="D_O")

            # Run signal dilution diagnostic once (first iteration)
            if verbose and it == 0 and coverage_mode == "cluster" and diag_n_sample > 0:
                diag_imgs = torch.cat(all_images[:max(1, diag_n_sample // batch_size)], dim=0)
                diag_imgs = diag_imgs[:diag_n_sample]
                compute_signal_dilution_diagnostic(
                    model, diag_imgs, top_neurons, cluster_comp, device,
                    pixel_frac=pixel_frac, noise_std=noise_std,
                )

            variant_trackers = {}
            for vname in VARIANTS:
                print(f"  [variant] {vname}...", flush=True)
                tracker = baseline.clone()
                batch_mags = []
                for bi, (images, imp, mask, cacts) in enumerate(zip(
                    all_images, all_importance, all_obj_masks, all_clean_acts,
                )):
                    perturbed = generate_variant(
                        vname, images, imp, mask, pixel_frac, noise_std,
                    )
                    acts = collector.collect(perturbed)
                    acts_cpu = {k: v.cpu() for k, v in acts.items()}
                    tracker.update_from_activations(acts_cpu)
                    batch_mags.append(
                        compute_magnitude_change(cacts, acts_cpu, top_neurons)
                    )
                    if (bi + 1) % 500 == 0:
                        print(f"    {bi+1}/{len(all_images)} batches", flush=True)
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
                    "variant": vname, "coverage_mode": coverage_mode,
                }
                for k in list(gnames) + ["overall"]:
                    row[f"C_O_{k}"] = C_O[k]
                    row[f"C_union_{k}"] = C_union[k]
                    row[f"delta_{k}"] = C_union[k] - C_O[k]
                    row[f"mag_{k}"] = avg_mag[k]
                records.append(row)
                print(f"  C(D_O∪D_{vname:>5s}): overall={C_union['overall']:.4f}"
                      f"  Δ={row['delta_overall']:+.6f}"
                      f"  mag={avg_mag['overall']:.4f}")

            # Combo overlap analysis after all variants
            if verbose:
                verbose_combo_overlap(baseline, variant_trackers,
                                      combo_mode="per-layer" if neuron_select in ("important-layer", "hybrid") else neuron_select)

        else:
            # ── Per-image union (plain) ────────────────────────────
            img_count = 0
            for batch_idx, (images, imp, mask) in enumerate(
                zip(all_images, all_importance, all_obj_masks)
            ):
                B = images.shape[0]
                clean_acts = all_clean_acts[batch_idx]

                variant_acts = {}
                for vname in VARIANTS:
                    perturbed = generate_variant(
                        vname, images, imp, mask, pixel_frac, noise_std,
                    )
                    variant_acts[vname] = {
                        k: v.cpu()
                        for k, v in collector.collect(perturbed).items()
                    }

                for i in range(B):
                    img_clean = {l: clean_acts[l][i:i+1] for l in clean_acts}
                    base = UnionCoverageTracker(thresholds, top_neurons)
                    base.update(img_clean)
                    C_clean = base.coverage()

                    for vname in VARIANTS:
                        img_pert = {
                            l: variant_acts[vname][l][i:i+1]
                            for l in variant_acts[vname]
                        }
                        tr = base.clone()
                        tr.update(img_pert)
                        C_union = tr.coverage()

                        mag = compute_magnitude_change(
                            img_clean, img_pert, top_neurons,
                        )
                        row = {
                            "iteration": it, "scope": "per-image",
                            "image_idx": img_count + i,
                            "variant": vname, "coverage_mode": coverage_mode,
                        }
                        for k in list(C_clean.keys()):
                            row[f"C_clean_{k}"] = C_clean[k]
                            row[f"C_union_{k}"] = C_union[k]
                            row[f"delta_{k}"] = C_union[k] - C_clean[k]
                            row[f"mag_{k}"] = mag[k]
                        records.append(row)

                img_count += B
                if (batch_idx + 1) % 25 == 0:
                    print(f"  Processed {img_count}/{len(ds)} images")

        collector.detach()

        # Per-iteration summary
        it_df = pd.DataFrame([r for r in records if r["iteration"] == it])
        for vname in ("I", "R"):
            d = it_df[it_df["variant"] == vname]["delta_overall"].mean()
            print(f"  Mean ΔC({vname}): {d:+.6f}")

    # ── Phase 3: Analysis ──────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_out = f"{out_prefix}_union_coverage.csv"
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    df.to_csv(csv_out, index=False)

    _print_summary(df, coverage_mode, pixel_frac, noise_std, len(ds), num_iters)

    log_path = "logs/rq2_opt_results.log"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        f.write(df.to_string(index=False) + "\n")
    print(f"\n  CSV → {csv_out}   Log → {log_path}")
    return csv_out


# ── Pretty summary ─────────────────────────────────────────────────
def _print_summary(df, coverage_mode, pixel_frac, noise_std, n_images, n_iters):
    print("\n" + "=" * 72)
    print("RQ2: Input Diversity via Importance-Guided Pixel Perturbation")
    print("=" * 72)
    print(f"  Coverage mode : {coverage_mode}")
    print(f"  Perturbation  : {pixel_frac*100:.1f}% pixels, noise std={noise_std}")
    print(f"  Images: {n_images} | Iterations: {n_iters}")

    C_base_col = "C_O_overall" if "C_O_overall" in df.columns else "C_clean_overall"
    C_base = df[C_base_col].mean()
    scope = df["scope"].iloc[0] if "scope" in df.columns else "per-image"
    print(f"\n  Union scope   : {scope}")
    print(f"  Mean baseline : {C_base:.4f}")

    def _section(title, imp_var, rand_var):
        di = df[df["variant"] == imp_var]["delta_overall"].mean()
        dr = df[df["variant"] == rand_var]["delta_overall"].mean()
        r = di / max(dr, 1e-8)
        print(f"\n  --- {title} ---")
        print(f"  ΔC({imp_var:>5s}) = {di:+.6f}  (importance-guided)")
        print(f"  ΔC({rand_var:>5s}) = {dr:+.6f}  (random)")
        print(f"  Ratio ΔI/ΔR = {r:.3f} {'✅' if r > 1.0 else '⚠️'}")
        for grp in [c.replace("delta_", "") for c in df.columns if c.startswith("delta_") and c != "delta_overall"]:
            dgi = df[df["variant"] == imp_var][f"delta_{grp}"].mean()
            dgr = df[df["variant"] == rand_var][f"delta_{grp}"].mean()
            rg = dgi / max(dgr, 1e-8)
            print(f"    {grp:>8s}: ΔI={dgi:+.6f}  ΔR={dgr:+.6f}  "
                  f"Ratio={rg:.3f} {'✅' if rg > 1.0 else '⚠️'}")
        return r

    r_classic = _section("Classic WISDOM (Full Image)", "I", "R")
    r_obj     = _section("Object Region", "I_obj", "R_obj")
    r_bg      = _section("Background Region", "I_bg", "R_bg")

    di_obj = df[df["variant"] == "I_obj"]["delta_overall"].mean()
    di_bg  = df[df["variant"] == "I_bg"]["delta_overall"].mean()
    print(f"\n  --- Importance: Object vs Background ---")
    print(f"  ΔC(I_obj) / ΔC(I_bg) = "
          f"{di_obj / max(di_bg, 1e-8):.3f} "
          f"{'✅ (obj > bg)' if di_obj > di_bg else '⚠️ (bg ≥ obj)'}")

    # Activation magnitude change (supplementary metric)
    if "mag_overall" in df.columns:
        print(f"\n  --- Activation Magnitude Change (supplementary) ---")
        for imp_v, rand_v, label in [
            ("I", "R", "Full Image"),
            ("I_obj", "R_obj", "Object Region"),
            ("I_bg", "R_bg", "Background"),
        ]:
            mi = df[df["variant"] == imp_v]["mag_overall"].mean()
            mr = df[df["variant"] == rand_v]["mag_overall"].mean()
            r_m = mi / max(mr, 1e-8)
            print(f"  {label:>15s}:  Mag(I)={mi:.4f}  Mag(R)={mr:.4f}  "
                  f"Ratio={r_m:.3f} {'✅' if r_m > 1.0 else '⚠️'}")
            for grp in [c.replace("mag_", "") for c in df.columns if c.startswith("mag_") and c != "mag_overall"]:
                mgi = df[df["variant"] == imp_v][f"mag_{grp}"].mean()
                mgr = df[df["variant"] == rand_v][f"mag_{grp}"].mean()
                rg = mgi / max(mgr, 1e-8)
                print(f"    {grp:>8s}:  Mag(I)={mgi:.4f}  Mag(R)={mgr:.4f}  "
                      f"Ratio={rg:.3f} {'✅' if rg > 1.0 else '⚠️'}")

    passes = sum([r_classic > 1.0, r_obj > 1.0])
    print(f"\n  {'=' * 60}")
    if passes >= 2:
        print("  ✅ WISDOM importance-guided perturbation produces more diverse")
        print("     neuron coverage than random perturbation.")
    elif passes >= 1:
        print("  ⚠️ Partial: importance outperforms random in some settings.")
    else:
        print("  ❌ Importance-guided perturbation does NOT outperform random.")
    print(f"  {'=' * 60}")


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="RQ2: Input diversity via importance-guided perturbation",
    )
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file",
                   default="neuron_eval_out/wisdom_yolo11n_scores_5000.csv")
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-prefix", default="results/rq2_opt")
    p.add_argument("--per-layer-k", type=int, default=5)
    p.add_argument("--num-iters", type=int, default=3)
    p.add_argument("--coverage-mode", choices=["plain", "cluster"],
                   default="plain")
    p.add_argument("--pixel-frac", type=float, default=PIXEL_FRAC)
    p.add_argument("--noise-std", type=float, default=NOISE_STD)
    p.add_argument("--importance", choices=["wisdom", "output", "bpw", "bga", "lrp"],
                   default="wisdom",
                   help="Pixel importance method: 'wisdom' (gradient of "
                        "WISDOM neuron activations), 'output' (gradient of "
                        "model detection output), 'bpw' (boundary-proximity "
                        "weighted), 'bga' (boundary-gradient attribution), "
                        "'lrp' (LRP relevance propagation from monitored neurons). "
                        "bpw/bga require --coverage-mode cluster.")
    p.add_argument("--n-clusters", type=int, default=3,
                   help="Number of KMeans clusters per neuron (default: 3). "
                        "Controls the combinatorial space: with k neurons per "
                        "layer, total combinations = n_clusters^k.")
    p.add_argument("--neuron-select", choices=["per-layer", "per-group", "important-layer", "hybrid"],
                   default="per-layer",
                   help="Neuron selection mode: 'per-layer' selects top-k "
                        "from each Conv2d layer independently (~60 tiny combo "
                        "spaces). 'per-group' selects top-k from each layer "
                        "group, yielding larger cross-layer combo spaces. "
                        "'important-layer' first ranks layers by aggregate "
                        "importance, selects top-L layers, then picks top-k "
                        "neurons per layer. 'hybrid' uses --csv-file for layer "
                        "ranking and --neuron-csv for neuron ranking within "
                        "selected layers.")
    p.add_argument("--neuron-csv", default=None,
                   help="Neuron-level scoring CSV for hybrid mode (e.g. lgxa scores)")
    p.add_argument("--n-groups", type=int, default=3,
                   choices=[2, 3, 4, 5],
                   help="Number of layer groups (default: 3)")
    p.add_argument("--custom-groups", type=str, default=None,
                   help="Custom asymmetric layer groups as 'name:lo-hi,...'. "
                        "Example: 'early:0-3,middle:4-15,late:16-22'. "
                        "Overrides --n-groups if provided.")
    p.add_argument("--top-layers", type=int, default=10,
                   help="Number of top layers to select (for 'important-layer' mode, default: 10)")
    p.add_argument("--layer-score-method", choices=["mean_positive", "sum", "max"],
                   default="mean_positive",
                   help="How to score layers: mean of positive neuron scores, "
                        "sum of all scores, or max score (default: mean_positive)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable per-layer combo count logging")
    p.add_argument("--log-file", default=None,
                   help="Write verbose log output to this file (default: stderr)")
    p.add_argument("--diag-n-sample", type=int, default=50,
                   help="Number of images for signal dilution diagnostic (default: 50)")
    a = p.parse_args()

    if a.verbose:
        import logging
        handlers = []
        log_fmt = "%(name)s %(message)s"
        if a.log_file:
            import os
            os.makedirs(os.path.dirname(a.log_file) if os.path.dirname(a.log_file) else ".", exist_ok=True)
            handlers.append(logging.FileHandler(a.log_file, mode="w"))
        else:
            handlers.append(logging.StreamHandler())
        logging.basicConfig(
            level=logging.INFO,
            format=log_fmt,
            handlers=handlers,
        )

    # Parse custom groups if provided
    custom_groups_dict = None
    if a.custom_groups:
        custom_groups_dict = {}
        for part in a.custom_groups.split(","):
            name, rng = part.strip().split(":")
            lo, hi = rng.split("-")
            custom_groups_dict[name.strip()] = (int(lo), int(hi))
        a.n_groups = len(custom_groups_dict)

    run_rq2_opt(**{k: v for k, v in vars(a).items()
                   if k not in ("log_file", "diag_n_sample", "custom_groups")},
               custom_groups=custom_groups_dict,
               log_file=getattr(a, "log_file", None),
               diag_n_sample=getattr(a, "diag_n_sample", 50))

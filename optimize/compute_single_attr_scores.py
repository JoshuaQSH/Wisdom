#!/usr/bin/env python
"""
compute_single_attr_scores.py — Single-Method Attribution Scoring for YOLOv11n
===============================================================================

Bypasses the full WISDOM consensus voting pipeline (which uses 3 methods +
pruning loss) to produce neuron importance CSVs from a single Captum
attribution method.  Much faster and avoids the pruning step that may not
suit YOLO-like architectures.

Supported methods:
  lgxa  — LayerGradientXActivation (fast, reliable)
  lig   — LayerIntegratedGradients (accurate, slower)
  lgs   — LayerGradientShap (accurate, slower)
  la    — LayerActivation (no gradient; magnitude-only baseline)

Output CSV format matches WISDOM: LayerName, NeuronIndex, Score
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wisdom.utils.yolo_wrapper import YOLOWrapper


def _is_conv(m: nn.Module) -> bool:
    return isinstance(m, nn.Conv2d)


def compute_single_attr(
    wrapper: YOLOWrapper,
    images: torch.Tensor,
    method: str,
    device: str,
    exclude_detect_head: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute per-neuron attribution scores using a single Captum method."""
    from captum.attr import (
        LayerGradientXActivation,
        LayerIntegratedGradients,
        LayerGradientShap,
        LayerActivation,
    )

    name2ctor = {
        "lgxa": LayerGradientXActivation,
        "lig": LayerIntegratedGradients,
        "lgs": LayerGradientShap,
        "la": LayerActivation,
    }
    key = method.lower()
    if key not in name2ctor:
        raise ValueError(f"Unknown method '{method}'. Options: {list(name2ctor.keys())}")

    wrapper = wrapper.to(device).eval()
    images = images.to(device)
    target = torch.zeros(images.size(0), dtype=torch.long, device=device)

    out: Dict[str, torch.Tensor] = {}
    for lname, layer in wrapper.named_modules():
        if not _is_conv(layer):
            continue
        if exclude_detect_head and "model.23." in lname:
            continue
        A = name2ctor[key](wrapper, layer)
        if key == "la":
            attr = A.attribute(images)
        elif key == "lgs":
            attr = A.attribute(images, baselines=torch.zeros_like(images), target=target)
        else:
            attr = A.attribute(images, target=target)

        # Aggregate: sum over batch and spatial dims → per-channel score
        # Match WISDOM convention: raw sum (no abs), so only positive attributions
        # contribute to high importance scores (consistent with nlargest ranking)
        if attr.dim() == 4:
            vec = attr.sum(dim=(0, 2, 3)).detach().cpu()
        else:
            vec = attr.sum(dim=0).detach().cpu()
        out[lname] = vec

    return out


def main():
    p = argparse.ArgumentParser(
        description="Single-method attribution scoring for YOLOv11n neurons",
    )
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--method", choices=["lgxa", "lig", "lgs", "la"], default="lgxa",
                   help="Attribution method (default: lgxa = GradientXActivation)")
    p.add_argument("--num-images", type=int, default=500,
                   help="Number of images for scoring (default: 500)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-csv", default=None,
                   help="Output CSV path (default: neuron_eval_out/yolo11n_{method}_{N}.csv)")
    a = p.parse_args()

    from ultralytics import YOLO
    from wisdom_yolo_train import COCOImageDataset, _collate
    from torch.utils.data import DataLoader

    if a.out_csv is None:
        a.out_csv = f"neuron_eval_out/yolo11n_{a.method}_{a.num_images}.csv"
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)

    print(f"Method: {a.method} | Images: {a.num_images} | Output: {a.out_csv}")

    yolo = YOLO(a.weights)
    model = yolo.model.eval().to(a.device)
    wrapper = YOLOWrapper(model, num_classes=80)

    ds = COCOImageDataset(a.img_dir, max_images=a.num_images, imgsz=a.imgsz)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=_collate)

    # Accumulate scores across batches
    accumulated: Dict[str, torch.Tensor] = {}
    t0 = time.time()
    for i, batch in enumerate(loader):
        images = batch[0].to(a.device)
        scores = compute_single_attr(wrapper, images, a.method, a.device)
        for lname, vec in scores.items():
            if lname not in accumulated:
                accumulated[lname] = torch.zeros_like(vec)
            accumulated[lname] += vec
        if (i + 1) % 25 == 0 or (i + 1) == len(loader):
            elapsed = time.time() - t0
            imgs_done = min((i + 1) * a.batch_size, len(ds))
            print(f"  Processed {imgs_done}/{len(ds)} images ({elapsed:.1f}s)")

    # Save CSV
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["LayerName", "NeuronIndex", "Score"])
        for lname in sorted(accumulated.keys()):
            t = accumulated[lname]
            for idx, score in enumerate(t.tolist()):
                w.writerow([lname, idx, score])

    elapsed = time.time() - t0
    n_layers = len(accumulated)
    n_neurons = sum(t.numel() for t in accumulated.values())
    n_positive = sum((t > 0).sum().item() for t in accumulated.values())
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Saved {n_neurons} neurons across {n_layers} layers to {a.out_csv}")
    print(f"Neurons with Score > 0: {n_positive}/{n_neurons}")

    # Show top-15 layers by mean score
    layer_means = []
    for lname, t in accumulated.items():
        pos = t[t > 0]
        if len(pos) > 0:
            layer_means.append((lname, float(pos.mean()), len(pos), t.numel()))
    layer_means.sort(key=lambda x: -x[1])
    print(f"\nTop-15 layers (by mean_positive score):")
    for i, (l, s, np_, nt) in enumerate(layer_means[:15]):
        print(f"  {i+1:3d}. {l:45s} score={s:12.1f}  pos={np_:4d}/{nt}")


if __name__ == "__main__":
    main()

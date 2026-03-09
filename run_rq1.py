#!/usr/bin/env python
"""
run_rq1.py – RQ1: Do WISDOM-identified neurons matter?
=======================================================
Prunes the top-N neurons (N ∈ {6, 8, 10, 15, 20}) and measures the
detection-performance drop on a validation subset of COCO.

For each attribution method (LGXA, IntegratedGradients, GradientShap)
and the WISDOM consensus:
  1. Compute per-layer importance scores on a small training subset.
  2. Select the top-N neurons globally.
  3. Zero their weights/biases and evaluate detection performance.
  4. Record the performance drop relative to the unpruned baseline.

A random pruning baseline is included for comparison.

Outputs: rq1_relevance.csv, rq1_acc_drop.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wisdom_yolo_train import COCOImageDataset, _collate
from wisdom.utils.yolo_wrapper import YOLOWrapper
from wisdom.core.wisdom_train import (
    _is_trainable_module,
    _compute_yolo_importance,
    _gradient_importance,
)

# ── Config ─────────────────────────────────────────────────────────
ATTRIBUTION_METHODS = {"lgxa": "GradXAct", "lig": "IntegGrad", "lgs": "GradShap"}
N_LIST = [6, 8, 10, 15, 20]


# ── YOLO evaluation helper ────────────────────────────────────────
def eval_yolo_confidence(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Return mean sum-of-class-confidence across batches (higher = better)."""
    model.eval().to(device)
    total_conf = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            out = model(images.to(device))
            preds = out[0] if isinstance(out, (tuple, list)) else out
            cls_scores = preds[:, 4:, :]  # (B, nc, A)
            total_conf += cls_scores.sum().item()
            n_batches += 1
    return total_conf / max(n_batches, 1)


def eval_yolo_map(model_wrapper, img_dir: str, device: str, max_images: int = 50) -> float:
    """
    Evaluate mAP using Ultralytics validation pipeline.
    Falls back to confidence-based metric if val fails.
    """
    try:
        from ultralytics import YOLO
        import tempfile

        # Save the current model weights temporarily
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(model_wrapper.state_dict(), tmp_path)
            yolo = YOLO(tmp_path)
            results = yolo.val(
                data=os.path.join(ROOT, "standalone", "data", "coco128.yaml"),
                imgsz=320,
                batch=4,
                verbose=False,
            )
            return float(results.box.map50)
        finally:
            os.unlink(tmp_path)
    except Exception:
        return -1.0  # signal to use confidence instead


# ── Neuron selection ───────────────────────────────────────────────
def wisdom_neurons(csv_file: str, top_k: int = 10) -> Dict[str, List[int]]:
    """Read WISDOM CSV and return top-k neurons grouped by layer."""
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by="Score", ascending=False).head(top_k)
    result: Dict[str, List[int]] = {}
    for lname, group in df_sorted.groupby("LayerName"):
        result[lname] = group["NeuronIndex"].tolist()
    return result


def flatten_importance(
    importance: Dict[str, torch.Tensor],
    exclude_layer: str | None = None,
) -> List[Tuple[str, float, int]]:
    """Flatten per-layer scores into sorted (layer, score, idx) list."""
    flat: List[Tuple[str, float, int]] = []
    for lname, scores in importance.items():
        if exclude_layer and lname == exclude_layer:
            continue
        if scores.dim() == 1:
            for idx, s in enumerate(scores):
                flat.append((lname, float(s.item()), idx))
        else:
            mean_scores = scores.mean(dim=tuple(range(1, scores.dim())))
            for idx, s in enumerate(mean_scores):
                flat.append((lname, float(s.item()), idx))
    flat.sort(key=lambda x: abs(x[1]), reverse=True)
    return flat


# ── Pruning ────────────────────────────────────────────────────────
def prune_neurons(model: nn.Module, selection: Dict[str, List[int]]) -> None:
    """Zero weights and biases of selected neurons in-place."""
    name2mod = dict(model.named_modules())
    for lname, idxs in selection.items():
        mod = name2mod.get(lname)
        if mod is None:
            continue
        with torch.no_grad():
            if isinstance(mod, nn.Conv2d):
                for idx in idxs:
                    mod.weight[idx].zero_()
                    if mod.bias is not None:
                        mod.bias[idx].zero_()
            elif isinstance(mod, nn.Linear):
                for idx in idxs:
                    mod.weight[idx].zero_()
                    if mod.bias is not None:
                        mod.bias[idx].zero_()


# ── RQ1 experiment ─────────────────────────────────────────────────
def run_rq1(
    weights: str,
    img_dir: str,
    csv_file: str,
    out_prefix: str = "rq1",
    device: str = "cuda:0",
    num_images: int = 20,
    batch_size: int = 2,
    imgsz: int = 320,
) -> Tuple[str, str]:
    """
    Run RQ1 experiment. Returns (relevance_csv_path, acc_drop_csv_path).
    """
    from ultralytics import YOLO

    yolo = YOLO(weights)
    torch_model = yolo.model.eval()

    # Prepare data
    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    # Baseline performance
    baseline_conf = eval_yolo_confidence(torch_model, loader, device)
    print(f"Baseline confidence: {baseline_conf:.2f}")

    # Get all trainable layer names (exclude final detection head convs)
    trainable = [(n, m) for n, m in torch_model.named_modules() if _is_trainable_module(m)]
    all_neurons = [(n, i) for n, m in trainable
                   for i in range(m.out_channels if isinstance(m, nn.Conv2d) else m.out_features)]

    relevance_records = []
    accuracy_records = []

    # ── WISDOM consensus ──
    print("\n=== WISDOM Consensus ===")
    for n_prune in N_LIST:
        pruned = copy.deepcopy(torch_model)
        top_neurons = wisdom_neurons(csv_file, top_k=n_prune)
        # Map wrapper names back to model names
        mapped: Dict[str, List[int]] = {}
        for lname, idxs in top_neurons.items():
            if lname.startswith("yolo_model."):
                mapped[lname[len("yolo_model."):]] = idxs
            else:
                mapped[lname] = idxs
        prune_neurons(pruned, mapped)
        pruned_conf = eval_yolo_confidence(pruned, loader, device)
        drop = baseline_conf - pruned_conf
        accuracy_records.append({
            "Attribution Method": "Wisdom",
            "Top-N": n_prune,
            "Confidence Drop": drop,
            "Baseline": baseline_conf,
            "Pruned": pruned_conf,
        })
        print(f"  Top-{n_prune}: conf={pruned_conf:.2f}, drop={drop:.2f}")

    # ── Attribution methods ──
    wrapper = YOLOWrapper(torch_model, num_classes=80)
    wrapper.eval().to(device)

    for attr_key, attr_name in ATTRIBUTION_METHODS.items():
        print(f"\n=== {attr_name} ({attr_key}) ===")

        # Compute importance on first batch
        first_batch = next(iter(loader))
        images = first_batch[0]
        importance = _compute_yolo_importance(wrapper, images, attr_key, device, num_classes=80)

        # Save relevance scores
        for lname, scores in importance.items():
            for idx, score in enumerate(scores):
                relevance_records.append({
                    "Attribution Method": attr_name,
                    "Layer Name": lname,
                    "Neuron Index": idx,
                    "Relevance Score": float(score),
                })

        # Flatten and rank
        flat_scores = flatten_importance(importance)
        total_neurons = len(flat_scores)

        # Attribution-guided pruning
        for n_prune in N_LIST:
            n = min(n_prune, total_neurons)
            top_N = flat_scores[:n]
            pruned = copy.deepcopy(torch_model)
            selection: Dict[str, List[int]] = {}
            for lname, _, idx in top_N:
                if lname.startswith("yolo_model."):
                    mapped_name = lname[len("yolo_model."):]
                else:
                    mapped_name = lname
                selection.setdefault(mapped_name, []).append(idx)
            prune_neurons(pruned, selection)
            pruned_conf = eval_yolo_confidence(pruned, loader, device)
            drop = baseline_conf - pruned_conf
            accuracy_records.append({
                "Attribution Method": attr_name,
                "Top-N": n_prune,
                "Confidence Drop": drop,
                "Baseline": baseline_conf,
                "Pruned": pruned_conf,
            })
            print(f"  Top-{n}: conf={pruned_conf:.2f}, drop={drop:.2f}")

        # Random pruning baseline
        print(f"  Random baseline:")
        for n_prune in N_LIST:
            n = min(n_prune, len(all_neurons))
            rand_sample = random.sample(all_neurons, n)
            pruned = copy.deepcopy(torch_model)
            selection = {}
            for lname, idx in rand_sample:
                selection.setdefault(lname, []).append(idx)
            prune_neurons(pruned, selection)
            pruned_conf = eval_yolo_confidence(pruned, loader, device)
            drop = baseline_conf - pruned_conf
            accuracy_records.append({
                "Attribution Method": f"Random ({attr_name})",
                "Top-N": n_prune,
                "Confidence Drop": drop,
                "Baseline": baseline_conf,
                "Pruned": pruned_conf,
            })
            print(f"    Top-{n}: conf={pruned_conf:.2f}, drop={drop:.2f}")

    # Save results
    rel_path = f"{out_prefix}_relevance.csv"
    drop_path = f"{out_prefix}_acc_drop.csv"
    pd.DataFrame(relevance_records).to_csv(rel_path, index=False)
    pd.DataFrame(accuracy_records).to_csv(drop_path, index=False)
    print(f"\nSaved: {rel_path}, {drop_path}")
    return rel_path, drop_path


# ── CLI ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RQ1: Critical neurons evaluation for YOLOv11")
    p.add_argument("--weights", default="standalone/models/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file", default="wisdom_yolo11n_scores.csv", help="WISDOM scores CSV")
    p.add_argument("--out-prefix", default="rq1_yolo11n")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-images", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=320)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_rq1(
        weights=args.weights,
        img_dir=args.img_dir,
        csv_file=args.csv_file,
        out_prefix=args.out_prefix,
        device=args.device,
        num_images=args.num_images,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
    )

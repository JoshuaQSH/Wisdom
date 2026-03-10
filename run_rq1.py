#!/usr/bin/env python
"""
run_rq1.py – RQ1: Do WISDOM-identified neurons matter?
=======================================================
Prunes the top-N neurons (N ∈ {6, 8, 10, 15, 20}) and measures the
detection-performance drop on a validation subset of COCO.

For each attribution method (LGXA, IntegratedGradients, GradientShap)
and the WISDOM consensus:
  1. Compute per-layer importance scores on a small training subset.
  2. Select the top-N neurons globally (excluding detection head model.23.*).
  3. Zero their weights/biases and evaluate detection performance.
  4. Record the performance drop relative to the unpruned baseline.

Metrics tracked:
  - Confidence drop (sum of class scores)
  - Mean IoU between predicted and ground-truth bounding boxes
  - Classification prediction accuracy (fraction of correct class predictions)

A random pruning baseline is included for comparison.

Outputs: rq1_relevance.csv, rq1_acc_drop.csv, rq1_yolo_{version}_acc_drop.pdf
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
DETECT_HEAD_PREFIX = "model.23."


# ── Dataset with labels ───────────────────────────────────────────
class COCOLabeledDataset(Dataset):
    """Loads COCO images + YOLO-format labels for IoU/accuracy evaluation."""

    def __init__(self, img_dir: str, max_images: int = 50, imgsz: int = 320):
        from torchvision import transforms
        self.img_dir = img_dir
        self.imgsz = imgsz
        # Determine label directory (images/val2017 → labels/val2017)
        self.label_dir = img_dir.replace("/images/", "/labels/")
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        all_imgs = sorted(
            f for f in os.listdir(img_dir) if f.lower().endswith(exts)
        )
        # Only keep images that have a corresponding label file
        self.pairs: List[Tuple[str, str]] = []
        for fname in all_imgs:
            label_fname = os.path.splitext(fname)[0] + ".txt"
            label_path = os.path.join(self.label_dir, label_fname)
            if os.path.isfile(label_path):
                self.pairs.append((
                    os.path.join(img_dir, fname),
                    label_path,
                ))
            if len(self.pairs) >= max_images:
                break
        self.transform = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image
        img_path, label_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        gt_boxes = self._parse_labels(label_path)
        return img_tensor, gt_boxes

    @staticmethod
    def _parse_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
        """Parse YOLO label file → list of (class_id, cx, cy, w, h).
        Handles both standard bbox format and segmentation polygon format."""
        boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                if len(coords) == 4:
                    # Standard YOLO format: cx cy w h
                    cx, cy, w, h = coords
                else:
                    # Segmentation polygon: x1 y1 x2 y2 ... → compute bbox
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    cx = (x_min + x_max) / 2
                    cy = (y_min + y_max) / 2
                    w = x_max - x_min
                    h = y_max - y_min
                boxes.append((cls_id, cx, cy, w, h))
        return boxes


def _collate_labeled(batch):
    """Collate images and variable-length label lists."""
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return images, labels


# ── IoU helpers ────────────────────────────────────────────────────
def _xywh_to_xyxy(cx, cy, w, h, img_size=1.0):
    """Convert normalized center-x/y/w/h to pixel-level x1/y1/x2/y2."""
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return x1, y1, x2, y2


def _box_iou(box1, box2):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _nms_predictions(raw_preds: torch.Tensor, conf_thresh: float = 0.25,
                     iou_thresh: float = 0.45, imgsz: int = 320):
    """Simple NMS on raw YOLO output (B, 84, A) → list of (cls, cx, cy, w, h, conf) per image."""
    B = raw_preds.shape[0]
    results = []
    for b in range(B):
        pred = raw_preds[b]  # (84, A)
        boxes_xywh = pred[:4, :]  # (4, A)
        cls_scores = pred[4:, :]  # (nc, A)
        max_scores, cls_ids = cls_scores.max(dim=0)  # (A,)

        mask = max_scores > conf_thresh
        if mask.sum() == 0:
            results.append([])
            continue

        boxes = boxes_xywh[:, mask].T  # (N, 4) in xywh pixel coords
        scores = max_scores[mask]
        classes = cls_ids[mask]

        # Convert xywh to xyxy for NMS
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        from torchvision.ops import nms
        keep = nms(xyxy, scores, iou_thresh)

        dets = []
        for k in keep:
            cx = boxes[k, 0].item() / imgsz
            cy = boxes[k, 1].item() / imgsz
            w = boxes[k, 2].item() / imgsz
            h = boxes[k, 3].item() / imgsz
            dets.append((int(classes[k].item()), cx, cy, w, h, float(scores[k].item())))
        results.append(dets)
    return results


def eval_iou_and_accuracy(
    model: nn.Module, loader: DataLoader, device: str, imgsz: int = 320,
    conf_thresh: float = 0.01,
) -> Tuple[float, float, float]:
    """Evaluate mean IoU, classification accuracy and detection recall.

    Uses a low default conf_thresh (0.01) so that even heavily-pruned
    models still produce some detections, avoiding the cliff-to-zero
    behaviour that makes the metric uninformative.

    Returns (mean_iou, classification_accuracy, detection_recall).
    """
    model.eval().to(device)
    all_ious = []
    correct_cls = 0
    total_matched = 0
    total_gt = 0

    with torch.no_grad():
        for images, gt_labels_batch in loader:
            out = model(images.to(device))
            raw_preds = out[0] if isinstance(out, (tuple, list)) else out
            preds_batch = _nms_predictions(raw_preds, conf_thresh=conf_thresh, imgsz=imgsz)

            for preds, gt_boxes in zip(preds_batch, gt_labels_batch):
                total_gt += len(gt_boxes)
                if not gt_boxes or not preds:
                    continue
                # Greedy matching: for each GT box, find best matching pred
                pred_matched = [False] * len(preds)

                for gi, (gt_cls, gt_cx, gt_cy, gt_w, gt_h) in enumerate(gt_boxes):
                    gt_xyxy = _xywh_to_xyxy(gt_cx, gt_cy, gt_w, gt_h)
                    best_iou = 0.0
                    best_pi = -1
                    for pi, (p_cls, p_cx, p_cy, p_w, p_h, p_conf) in enumerate(preds):
                        if pred_matched[pi]:
                            continue
                        p_xyxy = _xywh_to_xyxy(p_cx, p_cy, p_w, p_h)
                        iou = _box_iou(gt_xyxy, p_xyxy)
                        if iou > best_iou:
                            best_iou = iou
                            best_pi = pi
                    if best_pi >= 0 and best_iou > 0.1:
                        pred_matched[best_pi] = True
                        all_ious.append(best_iou)
                        total_matched += 1
                        if preds[best_pi][0] == gt_cls:
                            correct_cls += 1

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    cls_acc = correct_cls / total_matched if total_matched > 0 else 0.0
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    return mean_iou, cls_acc, recall


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


# ── Neuron selection ───────────────────────────────────────────────
def wisdom_neurons(csv_file: str, top_k: int = 10) -> Dict[str, List[int]]:
    """Read WISDOM CSV and return top-k neurons grouped by layer.
    Automatically excludes detection head layers (model.23.*)."""
    df = pd.read_csv(csv_file)
    # Filter out detection head layers
    df = df[~df["LayerName"].str.contains("model.23.")]
    df_sorted = df.sort_values(by="Score", ascending=False).head(top_k)
    result: Dict[str, List[int]] = {}
    for lname, group in df_sorted.groupby("LayerName"):
        result[lname] = group["NeuronIndex"].tolist()
    return result


def flatten_importance(
    importance: Dict[str, torch.Tensor],
    exclude_detect_head: bool = True,
) -> List[Tuple[str, float, int]]:
    """Flatten per-layer scores into sorted (layer, score, idx) list.
    Excludes detection head (model.23.*) by default."""
    flat: List[Tuple[str, float, int]] = []
    for lname, scores in importance.items():
        if exclude_detect_head and "model.23." in lname:
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
    """Zero weights and biases of selected neurons in-place.
    Skips any selection targeting the detection head (model.23.*)."""
    name2mod = dict(model.named_modules())
    for lname, idxs in selection.items():
        if DETECT_HEAD_PREFIX in lname:
            continue
        mod = name2mod.get(lname)
        if mod is None:
            continue
        with torch.no_grad():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                for idx in idxs:
                    mod.weight[idx].zero_()
                    if mod.bias is not None:
                        mod.bias[idx].zero_()


# ── RQ1 experiment ─────────────────────────────────────────────────
def run_rq1(
    weights: str,
    img_dir: str,
    csv_file: str,
    out_prefix: str = "results/rq1",
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

    # Prepare unlabeled data (confidence evaluation)
    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    # Prepare labeled data (IoU and classification accuracy)
    labeled_ds = COCOLabeledDataset(img_dir, max_images=num_images, imgsz=imgsz)
    labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=_collate_labeled)

    # Baseline performance
    baseline_conf = eval_yolo_confidence(torch_model, loader, device)
    baseline_iou, baseline_cls_acc, baseline_recall = eval_iou_and_accuracy(
        torch_model, labeled_loader, device, imgsz=imgsz
    )
    print(f"Baseline confidence: {baseline_conf:.2f}")
    print(f"Baseline mean IoU: {baseline_iou:.4f}")
    print(f"Baseline classification accuracy: {baseline_cls_acc:.4f}")
    print(f"Baseline detection recall: {baseline_recall:.4f}")

    # Get all trainable layer names (exclude detection head)
    trainable = [
        (n, m) for n, m in torch_model.named_modules()
        if _is_trainable_module(m) and DETECT_HEAD_PREFIX not in n
    ]
    all_neurons = [
        (n, i) for n, m in trainable
        for i in range(m.out_channels if isinstance(m, nn.Conv2d) else m.out_features)
    ]

    relevance_records = []
    accuracy_records = []

    # ── WISDOM consensus ──
    print("\n=== WISDOM Consensus ===")
    for n_prune in N_LIST:
        pruned = copy.deepcopy(torch_model)
        top_neurons = wisdom_neurons(csv_file, top_k=n_prune)
        mapped: Dict[str, List[int]] = {}
        for lname, idxs in top_neurons.items():
            if lname.startswith("yolo_model."):
                mapped[lname[len("yolo_model."):]] = idxs
            else:
                mapped[lname] = idxs
        prune_neurons(pruned, mapped)
        pruned_conf = eval_yolo_confidence(pruned, loader, device)
        pruned_iou, pruned_cls_acc, pruned_recall = eval_iou_and_accuracy(
            pruned, labeled_loader, device, imgsz=imgsz
        )
        conf_drop = baseline_conf - pruned_conf
        iou_drop = baseline_iou - pruned_iou
        cls_acc_drop = baseline_cls_acc - pruned_cls_acc
        recall_drop = baseline_recall - pruned_recall
        accuracy_records.append({
            "Attribution Method": "Wisdom",
            "Top-N": n_prune,
            "Confidence Drop": conf_drop,
            "Baseline Conf": baseline_conf,
            "Pruned Conf": pruned_conf,
            "Mean IoU": pruned_iou,
            "Baseline IoU": baseline_iou,
            "IoU Drop": iou_drop,
            "Cls Accuracy": pruned_cls_acc,
            "Baseline Cls Acc": baseline_cls_acc,
            "Cls Acc Drop": cls_acc_drop,
            "Det Recall": pruned_recall,
            "Baseline Recall": baseline_recall,
            "Recall Drop": recall_drop,
        })
        print(f"  Top-{n_prune}: conf_drop={conf_drop:.2f}, "
              f"IoU={pruned_iou:.4f} (Δ{iou_drop:+.4f}), "
              f"ClsAcc={pruned_cls_acc:.4f} (Δ{cls_acc_drop:+.4f}), "
              f"Recall={pruned_recall:.4f} (Δ{recall_drop:+.4f})")

    # ── Attribution methods ──
    wrapper = YOLOWrapper(torch_model, num_classes=80)
    wrapper.eval().to(device)

    for attr_key, attr_name in ATTRIBUTION_METHODS.items():
        print(f"\n=== {attr_name} ({attr_key}) ===")

        # Compute importance on first batch (excludes detection head)
        first_batch = next(iter(loader))
        images = first_batch[0]
        importance = _compute_yolo_importance(
            wrapper, images, attr_key, device, num_classes=80,
            exclude_detect_head=True,
        )

        # Save relevance scores
        for lname, scores in importance.items():
            for idx, score in enumerate(scores):
                relevance_records.append({
                    "Attribution Method": attr_name,
                    "Layer Name": lname,
                    "Neuron Index": idx,
                    "Relevance Score": float(score),
                })

        # Flatten and rank (excludes detection head)
        flat_scores = flatten_importance(importance, exclude_detect_head=True)
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
            pruned_iou, pruned_cls_acc, pruned_recall = eval_iou_and_accuracy(
                pruned, labeled_loader, device, imgsz=imgsz
            )
            conf_drop = baseline_conf - pruned_conf
            iou_drop = baseline_iou - pruned_iou
            cls_acc_drop = baseline_cls_acc - pruned_cls_acc
            recall_drop = baseline_recall - pruned_recall
            accuracy_records.append({
                "Attribution Method": attr_name,
                "Top-N": n_prune,
                "Confidence Drop": conf_drop,
                "Baseline Conf": baseline_conf,
                "Pruned Conf": pruned_conf,
                "Mean IoU": pruned_iou,
                "Baseline IoU": baseline_iou,
                "IoU Drop": iou_drop,
                "Cls Accuracy": pruned_cls_acc,
                "Baseline Cls Acc": baseline_cls_acc,
                "Cls Acc Drop": cls_acc_drop,
                "Det Recall": pruned_recall,
                "Baseline Recall": baseline_recall,
                "Recall Drop": recall_drop,
            })
            print(f"  Top-{n}: conf_drop={conf_drop:.2f}, "
                  f"IoU={pruned_iou:.4f} (Δ{iou_drop:+.4f}), "
                  f"ClsAcc={pruned_cls_acc:.4f} (Δ{cls_acc_drop:+.4f}), "
                  f"Recall={pruned_recall:.4f} (Δ{recall_drop:+.4f})")

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
            pruned_iou, pruned_cls_acc, pruned_recall = eval_iou_and_accuracy(
                pruned, labeled_loader, device, imgsz=imgsz
            )
            conf_drop = baseline_conf - pruned_conf
            iou_drop = baseline_iou - pruned_iou
            cls_acc_drop = baseline_cls_acc - pruned_cls_acc
            recall_drop = baseline_recall - pruned_recall
            accuracy_records.append({
                "Attribution Method": f"Random ({attr_name})",
                "Top-N": n_prune,
                "Confidence Drop": conf_drop,
                "Baseline Conf": baseline_conf,
                "Pruned Conf": pruned_conf,
                "Mean IoU": pruned_iou,
                "Baseline IoU": baseline_iou,
                "IoU Drop": iou_drop,
                "Cls Accuracy": pruned_cls_acc,
                "Baseline Cls Acc": baseline_cls_acc,
                "Cls Acc Drop": cls_acc_drop,
                "Det Recall": pruned_recall,
                "Baseline Recall": baseline_recall,
                "Recall Drop": recall_drop,
            })
            print(f"    Top-{n}: conf_drop={conf_drop:.2f}, "
                  f"IoU={pruned_iou:.4f} (Δ{iou_drop:+.4f}), "
                  f"ClsAcc={pruned_cls_acc:.4f} (Δ{cls_acc_drop:+.4f}), "
                  f"Recall={pruned_recall:.4f} (Δ{recall_drop:+.4f})")

    # Save results
    rel_path = f"{out_prefix}_relevance.csv"
    drop_path = f"{out_prefix}_acc_drop.csv"
    os.makedirs(os.path.dirname(rel_path) or ".", exist_ok=True)
    pd.DataFrame(relevance_records).to_csv(rel_path, index=False)
    pd.DataFrame(accuracy_records).to_csv(drop_path, index=False)
    print(f"\nSaved: {rel_path}, {drop_path}")

    # Generate visualization
    try:
        from wisdom.utils.visulization import viz_rq1_acc_drop
        model_tag = os.path.splitext(os.path.basename(weights))[0]  # e.g. "yolo11n"
        plot_path = f"{out_prefix}_{model_tag}_acc_drop.pdf"
        viz_rq1_acc_drop(drop_path, plot_path)
        print(f"Plot saved: {plot_path}")
    except Exception as e:
        print(f"Warning: could not generate plot: {e}")

    return rel_path, drop_path


# ── CLI ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RQ1: Critical neurons evaluation for YOLOv11")
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file", default="neuron_eval_out/wisdom_yolo11n_scores.csv", help="WISDOM scores CSV")
    p.add_argument("--out-prefix", default="results/rq1_yolo11n")
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

#!/usr/bin/env python
"""
run_rq1.py – RQ1: Do WISDOM-identified neurons matter?
=======================================================
Prunes the top-N neurons (N ∈ {6, 8, 10, 15, 20}) and measures the
detection-performance drop on a validation subset of COCO.

RQ1 compares:
  1. A WISDOM consensus neuron-importance CSV
  2. One or more single-attribution pretrained neuron-importance CSVs
  3. One shared random pruning baseline

This keeps the comparison fair: every attribution-guided method is evaluated
from a precomputed importance mapping rather than recomputing saliency on the
evaluation subset itself.

Metrics tracked:
  - Confidence drop (sum of class scores)
  - mAP50 drop
  - mAP50-95 drop
  - Mean IoU between predicted and ground-truth bounding boxes
  - Classification prediction accuracy (fraction of correct class predictions)
  - Detection recall

Outputs: rq1_relevance.csv, rq1_acc_drop.csv, rq1_yolo_{version}_acc_drop.pdf
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import shutil
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wisdom.core.wisdom_train import _is_trainable_module
from wisdom.utils.detection_loader import load_detection_model, normalize_detection_output

# ── Config ─────────────────────────────────────────────────────────
ATTRIBUTION_METHODS = {"lgxa": "GradXAct", "lig": "IntegGrad", "lgs": "GradShap"}
N_LIST = [6, 8, 10, 15, 20]
DETECT_HEAD_PREFIX = "model.23."


# ── Dataset with labels ───────────────────────────────────────────
class COCOLabeledDataset(Dataset):
    """Loads COCO images + YOLO-format labels for IoU/accuracy evaluation."""

    def __init__(
        self,
        img_dir: str,
        max_images: int = 50,
        imgsz: int = 320,
        sample_mode: str = "first",
        seed: Optional[int] = None,
        cache_images: bool = False,
    ):
        from torchvision import transforms

        self.img_dir = img_dir
        self.imgsz = imgsz
        self.label_dir = img_dir.replace("/images/", "/labels/")
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        all_imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(exts))

        all_pairs: List[Tuple[str, str]] = []
        for fname in all_imgs:
            label_fname = os.path.splitext(fname)[0] + ".txt"
            label_path = os.path.join(self.label_dir, label_fname)
            if os.path.isfile(label_path):
                all_pairs.append((os.path.join(img_dir, fname), label_path))

        if max_images is not None and len(all_pairs) > max_images:
            if sample_mode == "random":
                rng = random.Random(seed)
                chosen = sorted(rng.sample(range(len(all_pairs)), max_images))
                self.pairs = [all_pairs[i] for i in chosen]
            else:
                self.pairs = all_pairs[:max_images]
        else:
            self.pairs = all_pairs

        self.transform = transforms.Compose(
            [
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
            ]
        )
        self._cache = None
        if cache_images:
            self._cache = []
            for img_path, label_path in self.pairs:
                from PIL import Image

                img = Image.open(img_path).convert("RGB")
                self._cache.append((self.transform(img), self._parse_labels(label_path)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]

        from PIL import Image

        img_path, label_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        gt_boxes = self._parse_labels(label_path)
        return img_tensor, gt_boxes

    @staticmethod
    def _parse_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
        """Parse YOLO label file → list of (class_id, cx, cy, w, h)."""
        boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                if len(coords) == 4:
                    cx, cy, w, h = coords
                else:
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


def _link_or_copy(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_subset_data_yaml(
    image_label_pairs: List[Tuple[str, str]],
    subset_root: str,
    class_names,
) -> str:
    """Create a temporary YOLO dataset for the selected validation subset."""
    base = Path(subset_root)
    img_out = base / "images" / "val2017"
    label_out = base / "labels" / "val2017"
    img_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in image_label_pairs:
        _link_or_copy(img_path, img_out / Path(img_path).name)
        _link_or_copy(label_path, label_out / Path(label_path).name)

    if not isinstance(class_names, dict):
        class_names = {idx: name for idx, name in enumerate(class_names)}

    data_yaml = base / "data.yaml"
    with data_yaml.open("w") as f:
        f.write(f"path: {base}\n")
        f.write("train: images/val2017\n")
        f.write("val: images/val2017\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {repr(class_names)}\n")

    return str(data_yaml)


# ── IoU helpers ────────────────────────────────────────────────────
def _xywh_to_xyxy(cx, cy, w, h, img_size=1.0):
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return x1, y1, x2, y2


def _box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _nms_predictions(
    raw_preds,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    imgsz: int = 320,
    num_classes: int | None = None,
):
    """Simple NMS on normalized YOLO output (B, 4+nc, A)."""
    raw_preds = normalize_detection_output(raw_preds, num_classes=num_classes)
    results = []
    for b in range(raw_preds.shape[0]):
        pred = raw_preds[b]
        boxes_xywh = pred[:4, :]
        cls_scores = pred[4:, :]
        max_scores, cls_ids = cls_scores.max(dim=0)

        mask = max_scores > conf_thresh
        if mask.sum() == 0:
            results.append([])
            continue

        boxes = boxes_xywh[:, mask].T
        scores = max_scores[mask]
        classes = cls_ids[mask]

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
    model: nn.Module,
    loader: DataLoader,
    device: str,
    imgsz: int = 320,
    conf_thresh: float = 0.01,
    num_classes: int | None = None,
) -> Tuple[float, float, float]:
    """Evaluate mean IoU, classification accuracy and detection recall."""
    model.eval().to(device)
    all_ious = []
    correct_cls = 0
    total_matched = 0
    total_gt = 0

    with torch.no_grad():
        for images, gt_labels_batch in loader:
            out = model(images.to(device))
            preds_batch = _nms_predictions(out, conf_thresh=conf_thresh, imgsz=imgsz, num_classes=num_classes)

            for preds, gt_boxes in zip(preds_batch, gt_labels_batch):
                total_gt += len(gt_boxes)
                if not gt_boxes or not preds:
                    continue
                pred_matched = [False] * len(preds)

                for gt_cls, gt_cx, gt_cy, gt_w, gt_h in gt_boxes:
                    gt_xyxy = _xywh_to_xyxy(gt_cx, gt_cy, gt_w, gt_h)
                    best_iou = 0.0
                    best_pi = -1
                    for pi, (p_cls, p_cx, p_cy, p_w, p_h, _p_conf) in enumerate(preds):
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
            preds = normalize_detection_output(model(images.to(device)))
            cls_scores = preds[:, 4:, :]
            total_conf += cls_scores.sum().item()
            n_batches += 1
    return total_conf / max(n_batches, 1)


def eval_yolo_map(
    yolo_wrapper,
    model: nn.Module,
    data_yaml: str,
    device: str,
    imgsz: int = 320,
    batch_size: int = 2,
) -> Tuple[float, float]:
    """Evaluate dataset-level mAP50 and mAP50-95 on a temporary subset YAML."""
    original_model = yolo_wrapper.model
    try:
        yolo_wrapper.model = model.eval()
        metrics = yolo_wrapper.val(
            data=data_yaml,
            imgsz=imgsz,
            batch=batch_size,
            verbose=False,
            device=device,
            plots=False,
            save=False,
        )
        return float(metrics.box.map50), float(metrics.box.map)
    finally:
        yolo_wrapper.model = original_model


# ── Neuron selection ───────────────────────────────────────────────
def _normalize_layer_name(lname: str) -> str:
    if lname.startswith("yolo_model."):
        return lname[len("yolo_model."):]
    return lname


def load_top_neurons_from_csv(csv_file: str, top_k: int = 10) -> Dict[str, List[int]]:
    """Read a pretrained neuron-importance CSV and return top-k neurons grouped by layer."""
    df = pd.read_csv(csv_file)
    df = df[~df["LayerName"].str.contains("model.23.")]
    df_sorted = df.sort_values(by="Score", ascending=False).head(top_k)
    result: Dict[str, List[int]] = {}
    for lname, group in df_sorted.groupby("LayerName"):
        result[_normalize_layer_name(lname)] = group["NeuronIndex"].tolist()
    return result


def load_relevance_records(csv_file: str, method_label: str, method_key: str) -> List[Dict[str, object]]:
    df = pd.read_csv(csv_file)
    df = df[~df["LayerName"].str.contains("model.23.")]
    records: List[Dict[str, object]] = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "Attribution Method": method_label,
                "Method Key": method_key,
                "Layer Name": _normalize_layer_name(row.LayerName),
                "Neuron Index": int(row.NeuronIndex),
                "Relevance Score": float(row.Score),
                "Source CSV": csv_file,
            }
        )
    return records


# ── Pruning ────────────────────────────────────────────────────────
def prune_neurons(model: nn.Module, selection: Dict[str, List[int]]) -> None:
    """Zero weights and biases of selected neurons in-place."""
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
    methods: Optional[List[str]] = None,
    single_method_csvs: Optional[Dict[str, str]] = None,
    num_runs: int = 1,
    seed: int = 42,
    sample_mode: str = "auto",
    include_random: bool = True,
    eval_map: bool = False,
    num_workers: int = 0,
    cache_images: bool = False,
) -> Tuple[str, str]:
    """Run RQ1 experiment. Returns (relevance_csv_path, acc_drop_csv_path)."""
    if num_runs < 1:
        raise ValueError("--num-runs must be >= 1")
    if sample_mode not in {"auto", "first", "random"}:
        raise ValueError("--sample-mode must be one of: auto, first, random")

    single_method_csvs = single_method_csvs or {}
    if methods is None:
        methods = list(single_method_csvs)

    unknown_methods = [m for m in methods if m not in ATTRIBUTION_METHODS]
    if unknown_methods:
        raise ValueError(
            f"Unknown attribution method(s): {unknown_methods}. "
            f"Choose from {list(ATTRIBUTION_METHODS)}."
        )

    missing_csvs = [m for m in methods if m not in single_method_csvs]
    if missing_csvs:
        raise ValueError(
            "Missing pretrained CSV for single-method baselines: "
            f"{missing_csvs}. Provide them via single_method_csvs/--single-csv."
        )

    effective_sample_mode = "random" if sample_mode == "auto" and num_runs > 1 else sample_mode
    if effective_sample_mode == "auto":
        effective_sample_mode = "first"

    bundle = load_detection_model(weights, device=device)
    yolo = bundle.predictor
    torch_model = bundle.model.eval()
    model_tag = os.path.splitext(os.path.basename(weights))[0]

    trainable = [
        (n, m) for n, m in torch_model.named_modules()
        if _is_trainable_module(m) and DETECT_HEAD_PREFIX not in n
    ]
    all_neurons = [
        (n, i)
        for n, m in trainable
        for i in range(m.out_channels if isinstance(m, nn.Conv2d) else m.out_features)
    ]

    relevance_records = load_relevance_records(csv_file, "Wisdom", "wisdom")
    for attr_key in methods:
        relevance_records.extend(
            load_relevance_records(
                single_method_csvs[attr_key],
                ATTRIBUTION_METHODS[attr_key],
                attr_key,
            )
        )

    method_specs = [("Wisdom", "wisdom", csv_file)]
    method_specs.extend(
        (ATTRIBUTION_METHODS[attr_key], attr_key, single_method_csvs[attr_key])
        for attr_key in methods
    )

    accuracy_records = []

    for run_idx in range(num_runs):
        run_id = run_idx + 1
        run_seed = seed + run_idx
        print(
            f"\n=== RQ1 Run {run_id}/{num_runs} | "
            f"sample_mode={effective_sample_mode} | seed={run_seed} ==="
        )

        labeled_ds = COCOLabeledDataset(
            img_dir,
            max_images=num_images,
            imgsz=imgsz,
            sample_mode=effective_sample_mode,
            seed=run_seed,
            cache_images=cache_images,
        )
        labeled_loader = DataLoader(
            labeled_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_labeled,
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
        )

        subset_ctx = tempfile.TemporaryDirectory(prefix=f"rq1_val_run{run_id}_") if eval_map else nullcontext(None)
        with subset_ctx as subset_dir:
            subset_yaml = None
            baseline_map50 = None
            baseline_map = None
            if eval_map:
                subset_yaml = _prepare_subset_data_yaml(labeled_ds.pairs, subset_dir, yolo.names)

            baseline_conf = eval_yolo_confidence(torch_model, labeled_loader, device)
            baseline_iou, baseline_cls_acc, baseline_recall = eval_iou_and_accuracy(
                torch_model, labeled_loader, device, imgsz=imgsz, num_classes=bundle.num_classes
            )
            if subset_yaml is not None:
                if yolo is None:
                    raise ValueError("--eval-map is only supported for predictor-backed detection models.")
                baseline_map50, baseline_map = eval_yolo_map(
                    yolo, torch_model, subset_yaml, device, imgsz=imgsz, batch_size=batch_size
                )

            print(f"Baseline confidence: {baseline_conf:.2f}")
            if baseline_map50 is not None and baseline_map is not None:
                print(f"Baseline mAP50: {baseline_map50:.4f}")
                print(f"Baseline mAP50-95: {baseline_map:.4f}")
            print(f"Baseline mean IoU: {baseline_iou:.4f}")
            print(f"Baseline classification accuracy: {baseline_cls_acc:.4f}")
            print(f"Baseline detection recall: {baseline_recall:.4f}")

            for method_name, method_key, method_csv in method_specs:
                print(f"\n=== {method_name} ({method_key}) ===")
                for n_prune in N_LIST:
                    pruned = copy.deepcopy(torch_model)
                    selection = load_top_neurons_from_csv(method_csv, top_k=n_prune)
                    prune_neurons(pruned, selection)

                    pruned_conf = eval_yolo_confidence(pruned, labeled_loader, device)
                    pruned_iou, pruned_cls_acc, pruned_recall = eval_iou_and_accuracy(
                        pruned, labeled_loader, device, imgsz=imgsz, num_classes=bundle.num_classes
                    )
                    pruned_map50 = None
                    pruned_map = None
                    if subset_yaml is not None:
                        pruned_map50, pruned_map = eval_yolo_map(
                            yolo, pruned, subset_yaml, device, imgsz=imgsz, batch_size=batch_size
                        )

                    conf_drop = baseline_conf - pruned_conf
                    iou_drop = baseline_iou - pruned_iou
                    cls_acc_drop = baseline_cls_acc - pruned_cls_acc
                    recall_drop = baseline_recall - pruned_recall

                    record = {
                        "Model": model_tag,
                        "Run": run_id,
                        "Eval Seed": run_seed,
                        "Sample Mode": effective_sample_mode,
                        "Source CSV": method_csv,
                        "Method Key": method_key,
                        "Attribution Method": method_name,
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
                    }
                    if pruned_map50 is not None and pruned_map is not None:
                        record.update(
                            {
                                "mAP50": pruned_map50,
                                "Baseline mAP50": baseline_map50,
                                "mAP50 Drop": baseline_map50 - pruned_map50,
                                "mAP50-95": pruned_map,
                                "Baseline mAP50-95": baseline_map,
                                "mAP50-95 Drop": baseline_map - pruned_map,
                            }
                        )
                    accuracy_records.append(record)

                    msg = (
                        f"  Top-{n_prune}: conf_drop={conf_drop:.2f}, "
                        f"IoU={pruned_iou:.4f} (Δ{iou_drop:+.4f}), "
                        f"ClsAcc={pruned_cls_acc:.4f} (Δ{cls_acc_drop:+.4f}), "
                        f"Recall={pruned_recall:.4f} (Δ{recall_drop:+.4f})"
                    )
                    if pruned_map50 is not None and pruned_map is not None:
                        msg += (
                            f", mAP50={pruned_map50:.4f} (Δ{baseline_map50 - pruned_map50:+.4f}), "
                            f"mAP50-95={pruned_map:.4f} (Δ{baseline_map - pruned_map:+.4f})"
                        )
                    print(msg)

            if include_random:
                print("\n=== Random baseline ===")
                rng = random.Random(run_seed)
                for n_prune in N_LIST:
                    n = min(n_prune, len(all_neurons))
                    rand_sample = rng.sample(all_neurons, n)
                    pruned = copy.deepcopy(torch_model)
                    selection: Dict[str, List[int]] = {}
                    for lname, idx in rand_sample:
                        selection.setdefault(lname, []).append(idx)
                    prune_neurons(pruned, selection)

                    pruned_conf = eval_yolo_confidence(pruned, labeled_loader, device)
                    pruned_iou, pruned_cls_acc, pruned_recall = eval_iou_and_accuracy(
                        pruned, labeled_loader, device, imgsz=imgsz, num_classes=bundle.num_classes
                    )
                    pruned_map50 = None
                    pruned_map = None
                    if subset_yaml is not None:
                        pruned_map50, pruned_map = eval_yolo_map(
                            yolo, pruned, subset_yaml, device, imgsz=imgsz, batch_size=batch_size
                        )

                    conf_drop = baseline_conf - pruned_conf
                    iou_drop = baseline_iou - pruned_iou
                    cls_acc_drop = baseline_cls_acc - pruned_cls_acc
                    recall_drop = baseline_recall - pruned_recall

                    record = {
                        "Model": model_tag,
                        "Run": run_id,
                        "Eval Seed": run_seed,
                        "Sample Mode": effective_sample_mode,
                        "Source CSV": "",
                        "Method Key": "random",
                        "Attribution Method": "Random",
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
                    }
                    if pruned_map50 is not None and pruned_map is not None:
                        record.update(
                            {
                                "mAP50": pruned_map50,
                                "Baseline mAP50": baseline_map50,
                                "mAP50 Drop": baseline_map50 - pruned_map50,
                                "mAP50-95": pruned_map,
                                "Baseline mAP50-95": baseline_map,
                                "mAP50-95 Drop": baseline_map - pruned_map,
                            }
                        )
                    accuracy_records.append(record)

                    msg = (
                        f"  Top-{n}: conf_drop={conf_drop:.2f}, "
                        f"IoU={pruned_iou:.4f} (Δ{iou_drop:+.4f}), "
                        f"ClsAcc={pruned_cls_acc:.4f} (Δ{cls_acc_drop:+.4f}), "
                        f"Recall={pruned_recall:.4f} (Δ{recall_drop:+.4f})"
                    )
                    if pruned_map50 is not None and pruned_map is not None:
                        msg += (
                            f", mAP50={pruned_map50:.4f} (Δ{baseline_map50 - pruned_map50:+.4f}), "
                            f"mAP50-95={pruned_map:.4f} (Δ{baseline_map - pruned_map:+.4f})"
                        )
                    print(msg)

    rel_path = f"{out_prefix}_relevance.csv"
    drop_path = f"{out_prefix}_acc_drop.csv"
    os.makedirs(os.path.dirname(rel_path) or ".", exist_ok=True)
    pd.DataFrame(relevance_records).to_csv(rel_path, index=False)
    pd.DataFrame(accuracy_records).to_csv(drop_path, index=False)
    print(f"\nSaved: {rel_path}, {drop_path}")

    try:
        from wisdom.utils.visulization import viz_rq1_acc_drop, viz_rq1_topk_focus

        plot_path = f"{out_prefix}_{model_tag}_acc_drop.pdf"
        viz_rq1_acc_drop(drop_path, plot_path)
        print(f"Plot saved: {plot_path}")

        focus_dir = os.path.dirname(plot_path) or "."
        viz_rq1_topk_focus(
            drop_path,
            out_dir=focus_dir,
            out_prefix=os.path.basename(out_prefix),
            metrics=["confidence", "map50", "map5095"],
            split_metrics=True,
        )
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
    p.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(ATTRIBUTION_METHODS),
        default=[],
        help="Single-method baselines to evaluate alongside the WISDOM CSV. "
        "Each method must have a pretrained CSV provided via --single-csv.",
    )
    p.add_argument(
        "--single-csv",
        nargs="*",
        default=[],
        help="Pretrained single-method neuron CSVs in method=path form, e.g. "
        "--single-csv lgxa=neuron_eval_out/lgxa.csv lgs=neuron_eval_out/lgs.csv",
    )
    p.add_argument("--num-runs", type=int, default=1, help="Number of repeated RQ1 evaluation runs.")
    p.add_argument("--seed", type=int, default=42, help="Base seed for repeatable subset sampling.")
    p.add_argument(
        "--sample-mode",
        choices=["auto", "first", "random"],
        default="auto",
        help="Subset selection mode. auto=first for one run, random for repeated runs.",
    )
    p.add_argument(
        "--no-random",
        action="store_true",
        help="Disable the shared random pruning baseline.",
    )
    p.add_argument(
        "--eval-map",
        action="store_true",
        help="Also compute mAP50 and mAP50-95 drops via Ultralytics val().",
    )
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for full RQ1 evaluation.")
    p.add_argument(
        "--cache-images",
        action="store_true",
        help="Cache transformed evaluation images in RAM for repeated full-dataset passes.",
    )
    return p.parse_args()


def parse_single_csv_args(items: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --single-csv value '{item}'. Expected method=path.")
        method_key, csv_path = item.split("=", 1)
        if method_key not in ATTRIBUTION_METHODS:
            raise ValueError(
                f"Unknown method '{method_key}' in --single-csv. "
                f"Choose from {sorted(ATTRIBUTION_METHODS)}."
            )
        mapping[method_key] = csv_path
    return mapping


if __name__ == "__main__":
    args = parse_args()
    single_method_csvs = parse_single_csv_args(args.single_csv)
    run_rq1(
        weights=args.weights,
        img_dir=args.img_dir,
        csv_file=args.csv_file,
        out_prefix=args.out_prefix,
        device=args.device,
        num_images=args.num_images,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        methods=args.methods,
        single_method_csvs=single_method_csvs,
        num_runs=args.num_runs,
        seed=args.seed,
        sample_mode=args.sample_mode,
        include_random=not args.no_random,
        eval_map=args.eval_map,
        num_workers=args.num_workers,
        cache_images=args.cache_images,
    )

#!/usr/bin/env python
"""RQ2-style coverage testing and important-pixel visualization for ETerry YOLOv5s."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETERRY_ROOT = Path(__file__).resolve().parent
if str(ETERRY_ROOT) not in sys.path:
    sys.path.insert(0, str(ETERRY_ROOT))

from utils.general import non_max_suppression  # noqa: E402
from wisdom_pretrain import (  # noqa: E402
    ETERRY_ROOT as PRETRAIN_ROOT,
    ImagePathDataset,
    collect_image_paths,
    collate_images,
    iter_scored_layers,
    layer_group,
    layer_top_index,
    load_yolov5_model,
    yolo_confidence,
)

RESULTS_DIR = ETERRY_ROOT / "results"


def slug_float(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def load_groupwise_top_neurons(csv_path: Path, per_group_k: int) -> dict[str, list[int]]:
    df = pd.read_csv(csv_path)
    if "Group" not in df.columns:
        df["Group"] = df["LayerName"].map(layer_group)
    neurons: dict[str, list[int]] = defaultdict(list)
    for group in ("early", "middle", "late"):
        sub = df[(df["Group"] == group) & (df["Score"] > 0)].sort_values("Score", ascending=False)
        for _, row in sub.head(per_group_k).iterrows():
            neurons[str(row["LayerName"])].append(int(row["NeuronIndex"]))
    return {layer: sorted(set(indices)) for layer, indices in neurons.items()}


def selected_order(neurons: dict[str, list[int]]) -> list[tuple[str, int]]:
    order = []
    for layer in sorted(neurons, key=lambda item: (layer_top_index(item), item)):
        order.extend((layer, idx) for idx in sorted(neurons[layer]))
    return order


def collect_selected_activations(
    model,
    layers,
    images: torch.Tensor,
    neurons: dict[str, list[int]],
    device: str,
) -> torch.Tensor:
    acts = {}
    hooks = []
    for layer, indices in neurons.items():
        module = layers[layer]
        idx = torch.tensor(indices, dtype=torch.long, device=device)

        def hook(_module, _inputs, output, layer=layer, idx=idx):
            acts[layer] = output[:, idx].mean(dim=(2, 3)).detach().cpu()

        hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(images.to(device))
    finally:
        for hook_handle in hooks:
            hook_handle.remove()

    vectors = []
    for layer, idx in selected_order(neurons):
        layer_indices = neurons[layer]
        pos = layer_indices.index(idx)
        vectors.append(acts[layer][:, pos])
    return torch.stack(vectors, dim=1)


def calibrate_thresholds(model, layers, loader, neurons, device: str, percentile: float) -> torch.Tensor:
    chunks = []
    for batch in tqdm(loader, desc="Calibrating coverage thresholds"):
        chunks.append(collect_selected_activations(model, layers, batch.image, neurons, device))
    activations = torch.cat(chunks, dim=0).numpy()
    return torch.tensor(np.percentile(activations, percentile, axis=0), dtype=torch.float32)


def pixel_importance(model, layers, image: torch.Tensor, neurons: dict[str, list[int]], device: str) -> torch.Tensor:
    acts = {}
    hooks = []
    for layer, indices in neurons.items():
        module = layers[layer]
        idx = torch.tensor(indices, dtype=torch.long, device=device)

        def hook(_module, _inputs, output, layer=layer, idx=idx):
            acts[layer] = output[:, idx].mean(dim=(2, 3))

        hooks.append(module.register_forward_hook(hook))
    try:
        model.zero_grad(set_to_none=True)
        x = image.to(device).clone().requires_grad_(True)
        model(x)
        target = sum(value.sum() for value in acts.values())
        target.backward()
        return x.grad.detach().abs().mean(dim=1).cpu()
    finally:
        for hook_handle in hooks:
            hook_handle.remove()
        model.zero_grad(set_to_none=True)


def object_mask(model, image: torch.Tensor, device: str, conf_thresh: float, iou_thresh: float) -> torch.Tensor:
    _, _, h, w = image.shape
    with torch.no_grad():
        pred = model(image.to(device))[0]
        dets = non_max_suppression(pred, conf_thres=conf_thresh, iou_thres=iou_thresh)
    mask = torch.zeros(1, 1, h, w, dtype=torch.float32)
    for det in dets:
        if det is None or len(det) == 0:
            continue
        for *xyxy, _conf, _cls in det.detach().cpu().tolist():
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 > x1 and y2 > y1:
                mask[0, 0, y1:y2, x1:x2] = 1.0
    return mask


def select_pixels(importance: torch.Tensor, frac: float, allowed: torch.Tensor | None = None) -> torch.Tensor:
    flat = importance.reshape(-1)
    if allowed is None:
        candidates = torch.arange(flat.numel())
    else:
        candidates = torch.nonzero(allowed.reshape(-1) > 0.5, as_tuple=True)[0]
    if candidates.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    k = max(1, int(candidates.numel() * float(frac)))
    k = min(k, candidates.numel())
    top = flat[candidates].topk(k).indices
    return candidates[top]


def random_pixels(total: int, frac: float, allowed: torch.Tensor | None = None, exclude: torch.Tensor | None = None) -> torch.Tensor:
    if allowed is None:
        candidates = torch.arange(total)
    else:
        candidates = torch.nonzero(allowed.reshape(-1) > 0.5, as_tuple=True)[0]
    if exclude is not None and exclude.numel() > 0:
        excluded = set(int(v) for v in exclude.reshape(-1).tolist())
        candidates = torch.tensor([int(v) for v in candidates.tolist() if int(v) not in excluded], dtype=torch.long)
    if candidates.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    base = candidates.numel()
    k = max(1, int(base * float(frac)))
    k = min(k, candidates.numel())
    return candidates[torch.randperm(candidates.numel())[:k]]


def perturb(image: torch.Tensor, indices: torch.Tensor, noise_std: float) -> torch.Tensor:
    out = image.clone()
    if indices.numel() == 0:
        return out
    _, c, h, w = out.shape
    rows = indices // w
    cols = indices % w
    noise = torch.randn(c, indices.numel()) * noise_std
    out[0, :, rows, cols] = (out[0, :, rows, cols] + noise).clamp(0, 1)
    return out


def coverage_bits(model, layers, image: torch.Tensor, neurons, thresholds: torch.Tensor, device: str) -> torch.Tensor:
    acts = collect_selected_activations(model, layers, image, neurons, device)[0]
    return acts > thresholds


def run_coverage(args: argparse.Namespace) -> pd.DataFrame:
    started = time.time()
    model = load_yolov5_model(args.weights, args.device)
    layers = iter_scored_layers(model)
    neurons = load_groupwise_top_neurons(Path(args.wisdom_csv), args.per_group_k)
    if not neurons:
        raise RuntimeError(f"No positive neurons loaded from {args.wisdom_csv}")

    dataset = ImagePathDataset(Path(args.image_source), imgsz=args.imgsz, max_images=args.max_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.workers),
        pin_memory=str(args.device).startswith("cuda"),
        collate_fn=collate_images,
    )
    thresholds = calibrate_thresholds(model, layers, loader, neurons, args.device, args.threshold_percentile)
    total_neurons = int(thresholds.numel())
    rows = []

    for frac in parse_float_list(args.pixel_fracs):
        clean_values: list[float] = []
        whole_i_values: list[float] = []
        whole_r_values: list[float] = []
        obj_i_values: list[float] = []
        obj_r_values: list[float] = []

        iterator = tqdm(dataset, desc=f"RQ2 coverage pixel_frac={frac:.2f}")
        for image, _path in iterator:
            image = image.unsqueeze(0)
            importance = pixel_importance(model, layers, image, neurons, args.device)
            obj_mask = object_mask(model, image, args.device, args.conf_thresh, args.iou_thresh)
            total_pixels = image.shape[-1] * image.shape[-2]

            whole_idx = select_pixels(importance[0], frac)
            whole_rand_idx = random_pixels(total_pixels, frac, exclude=whole_idx)
            obj_idx = select_pixels(importance[0], frac, allowed=obj_mask[0, 0])
            obj_rand_idx = random_pixels(total_pixels, frac, allowed=obj_mask[0, 0], exclude=obj_idx)

            clean_bits = coverage_bits(model, layers, image, neurons, thresholds, args.device)
            whole_i_bits = coverage_bits(model, layers, perturb(image, whole_idx, args.noise_std), neurons, thresholds, args.device)
            whole_r_bits = coverage_bits(model, layers, perturb(image, whole_rand_idx, args.noise_std), neurons, thresholds, args.device)
            obj_i_bits = coverage_bits(model, layers, perturb(image, obj_idx, args.noise_std), neurons, thresholds, args.device)
            obj_r_bits = coverage_bits(model, layers, perturb(image, obj_rand_idx, args.noise_std), neurons, thresholds, args.device)

            clean_values.append(float(clean_bits.float().mean()))
            whole_i_values.append(float((clean_bits | whole_i_bits).float().mean()))
            whole_r_values.append(float((clean_bits | whole_r_bits).float().mean()))
            obj_i_values.append(float((clean_bits | obj_i_bits).float().mean()))
            obj_r_values.append(float((clean_bits | obj_r_bits).float().mean()))

        clean_value = float(np.mean(clean_values)) if clean_values else 0.0
        for scope, imp_cov, rand_cov in [
            ("whole_image", whole_i_values, whole_r_values),
            ("object_only", obj_i_values, obj_r_values),
        ]:
            imp_value = float(np.mean(imp_cov)) if imp_cov else 0.0
            rand_value = float(np.mean(rand_cov)) if rand_cov else 0.0
            imp_delta = imp_value - clean_value
            rand_delta = rand_value - clean_value
            rows.append(
                {
                    "pixel_frac": frac,
                    "scope": scope,
                    "clean_coverage": clean_value,
                    "important_union_coverage": imp_value,
                    "random_union_coverage": rand_value,
                    "important_delta": imp_delta,
                    "random_delta": rand_delta,
                    "delta_gap": imp_delta - rand_delta,
                    "delta_ratio": imp_delta / rand_delta if rand_delta > 0 else float("inf"),
                    "num_images": len(dataset),
                    "selected_neurons": total_neurons,
                    "threshold_percentile": args.threshold_percentile,
                }
            )

    summary = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "yolov5s_eterry_rq2_coverage_summary.csv"
    summary.to_csv(summary_csv, index=False)
    plot_coverage(summary, out_dir)
    metadata = {
        "weights": str(Path(args.weights).resolve()),
        "wisdom_csv": str(Path(args.wisdom_csv).resolve()),
        "image_source": str(Path(args.image_source).resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "runtime_seconds": round(time.time() - started, 3),
        "selected_neurons": selected_order(neurons),
    }
    (out_dir / "yolov5s_eterry_rq2_coverage_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    return summary


def plot_coverage(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, scope in zip(axes, ["whole_image", "object_only"]):
        sub = summary[summary["scope"] == scope].sort_values("pixel_frac")
        x = np.arange(len(sub))
        width = 0.34
        ax.bar(x - width / 2, sub["important_delta"], width=width, label="Important", color="#3366a3")
        ax.bar(x + width / 2, sub["random_delta"], width=width, label="Random", color="#8a8a8a")
        ax.set_title(scope.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v * 100:.0f}%" for v in sub["pixel_frac"]])
        ax.set_xlabel("Perturbed pixels")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Coverage delta over clean")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.22, wspace=0.12)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"yolov5s_eterry_rq2_coverage_whole_object.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def normalize_heatmap(importance: torch.Tensor) -> np.ndarray:
    arr = importance.detach().cpu().float().numpy()
    arr -= float(arr.min())
    max_value = float(arr.max())
    if max_value > 0:
        arr /= max_value
    return arr


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).NEAREST
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, resample=resampling)) > 127


def draw_detections(image: Image.Image, detections: torch.Tensor, names) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    width = max(2, min(out.size) // 260)
    for *xyxy, conf, cls_id in detections.detach().cpu().tolist():
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        draw.rectangle([x1, y1, x2, y2], outline=(40, 220, 120), width=width)
        label = f"{names.get(int(cls_id), cls_id)} {conf:.2f}" if isinstance(names, dict) else f"{names[int(cls_id)]} {conf:.2f}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        y0 = max(0, y1 - text_h - 6)
        draw.rectangle([x1, y0, x1 + text_w + 8, y0 + text_h + 6], fill=(40, 220, 120))
        draw.text((x1 + 4, y0 + 3), label, fill=(0, 0, 0), font=font)
    return out


def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 37, 120), alpha: float = 0.72) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).copy()
    selected = mask.astype(bool)
    if selected.any():
        color_arr = np.asarray(color, dtype=np.float32)
        arr[selected] = (arr[selected].astype(np.float32) * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return Image.fromarray(arr)


def predict_detections(model, image: torch.Tensor, original_size: tuple[int, int], device: str, conf: float, iou: float) -> torch.Tensor:
    with torch.no_grad():
        pred = model(image.to(device))[0]
        det = non_max_suppression(pred, conf_thres=conf, iou_thres=iou)[0]
    if det is None or len(det) == 0:
        return torch.zeros((0, 6))
    det = det.detach().cpu().clone()
    orig_w, orig_h = original_size
    model_h, model_w = image.shape[-2:]
    det[:, [0, 2]] *= orig_w / model_w
    det[:, [1, 3]] *= orig_h / model_h
    return det


def save_visualizations(args: argparse.Namespace) -> list[Path]:
    model = load_yolov5_model(args.weights, args.device)
    layers = iter_scored_layers(model)
    neurons = load_groupwise_top_neurons(Path(args.wisdom_csv), args.per_group_k)
    transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
    ])
    candidates = collect_image_paths(Path(args.visual_source))
    if not candidates:
        raise FileNotFoundError(f"No visualization images found under {args.visual_source}")
    if args.visual_count == 1:
        images = [candidates[0]]
    else:
        picks = np.linspace(0, len(candidates) - 1, args.visual_count).round().astype(int)
        images = [candidates[int(idx)] for idx in picks]

    out_dir = Path(args.out_dir) / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for image_path in images:
        with Image.open(image_path) as pil:
            original = pil.convert("RGB")
        image = transform(original).unsqueeze(0)
        importance = pixel_importance(model, layers, image, neurons, args.device)[0]
        heatmap = normalize_heatmap(importance)
        detections = predict_detections(model, image, original.size, args.device, args.conf_thresh, args.iou_thresh)
        boxed = draw_detections(original, detections, getattr(model, "names", {}))
        boxed_path = out_dir / f"{image_path.stem}_yolov5s_detection_boxes.png"
        boxed.save(boxed_path)
        outputs.append(boxed_path)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.imshow(original)
        ax.imshow(heatmap, cmap="magma", alpha=0.50, extent=[0, original.width, original.height, 0])
        ax.axis("off")
        heatmap_path = out_dir / f"{image_path.stem}_wisdom_pixel_heatmap.png"
        fig.savefig(heatmap_path, dpi=220, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        outputs.append(heatmap_path)

        for frac in parse_float_list(args.visual_pixel_fracs):
            top_idx = select_pixels(importance, frac)
            small_mask = np.zeros((args.imgsz, args.imgsz), dtype=bool)
            if top_idx.numel() > 0:
                rows = (top_idx // args.imgsz).numpy()
                cols = (top_idx % args.imgsz).numpy()
                small_mask[rows, cols] = True
            mask = resize_mask(small_mask, original.size)
            overlay = draw_detections(overlay_mask(original, mask), detections, getattr(model, "names", {}))
            overlay_path = out_dir / f"{image_path.stem}_important_pixels_frac_{slug_float(frac)}.png"
            overlay.save(overlay_path)
            outputs.append(overlay_path)
    for path in outputs:
        print(f"[eterry-wisdom] wrote {path}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ETerry YOLOv5s WISDOM coverage and visualizations.")
    parser.add_argument("--weights", default=str(ETERRY_ROOT / "weights" / "best.pt"))
    parser.add_argument("--wisdom-csv", default=str(RESULTS_DIR / "wisdom" / "wisdom_yolov5s_eterry_train.csv"))
    parser.add_argument("--image-source", default=str(ETERRY_ROOT / "dataset" / "images" / "train"))
    parser.add_argument("--visual-source", default=str(ETERRY_ROOT / "dataset" / "detect"))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "coverage"))
    parser.add_argument("--stage", choices=["coverage", "visualize", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--per-group-k", type=int, default=5)
    parser.add_argument("--pixel-fracs", default="0.02,0.05")
    parser.add_argument("--visual-pixel-fracs", default="0.02,0.05")
    parser.add_argument("--noise-std", type=float, default=0.30)
    parser.add_argument("--threshold-percentile", type=float, default=75.0)
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.45)
    parser.add_argument("--visual-count", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage in {"coverage", "all"}:
        run_coverage(args)
    if args.stage in {"visualize", "all"}:
        save_visualizations(args)


if __name__ == "__main__":
    main()

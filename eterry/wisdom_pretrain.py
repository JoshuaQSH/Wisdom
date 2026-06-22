#!/usr/bin/env python
"""WISDOM-style neuron pretraining for the standalone ETerry YOLOv5s model."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Hyperparameters and constants for ETerry YOLOv5s WISDOM pretraining
ETERRY_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ETERRY_ROOT / "results"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# For YOLOv5s WISDOM pretraining, we use three scoring methods: activation magnitude, gradient magnitude, and the product of activation and gradient (gxact)
METHODS = ("act", "grad", "gxact")

if str(ETERRY_ROOT) not in sys.path:
    sys.path.insert(0, str(ETERRY_ROOT))

from models.experimental import attempt_load  # noqa: E402

# Clean container for a mini-batch
# (images, paths)
@dataclass(frozen=True)
class ImageBatch:
    image: torch.Tensor
    path: tuple[str, ...]

# Load images from a folder, image file, or .txt file containing image paths.
class ImagePathDataset(Dataset):
    def __init__(self, image_source: Path, imgsz: int, max_images: int | None = None):
        self.paths = collect_image_paths(image_source)
        if max_images is not None:
            self.paths = self.paths[: int(max_images)]
        self.transform = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(path)


def collate_images(batch: list[tuple[torch.Tensor, str]]) -> ImageBatch:
    images, paths = zip(*batch)
    return ImageBatch(torch.stack(list(images), dim=0), tuple(paths))


def collect_image_paths(image_source: str | Path) -> list[Path]:
    source = Path(image_source)
    if source.is_file() and source.suffix.lower() == ".txt":
        paths = []
        for raw in source.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            path = Path(line)
            if not path.is_absolute():
                path = (source.parent / path).resolve()
            paths.append(path)
        return paths
    if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Unsupported image source: {source}")
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTS)


def load_yolov5_model(weights: str | Path, device: str) -> nn.Module:
    model = attempt_load(str(weights), device=device, fuse=False)
    model.eval().float().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def layer_top_index(layer_name: str) -> int:
    import re

    match = re.search(r"model\.(\d+)", layer_name)
    return int(match.group(1)) if match else 999


def layer_group(layer_name: str) -> str:
    idx = layer_top_index(layer_name)
    if idx <= 5:
        return "early"
    if idx <= 14:
        return "middle"
    return "late"


def iter_scored_layers(model: nn.Module) -> dict[str, nn.Conv2d]:
    layers: dict[str, nn.Conv2d] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if name.startswith("model.24."):
            continue
        layers[name] = module
    return layers

# Converts YOLO output into a scalar target.
def yolo_confidence(output) -> torch.Tensor:
    preds = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(preds, torch.Tensor) or preds.dim() != 3:
        raise ValueError(f"Unsupported YOLO output shape: {type(preds)}")
    if preds.shape[-1] >= 6:
        obj = preds[..., 4:5]
        cls = preds[..., 5:]
        return (obj * cls).sum()
    if preds.shape[1] >= 6:
        obj = preds[:, 4:5, :]
        cls = preds[:, 5:, :]
        return (obj * cls).sum()
    raise ValueError(f"Cannot infer confidence tensor from shape {tuple(preds.shape)}")


def init_score_buffers(layers: dict[str, nn.Conv2d]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros(module.out_channels, dtype=torch.float32) for name, module in layers.items()}


def select_top_per_group(
    scores: dict[str, torch.Tensor],
    *,
    top_m: int,
    selection_mode: str,
) -> list[tuple[str, int, float]]:
    selected: list[tuple[str, int, float]] = []
    if selection_mode == "global":
        flat = [
            (layer, idx, float(value))
            for layer, values in scores.items()
            for idx, value in enumerate(values.tolist())
        ]
        flat.sort(key=lambda item: item[2], reverse=True)
        return flat[:top_m]

    per_group = max(1, int(math.ceil(top_m / 3)))
    grouped: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for layer, values in scores.items():
        group = layer_group(layer)
        for idx, value in enumerate(values.tolist()):
            grouped[group].append((layer, idx, float(value)))
    for group in ("early", "middle", "late"):
        candidates = sorted(grouped[group], key=lambda item: item[2], reverse=True)
        selected.extend(candidates[:per_group])
    selected.sort(key=lambda item: item[2], reverse=True)
    return selected[: max(top_m, 3)]


def add_rank_votes(buffer: dict[str, torch.Tensor], selected: list[tuple[str, int, float]], weight: float = 1.0) -> None:
    ranked = list(reversed(selected))
    for rank, (layer, idx, _score) in enumerate(ranked, start=1):
        if layer in buffer and 0 <= idx < buffer[layer].numel():
            buffer[layer][idx] += float(rank) * float(weight)


def masked_confidence(
    model: nn.Module,
    layers: dict[str, nn.Conv2d],
    images: torch.Tensor,
    selected: list[tuple[str, int, float]],
) -> float:
    selected_by_layer: dict[str, list[int]] = defaultdict(list)
    for layer, idx, _score in selected:
        selected_by_layer[layer].append(idx)

    hooks = []
    for layer, indices in selected_by_layer.items():
        module = layers.get(layer)
        if module is None:
            continue
        idx_tensor = torch.tensor(sorted(set(indices)), dtype=torch.long, device=images.device)

        def hook(_module, _inputs, output, idx_tensor=idx_tensor):
            masked = output.clone()
            masked[:, idx_tensor] = 0
            return masked

        hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            return float(yolo_confidence(model(images)).detach().cpu())
    finally:
        for hook in hooks:
            hook.remove()


def normalize_selected(selected: list[tuple[str, int, float]], method_weight: float) -> list[tuple[str, int, float]]:
    if not selected:
        return []
    max_score = max(abs(score) for _layer, _idx, score in selected) or 1.0
    return [(layer, idx, float(score) / max_score * float(method_weight)) for layer, idx, score in selected]


def score_batch(
    model: nn.Module,
    layers: dict[str, nn.Conv2d],
    images: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    activations: dict[str, torch.Tensor] = {}
    hooks = []
    for layer_name, module in layers.items():
        def hook(_module, _inputs, output, layer_name=layer_name):
            activations[layer_name] = output
            output.retain_grad()

        hooks.append(module.register_forward_hook(hook))

    try:
        model.zero_grad(set_to_none=True)
        output = model(images)
        target = yolo_confidence(output)
        target.backward()

        scores = {method: {} for method in METHODS}
        for layer_name, activation in activations.items():
            grad = activation.grad
            if grad is None:
                continue
            scores["act"][layer_name] = activation.detach().abs().mean(dim=(0, 2, 3)).cpu()
            scores["grad"][layer_name] = grad.detach().abs().mean(dim=(0, 2, 3)).cpu()
            scores["gxact"][layer_name] = (activation.detach() * grad.detach()).abs().mean(dim=(0, 2, 3)).cpu()
        return scores
    finally:
        for hook_handle in hooks:
            hook_handle.remove()
        model.zero_grad(set_to_none=True)


def save_scores_csv(scores: dict[str, torch.Tensor], out_csv: Path) -> Path:
    rows = []
    for layer_name in sorted(scores, key=lambda item: (layer_top_index(item), item)):
        values = scores[layer_name]
        group = layer_group(layer_name)
        for idx, value in enumerate(values.tolist()):
            rows.append(
                {
                    "LayerName": layer_name,
                    "NeuronIndex": int(idx),
                    "Score": float(value),
                    "Group": group,
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def run(args: argparse.Namespace) -> dict:
    weights = Path(args.weights).resolve()
    image_source = Path(args.image_source).resolve()
    out_csv = Path(args.out_csv).resolve()
    method_dir = Path(args.method_dir).resolve()
    report_json = Path(args.report_json).resolve()

    model = load_yolov5_model(weights, args.device)
    layers = iter_scored_layers(model)
    if not layers:
        raise RuntimeError("No YOLOv5 convolution layers found for scoring.")

    dataset = ImagePathDataset(image_source, imgsz=args.imgsz, max_images=args.max_images)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No images found under {image_source}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.workers),
        pin_memory=str(args.device).startswith("cuda"),
        collate_fn=collate_images,
    )

    consensus_scores = init_score_buffers(layers)
    method_scores = {method: init_score_buffers(layers) for method in METHODS}

    started = time.time()
    for batch in tqdm(loader, desc="WISDOM YOLOv5 pretrain"):
        images = batch.image.to(args.device, non_blocking=True).requires_grad_(True)
        batch_scores = score_batch(model, layers, images)
        with torch.no_grad():
            base_conf = float(yolo_confidence(model(images.detach())).detach().cpu())

        weighted_candidates: list[tuple[str, int, float]] = []
        for method in METHODS:
            selected = select_top_per_group(
                batch_scores[method],
                top_m=args.top_m,
                selection_mode=args.selection_mode,
            )
            masked_conf = masked_confidence(model, layers, images.detach(), selected)
            drop = max(0.0, base_conf - masked_conf)
            weight = drop if drop > 1.0e-12 else 1.0
            add_rank_votes(method_scores[method], selected)
            weighted_candidates.extend(normalize_selected(selected, weight))

        weighted_candidates.sort(key=lambda item: item[2], reverse=True)
        add_rank_votes(consensus_scores, weighted_candidates[: args.top_m])

    out_csv = save_scores_csv(consensus_scores, out_csv)
    method_dir.mkdir(parents=True, exist_ok=True)
    method_csvs = {}
    for method, scores in method_scores.items():
        method_csv = save_scores_csv(scores, method_dir / f"wisdom_yolov5s_eterry_{method}_scores.csv")
        method_csvs[method] = str(method_csv)

    df = pd.read_csv(out_csv)
    summary = {
        "weights": str(weights),
        "image_source": str(image_source),
        "out_csv": str(out_csv),
        "method_csvs": method_csvs,
        "num_images": len(dataset),
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "top_m": args.top_m,
        "selection_mode": args.selection_mode,
        "scored_layers": len(layers),
        "rows": int(len(df)),
        "nonzero_rows": int((df["Score"] > 0).sum()),
        "score_sum": float(df["Score"].sum()),
        "runtime_seconds": round(time.time() - started, 3),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain WISDOM neuron rankings for ETerry YOLOv5s.")
    parser.add_argument("--weights", default=str(ETERRY_ROOT / "weights" / "best.pt"))
    parser.add_argument("--image-source", default=str(ETERRY_ROOT / "dataset" / "images" / "train"))
    parser.add_argument("--out-csv", default=str(RESULTS_DIR / "wisdom" / "wisdom_yolov5s_eterry_train.csv"))
    parser.add_argument("--method-dir", default=str(RESULTS_DIR / "wisdom" / "method_scores"))
    parser.add_argument("--report-json", default=str(RESULTS_DIR / "wisdom" / "wisdom_yolov5s_eterry_summary.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--top-m", type=int, default=20)
    parser.add_argument("--selection-mode", choices=["per-group", "global"], default="per-group")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

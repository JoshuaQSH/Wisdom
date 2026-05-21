from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import torch
import torch.nn as nn


@dataclass
class DetectionModelBundle:
    family: str
    model: nn.Module
    predictor: Any
    names: dict[int, str] | list[str]
    num_classes: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_yolov5_weights(weights_path: str) -> bool:
    stem = Path(str(weights_path)).stem.lower()
    return "yolov5" in stem


def _ensure_yolov5_path() -> Path:
    yolov5_root = _repo_root() / "standalone" / "ultralytics_yolov5"
    if str(yolov5_root) not in sys.path:
        sys.path.insert(0, str(yolov5_root))
    return yolov5_root


def infer_num_classes(model: nn.Module, names: dict[int, str] | list[str] | None = None) -> int:
    if names:
        return len(names)
    model_names = getattr(model, "names", None)
    if model_names:
        return len(model_names)
    last = getattr(model, "model", None)
    if last is not None and len(last) > 0:
        nc = getattr(last[-1], "nc", None)
        if nc is not None:
            return int(nc)
    nc = getattr(model, "nc", None)
    if nc is not None:
        return int(nc)
    return 80


def load_detection_model(weights: str, device: str = "cpu") -> DetectionModelBundle:
    def _load_yolov5_bundle() -> DetectionModelBundle:
        _ensure_yolov5_path()
        from models.experimental import attempt_load

        model = attempt_load(weights, device=device, fuse=False)
        model = model.eval().to(device)
        names = getattr(model, "names", {})
        num_classes = infer_num_classes(model, names)
        return DetectionModelBundle(
            family="yolov5",
            model=model,
            predictor=None,
            names=names,
            num_classes=num_classes,
        )

    if is_yolov5_weights(weights):
        return _load_yolov5_bundle()

    from ultralytics import YOLO

    try:
        predictor = YOLO(weights)
        model = predictor.model.eval().to(device)
        names = getattr(predictor, "names", getattr(model, "names", {}))
        num_classes = infer_num_classes(model, names)
        return DetectionModelBundle(
            family="ultralytics",
            model=model,
            predictor=predictor,
            names=names,
            num_classes=num_classes,
        )
    except TypeError as exc:
        message = str(exc)
        if "YOLOv5 model" in message or "originally trained with https://github.com/ultralytics/yolov5" in message:
            return _load_yolov5_bundle()
        raise


def normalize_detection_output(
    output: Any,
    num_classes: int | None = None,
) -> torch.Tensor:
    preds = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(preds, torch.Tensor) or preds.dim() != 3:
        raise ValueError(f"Unsupported detection output type/shape: {type(preds)}")

    if num_classes is None:
        small_dim = min(preds.shape[1], preds.shape[2])
        if small_dim == 84:
            num_classes = 80
        elif small_dim == 85:
            num_classes = 80
        elif small_dim >= 6:
            num_classes = small_dim - 5
        else:
            num_classes = 80

    if preds.shape[1] == 4 + num_classes:
        return preds
    if preds.shape[1] == 5 + num_classes:
        boxes = preds[:, :4, :]
        obj = preds[:, 4:5, :]
        cls = preds[:, 5:, :] * obj
        return torch.cat([boxes, cls], dim=1)
    if preds.shape[2] == 4 + num_classes:
        return preds.permute(0, 2, 1).contiguous()
    if preds.shape[2] == 5 + num_classes:
        boxes = preds[..., :4].permute(0, 2, 1).contiguous()
        obj = preds[..., 4:5]
        cls = (preds[..., 5:] * obj).permute(0, 2, 1).contiguous()
        return torch.cat([boxes, cls], dim=1)

    raise ValueError(f"Cannot normalize detection output with shape {tuple(preds.shape)}")


def detect_head_prefixes(model: nn.Module, wrapper_prefix: str = "yolo_model.") -> list[str]:
    prefixes: list[str] = []
    for name, module in model.named_modules():
        if module.__class__.__name__.lower() == "detect" and name:
            prefixes.append(f"{wrapper_prefix}{name}.")
    if prefixes:
        return prefixes
    model_seq = getattr(model, "model", None)
    if model_seq is not None and len(model_seq) > 0:
        prefixes.append(f"{wrapper_prefix}model.{len(model_seq) - 1}.")
    return prefixes

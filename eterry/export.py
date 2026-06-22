"""Minimal YOLOv5 export compatibility helpers for ETerry detection.

The full upstream export CLI is intentionally not included in this cleaned
standalone folder. `models.common.DetectMultiBackend` only needs this table to
identify model suffixes, so keeping the small helper preserves normal `.pt`
detection without carrying export-only dependencies.
"""
from __future__ import annotations

import pandas as pd


def export_formats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Format": [
                "PyTorch",
                "TorchScript",
                "ONNX",
                "OpenVINO",
                "TensorRT",
                "CoreML",
                "TensorFlow SavedModel",
                "TensorFlow GraphDef",
                "TensorFlow Lite",
                "TensorFlow Edge TPU",
                "TensorFlow.js",
                "PaddlePaddle",
            ],
            "Suffix": [
                ".pt",
                ".torchscript",
                ".onnx",
                "_openvino_model",
                ".engine",
                ".mlpackage",
                "_saved_model",
                ".pb",
                ".tflite",
                "_edgetpu.tflite",
                "_web_model",
                "_paddle_model",
            ],
        }
    )

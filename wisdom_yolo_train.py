#!/usr/bin/env python
"""
wisdom_yolo_train.py
====================
Consensus-based neuron importance scoring for YOLOv11 using WISDOM.

Loads a pretrained YOLOv11 model, runs consensus voting across multiple
attribution methods on a subset of COCO images, and writes per-layer
neuron importance scores to a CSV file.

Usage
-----
    python wisdom_yolo_train.py \
        --weights standalone/models/yolo11n.pt \
        --data standalone/data/coco128.yaml \
        --batch-size 4 \
        --num-images 100 \
        --top-m 20 \
        --methods lgxa lig lgs \
        --voting-mode fine-grained \
        --out-csv wisdom_yolo_scores.csv \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image

# Ensure the repo root is on the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wisdom.core.wisdom_train import ConsensusWisdom, WisdomTrainConfig


# ------------------------------------------------------------------
# Simple COCO image dataset (no labels needed for YOLO surrogate)
# ------------------------------------------------------------------
class COCOImageDataset(torch.utils.data.Dataset):
    """Returns (image_tensor,) for each JPEG in *img_dir*."""

    def __init__(self, img_dir: str, max_images: int | None = None, imgsz: int = 640):
        self.paths = sorted(Path(img_dir).glob("*.jpg"))
        if max_images is not None:
            self.paths = self.paths[:max_images]
        self.transform = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img),  # 1-element tuple


def _collate(batch):
    """Stack single-element tuples into (B, C, H, W) tensor."""
    imgs = torch.stack([b[0] for b in batch])
    return (imgs,)


# ------------------------------------------------------------------
# Core training function (importable by tests)
# ------------------------------------------------------------------
def train_wisdom_yolo(
    weights: str,
    img_dir: str,
    out_csv: str,
    batch_size: int = 4,
    num_images: int = 100,
    top_m: int = 20,
    methods: list[str] | None = None,
    voting_mode: str = "fine-grained",
    device: str = "cuda:0",
    imgsz: int = 640,
) -> str:
    """
    Run WISDOM consensus training on a YOLOv11 model.

    Returns the path to the generated CSV file.
    """
    if methods is None:
        methods = ["lgxa", "lig", "lgs"]

    # Load YOLOv11 torch model
    from ultralytics import YOLO
    yolo = YOLO(weights)
    torch_model = yolo.model.eval()

    # Build dataloader
    ds = COCOImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
    if len(ds) == 0:
        raise FileNotFoundError(f"No images found in {img_dir}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    # Configure WISDOM
    cfg = WisdomTrainConfig(
        methods=methods,
        device=device,
        voting_mode=voting_mode,
        out_csv=out_csv,
        is_yolo=True,
        num_classes=80,
    )

    # Run consensus
    cw = ConsensusWisdom(torch_model, device=device)
    layer_scores, csv_path = cw.fit(loader, cfg, top_m_neurons=top_m, prune_mode="mask")

    print(f"Saved layer scores to {csv_path}")
    print(f"Layers scored: {len(layer_scores)}")
    total_scored = sum(t.numel() for t in layer_scores.values())
    non_zero = sum((t != 0).sum().item() for t in layer_scores.values())
    print(f"Total neurons: {total_scored}, non-zero scores: {non_zero}")
    return csv_path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="WISDOM consensus training for YOLOv11")
    p.add_argument("--weights", default="standalone/models/yolo11n.pt", help="YOLOv11 weights file")
    p.add_argument("--data", default="standalone/data/coco128.yaml", help="Dataset YAML (used to locate images)")
    p.add_argument("--img-dir", default=None, help="Override: path to image directory")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-images", type=int, default=100, help="Max images to use")
    p.add_argument("--top-m", type=int, default=20, help="Top-M neurons per method")
    p.add_argument("--methods", nargs="+", default=["lgxa", "lig", "lgs"])
    p.add_argument("--voting-mode", default="fine-grained", choices=["fine-grained", "coarse"])
    p.add_argument("--out-csv", default="wisdom_yolo_scores.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve image directory from YAML or override
    if args.img_dir:
        img_dir = args.img_dir
    else:
        import yaml
        with open(args.data) as f:
            data_cfg = yaml.safe_load(f)
        img_dir = data_cfg.get("train", "")
        if not os.path.isabs(img_dir):
            img_dir = os.path.join(os.path.dirname(args.data), img_dir)

    csv_path = train_wisdom_yolo(
        weights=args.weights,
        img_dir=img_dir,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        num_images=args.num_images,
        top_m=args.top_m,
        methods=args.methods,
        voting_mode=args.voting_mode,
        device=args.device,
        imgsz=args.imgsz,
    )
    print(f"\nDone. CSV: {csv_path}")

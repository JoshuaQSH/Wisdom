#!/usr/bin/env python
"""
wisdom_yolo_train.py
====================
Detection WISDOM pretraining CLI wrapper.

Loads a pretrained YOLO detection model, resolves a dataset image source from a
YAML file or explicit path, and delegates the actual WISDOM pretraining work to
`wisdom.core.wisdom_train.train_wisdom_yolo(...)`.

Usage
-----
    python wisdom_yolo_train.py \
        --weights weights/yolo11n.pt \
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

from wisdom.core.wisdom_train import COCOImageDataset, collate_image_tuples, train_wisdom_yolo


# Backward-compatible wrapper: the packaged implementation now lives in
# wisdom.core.wisdom_train.
_collate = collate_image_tuples


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detection WISDOM pretraining wrapper")
    p.add_argument("--weights", default="weights/yolo11n.pt", help="YOLO weights file")
    p.add_argument("--data", default="standalone/data/coco128.yaml", help="Dataset YAML whose train entry resolves the image source.")
    p.add_argument("--img-dir", default=None, help="Override image source: directory, txt list, or single image path.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for image decoding")
    p.add_argument("--num-images", type=int, default=100, help="Max images to use")
    p.add_argument("--top-m", type=int, default=20, help="Top-M neurons per method")
    p.add_argument("--methods", nargs="+", default=["lgxa", "lig", "lgs"])
    p.add_argument("--voting-mode", default="fine-grained", choices=["fine-grained", "coarse"])
    p.add_argument("--selection-mode", default="global", choices=["global", "per-group"],
                   help="global: top-M across all layers; per-group: top-M/3 per early/middle/late")
    p.add_argument("--out-csv", default="neuron_eval_out/wisdom_yolo_scores.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pt checkpoint file for resume support")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Save checkpoint every N batches (default 50)")
    p.add_argument(
        "--method-out-csv",
        nargs="*",
        default=[],
        help="Optional method-level CSV outputs in method=path form.",
    )
    return p


def _resolve_image_source(args) -> str:
    if args.img_dir:
        return args.img_dir
    import yaml

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    image_source = data_cfg.get("train", "")
    if not os.path.isabs(image_source):
        image_source = os.path.join(os.path.dirname(args.data), image_source)
    return image_source


def parse_args():
    return build_parser().parse_args()


def parse_method_out_csvs(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --method-out-csv value '{item}'. Expected method=path.")
        method, path = item.split("=", 1)
        method = method.strip().lower()
        if not method:
            raise ValueError(f"Invalid --method-out-csv value '{item}'. Empty method.")
        mapping[method] = path
    return mapping


def main() -> str:
    args = parse_args()
    image_source = _resolve_image_source(args)
    method_out_csvs = parse_method_out_csvs(args.method_out_csv)

    csv_path = train_wisdom_yolo(
        weights=args.weights,
        img_dir=image_source,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        num_images=args.num_images,
        top_m=args.top_m,
        methods=args.methods,
        voting_mode=args.voting_mode,
        selection_mode=args.selection_mode,
        device=args.device,
        imgsz=args.imgsz,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        method_out_csvs=method_out_csvs or None,
        num_workers=args.num_workers,
    )
    print(f"\nDone. CSV: {csv_path}")
    return csv_path


if __name__ == "__main__":
    main()

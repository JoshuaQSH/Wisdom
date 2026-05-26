#!/usr/bin/env python
"""
wisdom_classification_train.py
=============================
Consensus-based neuron importance scoring for classification models using WISDOM.

Supports either the packaged public datasets (`mnist`, `cifar10`, `cifar100`,
`imagenet`) or a custom `ImageFolder` dataset root.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from run_cases.support import get_data, resolve_saved_model_path
from wisdom import train_wisdom_classification
from wisdom.utils.common import get_model, get_trainable_modules_main


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _normalize_transform(mode: str, grayscale: bool):
    key = mode.lower()
    if key == "none":
        return None
    if key == "imagenet":
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if key == "cifar":
        return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if key == "mnist":
        if not grayscale:
            raise ValueError("--normalize mnist requires --grayscale for custom ImageFolder datasets.")
        return transforms.Normalize((0.1307,), (0.3081,))
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _build_imagefolder_loader(
    root: str,
    batch_size: int,
    image_size: int,
    grayscale: bool,
    normalize: str,
) -> DataLoader:
    transform_steps = [transforms.Resize((image_size, image_size))]
    if grayscale:
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
    else:
        transform_steps.append(transforms.Lambda(lambda img: img.convert("RGB")))
    transform_steps.append(transforms.ToTensor())
    normalizer = _normalize_transform(normalize, grayscale)
    if normalizer is not None:
        transform_steps.append(normalizer)
    dataset = datasets.ImageFolder(root=root, transform=transforms.Compose(transform_steps))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _load_legacy_torch_model(model_path: str):
    resolved = resolve_saved_model_path(model_path)
    model_file = Path(resolved).resolve()
    repo_root = model_file.parents[2] if len(model_file.parents) > 2 else Path.cwd()
    added = False
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        added = True
    try:
        model, _, _ = get_model(str(model_file))
    finally:
        if added:
            sys.path.pop(0)
    return model.eval(), str(model_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WISDOM consensus training for classification models")
    parser.add_argument("--model-path", required=True, help="Classification .pth checkpoint")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset",
        choices=["mnist", "cifar10", "cifar100", "imagenet"],
        help="Packaged public dataset to load through run_cases.support.get_data().",
    )
    source_group.add_argument(
        "--imagefolder-root",
        help="Custom classification dataset root in torchvision ImageFolder layout.",
    )
    parser.add_argument("--data-path", default="/scratch/staff/lrr550/datasets")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224, help="Resize used for custom ImageFolder datasets.")
    parser.add_argument("--grayscale", action="store_true", help="Convert custom ImageFolder inputs to one channel.")
    parser.add_argument(
        "--normalize",
        choices=["none", "imagenet", "cifar", "mnist"],
        default="none",
        help="Normalization preset for custom ImageFolder datasets.",
    )
    parser.add_argument("--top-m", type=int, default=20, help="Top-M neurons per method")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--voting-mode", default="fine-grained", choices=["fine-grained", "coarse"])
    parser.add_argument("--out-csv", default="neuron_eval_out/wisdom_classification_scores.csv")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint file for resume support")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N batches")
    return parser


def main() -> str:
    args = build_parser().parse_args()
    model, _resolved_model_path = _load_legacy_torch_model(args.model_path)
    _trainable_modules, trainable_names = get_trainable_modules_main(model)
    final_layer = trainable_names[-1] if trainable_names else None

    if args.dataset:
        _train_loader, _test_loader, train_dataset, _test_dataset, _classes = get_data(
            args.dataset,
            args.batch_size,
            args.data_path,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        train_loader = _build_imagefolder_loader(
            root=args.imagefolder_root,
            batch_size=args.batch_size,
            image_size=args.image_size,
            grayscale=args.grayscale,
            normalize=args.normalize,
        )

    csv_path = train_wisdom_classification(
        model=model,
        train_loader=train_loader,
        out_csv=args.out_csv,
        top_m=args.top_m,
        methods=args.methods,
        voting_mode=args.voting_mode,
        device=args.device,
        final_layer=final_layer,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"\nDone. CSV: {csv_path}")
    return csv_path


if __name__ == "__main__":
    main()

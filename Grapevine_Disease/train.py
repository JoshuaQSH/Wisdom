#!/usr/bin/env python
"""
Reproducing "Multiclass classification of diseased grape leaf identification using DCNN" with PyTorch.

This training script supports three model variants from the paper:
  * simple_cnn        - baseline CNN without augmentation
  * cnn_aug           - baseline CNN with augmentation
  * dcnn              - VGG16‑based DCNN with three extra conv layers (transfer learning)

Usage examples
--------------
Train a simple baseline CNN without augmentation:
python train.py --data-root ./datasets/grape_disease_original --model simple_cnn --epochs 30

Train a CNN  + augmentation:
python train.py --data-root ./datasets/grape_disease_original --model cnn_aug --epochs 30 --augment

Train the VGG16-based DCNN with augmentation:
python train.py --data-root ./datasets/grape_disease_original --model dcnn --epochs 30 --augment
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import SimpleCNN, DCNN
from utils import parse_args, set_seed, build_datasets, compute_metrics, plot_confusion_matrix, plot_curves

SEED = 42
NUM_CLASSES = 4  # Black Rot, ESCA, Leaf Blight, Healthy
IMAGE_SIZE = 224  # VGG default
CLASS = ["Black Rot", "ESCA", "Leaf Blight", "Healthy"]

def build_model(name: str) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)
    elif name == "cnn_aug":
        return SimpleCNN(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)
    elif name == "dcnn":
        return DCNN(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model name {name}")

def epoch_step(model, dataloader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_meter = 0.0
    all_preds = []
    all_targets = []
    loop = tqdm(dataloader, leave=False, ncols=80)
    for images, targets in loop:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_meter += loss.item() * images.size(0)
        all_preds.append(outputs.softmax(dim=1).argmax(dim=1).cpu())
        all_targets.append(targets.cpu())
    loss_meter /= len(dataloader.dataset)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return loss_meter, preds, targets

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    set_seed(SEED)

    data_root = Path(args.data_root)
    train_ds, val_ds, test_ds = build_datasets(data_root, augment=args.augment or args.model == "cnn_aug")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # run_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("logs")
    run_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", ncols=80):
    # for epoch in range(1, args.epochs + 1):
        train_loss, train_preds, train_targets = epoch_step(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = epoch_step(model, val_loader, criterion, None, device)
        scheduler.step()

        train_metrics = compute_metrics(train_preds, train_targets, NUM_CLASSES)
        val_metrics = compute_metrics(val_preds, val_targets, NUM_CLASSES)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        
        # log = {
        #     "epoch": epoch,
        #     "train_loss": train_loss,
        #     "val_loss": val_loss,
        #     **{f"train_{k}": v for k, v in train_metrics.items()},
        #     **{f"val_{k}": v for k, v in val_metrics.items()},
        # }
        
        # print(json.dumps(log))

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, run_dir / f"best_{args.model}.pt")

    # Evaluate on the test set with the best checkpoint
    ckpt = torch.load(run_dir / f"best_{args.model}.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    
    # Save the train/val accuracy and loss history
    plot_curves(history, run_dir, args.model)
    
    test_loss, test_preds, test_targets = epoch_step(model, test_loader, criterion, None, device)
    test_metrics = compute_metrics(test_preds, test_targets, NUM_CLASSES)
    print(f"Test metrics for {args.model}:", json.dumps(test_metrics, indent=2))
    
    # Plot confusion matrix
    plot_confusion_matrix(test_preds, test_targets, CLASS, run_dir, args.model)


def evaluate_only(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    data_root = Path(args.data_root)
    _, _, test_ds = build_datasets(data_root, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.model).to(device)
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    criterion = nn.CrossEntropyLoss()

    test_loss, test_preds, test_targets = epoch_step(model, test_loader, criterion, None, device)
    print("Loaded checkpoint from epoch", ckpt.get("epoch", "N/A"))
    print(json.dumps(compute_metrics(test_preds, test_targets, NUM_CLASSES), indent=2))


if __name__ == "__main__":
    args = parse_args()
    print(args)     
    if args.eval:
        evaluate_only(args)
    else:
        train(args)
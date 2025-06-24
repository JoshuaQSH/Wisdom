import argparse
import random
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce grape leaf disease classification experiments")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing train/ and test/ folders")
    parser.add_argument("--model", type=str, default="dcnn", choices=["simple_cnn", "cnn_aug", "dcnn"], help="Model variant to train")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training/evaluation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation for training split")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume/evaluate")
    parser.add_argument("--eval", action="store_true", help="Only evaluate the given checkpoint")

    args = parser.parse_args()
    return args

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(preds, targets, num_classes = 4):
    acc = MulticlassAccuracy(num_classes)(preds, targets)
    prec = MulticlassPrecision(num_classes, average="macro")(preds, targets)
    rec = MulticlassRecall(num_classes, average="macro")(preds, targets)
    f1 = MulticlassF1Score(num_classes, average="macro")(preds, targets)
    return {
        "accuracy": acc.item(),
        "precision": prec.item(),
        "recall": rec.item(),
        "f1": f1.item(),
    }

def get_transforms(split: str, augment: bool = False, image_size: int = 224):
    if split == "train":
        if augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.1)),  # zoom ±10 %
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def build_datasets(data_root: Path, augment: bool) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    full_train = datasets.ImageFolder(train_dir, transform=get_transforms("train", augment))
    val_len = int(0.2 * len(full_train))
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len])

    test_ds = datasets.ImageFolder(test_dir, transform=get_transforms("test"))
    return train_ds, val_ds, test_ds


def plot_curves(history: dict[str, list[float]], save_dir: Path, pattern: str='Simple_CNN') -> None:
    """Plot loss and accuracy curves; save two PNG files in *save_dir*."""

    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross‑entropy loss")
    plt.title("Training vs validation loss" + f" ({pattern})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"loss_curve_{pattern}.png", dpi=300)
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs validation accuracy" + f" ({pattern})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"accuracy_curve_{pattern}.png", dpi=300)
    plt.close()

    print(f"Saved curves → {save_dir}")

def plot_confusion_matrix(y_true: list[int], y_pred: list[int], class_names: list[str], save_dir: Path, pattern: str='Simple_CNN') -> None:
    """Compute and save a confusion‑matrix PNG to *save_dir*."""

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.title("Confusion matrix (test set)" + f"({pattern})")
    plt.tight_layout()
    save_path = save_dir / f"confusion_matrix_{pattern}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix → {save_path}")
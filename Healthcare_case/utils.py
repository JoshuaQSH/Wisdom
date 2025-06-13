import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from typing import Dict
from sklearn.model_selection import train_test_split

# A helper function to plot training history
def plot_history(hist):
    """
    hist = {
        'train_loss': [...], 'val_loss' : [...],
        'train_acc' : [...], 'val_acc'  : [...]
    }
    """
    start_ms = int(time.time() * 1000)
    timestamp = time.strftime("%Y%m%d‑%H%M%S", time.localtime(start_ms / 1000))
    
    train_iter = range(1, len(hist['train_loss']) + 1)
    epochs = range(1, len(hist['val_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].plot(train_iter, hist['train_loss'], lw=2)
    axes[0, 0].set_title('Train Loss')
    axes[0, 0].set_xlabel('Training Iterations')
    axes[0, 0].set_ylabel('Loss')
    
    axes[0, 1].plot(train_iter, hist['train_acc'], lw=2)
    axes[0, 1].set_title('Train Accuracy')
    axes[0, 1].set_xlabel('Training Iterations')
    axes[0, 1].set_ylabel('Accuracy')
    
    axes[1, 0].plot(epochs, hist['val_loss'], lw=2, color='tab:orange')
    axes[1, 0].set_title('Validation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    
    axes[1, 1].plot(epochs, hist['val_acc'], lw=2, color='tab:orange')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    
    for ax in axes.ravel():
        ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('training_{}.pdf'.format(timestamp), dpi=1200, format='pdf')
    # plt.show()

def csv_to_torch_loader(data, batch_size=256, is_imbalanced=False):
    X = data[['HR', 'HRV']].values            # shape (N, 2)
    y = data['label'].astype('int64').values  # CrossEntropy → int64 / lon
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=34, stratify=y)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test , dtype=torch.float32)
    y_test  = torch.tensor(y_test , dtype=torch.long)
    
    #  Wrap in TensorDataset for DataLoader
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_test , y_test )
    
    if is_imbalanced:
        class_counts  = torch.tensor([61_359, 24_850, 246_098])
        class_weights = (class_counts.sum() / class_counts).float()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=[class_weights[y] for y in y_train],
            num_samples=len(y_train),
            replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
        
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader

def build_optimizer(model: nn.Module, cfg: Dict) -> optim.Optimizer:
    opt_name = cfg.get("optimizer", "adamw").lower()

    if opt_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            momentum=cfg["momentum"],
        )

    if opt_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=tuple(cfg["betas"]),
        )

    return optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=tuple(cfg["betas"]),
    )

def train_nni(dataloader, model, device, criterion, optimizer, scheduler):
    model.train()
    for batch, (xb, yb) in enumerate(dataloader):
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

def val_nni(dataloader, model, device, criterion):
    model.eval()
    val_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            val_loss += criterion(preds, yb).item() * xb.size(0)
            v_correct += (preds.argmax(1) == yb).sum().item()
            v_total += xb.size(0)
    val_loss /= v_total
    val_acc  = v_correct / v_total
    return val_acc

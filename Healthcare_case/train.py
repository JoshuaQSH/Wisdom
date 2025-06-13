import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from torch.optim.lr_scheduler import OneCycleLR

from collections import defaultdict 
from tqdm import tqdm

import nni
from nni.utils import merge_parameter 

from model import *

class EarlyStopping:
    """
    Stops training when `val_loss` hasn’t improved by at least `min_delta`
    for `patience` consecutive epochs.  Restores the best weights.
    """
    def __init__(self, patience=10, min_delta=1e-4, ckpt_path="best.pt"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.ckpt_path  = ckpt_path
        self.best_loss  = float("+inf")
        self.bad_epochs = 0

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.bad_epochs = 0
            torch.save(model.state_dict(), self.ckpt_path)
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            model.load_state_dict(torch.load(self.ckpt_path))
            return True      # signal to stop
        return False

def _class_weights(loader, device):
    """Return   tensor([w0, w1, w2], device)   for CrossEntropyLoss."""
    y = torch.cat([labels for _, labels in loader])
    freq = torch.bincount(y).float()
    w = 1.0 / freq
    w = w * len(freq) / w.sum()
    return w.to(device)

def train_model(train_loader, val_loader, model, max_epochs=60, lr=3e-3, device='cpu'):
    
    early_stopping = EarlyStopping(patience=8, min_delta=1e-6)
    train_info = defaultdict(list)
    
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(weight=_class_weights(train_loader, device))
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)    
    scheduler  = OneCycleLR(
        optimizer,
        max_lr = lr * 10, 
        epochs = max_epochs,
        steps_per_epoch = len(train_loader),
        pct_start = 0.3,
        div_factor = 10
    )
    
    # SGD + CosineAnnealingLR
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    model.to(device)
    
    # Main training loop
    for epoch in tqdm(range(max_epochs), desc="Epochs"):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        # xb: (batch, input_dim); yb: class indices (LongTensor)
        for xb, yb in tqdm(train_loader, desc="Training", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            scheduler.step() # Update learning rate
            
            running_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(1) == yb).sum().item()
            total += xb.size(0)
        
            train_info['train_loss'].append(running_loss / total)
            train_info['train_acc'].append(correct / total)

        # Validation phase
        model.eval()
        val_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="Validation", leave=False):
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
                v_correct += (preds.argmax(1) == yb).sum().item()
                v_total += xb.size(0)

        val_loss /= v_total
        val_acc  = v_correct / v_total
        train_info['val_loss'].append(val_loss)
        train_info['val_acc'].append(val_acc)
        print(f"Epoch {epoch:03d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # Early-stop check
        if early_stopping.step(val_loss, model):
            print("Early stopping triggered – best weights restored.")
            break
        
    return model, train_info

def train_model_with_nni(train_loader, val_loader, model, device, optimizer, max_epochs, cfg):
    """
    cfg: lr, weight_decay
    `nni.report_intermediate_result()` to report per-epoch accuracy metrics.
    `nni.report_final_result()` to report final accuracy.
    """
    train_info = defaultdict(list)
    criterion = nn.CrossEntropyLoss()    
    scheduler = OneCycleLR(optimizer, 
                           max_lr=cfg['lr'] * 10,
                           epochs=max_epochs, 
                           steps_per_epoch=len(train_loader),
                           pct_start=0.3, 
                           div_factor=10)
    best_acc = 0.0
    for epoch in range(max_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        # xb: (batch, input_dim); yb: class indices (LongTensor)
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            scheduler.step() # Update learning rate
            
            running_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(1) == yb).sum().item()
            total += xb.size(0)
        
            train_info['train_loss'].append(running_loss / total)
            train_info['train_acc'].append(correct / total)
        
         # Validation phase
        model.eval()
        val_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
                v_correct += (preds.argmax(1) == yb).sum().item()
                v_total += xb.size(0)

        val_loss /= v_total
        val_acc  = v_correct / v_total
        train_info['val_loss'].append(val_loss)
        train_info['val_acc'].append(val_acc)
        best_acc = max(best_acc, val_acc)
        print(f"Epoch {epoch:03d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        nni.report_intermediate_result(val_acc)
        
    nni.report_final_result(val_acc)
    return train_info, best_acc
    
def model_info(model):
    torchinfo.summary(model, input_size=(1, 2), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20)


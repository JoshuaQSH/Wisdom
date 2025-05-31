import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo

from model import *

from tqdm import tqdm

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

def train_model(train_loader, val_loader, model, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    max_epochs = 40
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    model.to(device)
    for epoch in tqdm(range(max_epochs), desc="Epochs"):
        model.train()
        # xb: (batch, input_dim); yb: class indices (LongTensor)
        for xb, yb in tqdm(train_loader, desc="Training", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="Validation", leave=False):
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
                correct += (preds.argmax(1) == yb).sum().item()
                total += xb.size(0)

        val_loss /= total
        val_acc  = correct / total
        print(f"Epoch {epoch:03d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # --- early-stop check ---
        if early_stopping.step(val_loss, model):
            print("Early stopping triggered – best weights restored.")
            break

def model_info():
    model = MLP(input_dim=2)
    torchinfo.summary(model, input_size=(1, 2), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20)
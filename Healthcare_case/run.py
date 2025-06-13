import pandas as pd
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from model.model import *
from utils import csv_to_torch_loader, plot_history
from train import train_model

from sklearn.model_selection import train_test_split

PREPARE_CASE = ['healthcare', 'hrp', 'swell', 'stresspred']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    
def healthcare_case(data_path):
    print("Preparing healthcare case...")

    # Load the dataset
    data = pd.read_csv(data_path + '/Healthcare/hrv.csv')
    data.dropna(inplace=True)
    
    # Convert to PyTorch DataLoader    
    train_loader, val_loader = csv_to_torch_loader(data, 256, is_imbalanced=True)
    model = MLPv2(train_loader.dataset[0][0].shape[-1], p_dropout=0.10)
    model, train_info = train_model(train_loader=train_loader, val_loader=val_loader, model=model, max_epochs=60, lr=0.001, device=DEVICE)
    plot_history(train_info)

if __name__ == "__main__":
    # Example usage
    data_path = os.getcwd() + '/datasets'
    healthcare_case(data_path)

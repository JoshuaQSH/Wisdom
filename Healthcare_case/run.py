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
    
def healthcare_case(data_path):
    print("Preparing healthcare case...")

    # Load the dataset
    data = pd.read_csv(data_path + '/Healthcare/hrv.csv')
    data.dropna(inplace=True)
    
    # Convert to PyTorch DataLoader    
    train_loader, val_loader = csv_to_torch_loader(data, 256, is_imbalanced=False)
    model = MLPv2(train_loader.dataset[0][0].shape[-1], p_dropout=0.10)
    model, train_info = train_model(train_loader=train_loader, val_loader=val_loader, model=model, max_epochs=60, lr=0.001, device=DEVICE)
    plot_history(train_info)

if __name__ == "__main__":
    # Example usage
    data_path = os.getcwd() + '/datasets'
    healthcare_case(data_path)

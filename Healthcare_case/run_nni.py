import nni
from model.model import *
from utils import train_nni, val_nni, build_optimizer, csv_to_torch_loader
import pandas as pd
import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import OneCycleLR


PREPARE_CASE = ['healthcare', 'hrp', 'swell', 'stresspred']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def nni_tune(train_case='healthcare', data_path='./datasets', device=DEVICE):
    """
    `nni.get_next_parameter()` to fetch the hyperparameters to be evalutated.
    """
    params = dict(
            hidden1=16,
            hidden2=32,
            hidden3=16,
            dropout=0.3,
            optimizer="adamw",
            lr=3e-3, 
            weight_decay=2e-4, 
            betas=[0.9, 0.999],
            momentum=0.9,
    )
    
    batch_size = 256
    num_classes = 3
    max_epochs = 60
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    
    print("Preparing healthcare case...")
    data = pd.read_csv(data_path + '/Healthcare/hrv.csv')
    data.dropna(inplace=True)
    
    # Convert to PyTorch DataLoader    
    train_loader, val_loader = csv_to_torch_loader(data, batch_size, is_imbalanced=False)
    

    h1 = params['hidden1']
    h2 = params['hidden2']
    h3 = params['hidden3']
    model = nn.Sequential(
            nn.Linear(train_loader.dataset[0][0].shape[-1], h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(params['dropout']),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(params['dropout']),
            
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(inplace=True),
            nn.Dropout(params['dropout']),

            nn.Linear(h3, num_classes)
    ).to(device)
        
    # model = MLPv1(input_dim=train_loader.dataset[0][0].shape[-1], features=params['features'], dropout=params['dropout'], num_classes=num_classes)
    optimizer = build_optimizer(model, params)
    # optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], betas=tuple(params["betas"]))
    criterion = nn.CrossEntropyLoss()    
    scheduler = OneCycleLR(optimizer, 
                            max_lr=params['lr'] * 10,
                            epochs=max_epochs, 
                            steps_per_epoch=len(train_loader),
                            pct_start=0.3, 
                            div_factor=10)
        
    # Train the model
    for t in range(max_epochs):
        train_nni(train_loader, model, device, criterion, optimizer, scheduler)
        accuracy = val_nni(val_loader, model, device, criterion)
        nni.report_intermediate_result(accuracy)
    nni.report_final_result(accuracy)

if __name__ == "__main__":
    # Autotuning with NNI
    data_path = os.getcwd() + '/datasets'
    nni_tune(train_case=PREPARE_CASE[0], data_path=data_path, device=DEVICE)
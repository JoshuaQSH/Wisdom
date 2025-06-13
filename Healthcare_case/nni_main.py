from pathlib import Path
import signal
import pandas as pd
import os
import argparse

import nni
from nni.experiment import Experiment
from nni.nas.experiment import NasExperiment
import nni.nas.strategy as strategy
from nni.nas.evaluator import FunctionalEvaluator

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.model_selection import train_test_split

from model.nas_model import CardioMLP
from utils import train_nni, val_nni


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
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
        
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader

def nni_nas_evaluate(model):
    max_epochs = 60
    lr = 3e-3
    batch_size = 256
    data_path = os.getcwd() + '/datasets'
    data = pd.read_csv(data_path + '/Healthcare/hrv.csv')
    data.dropna(inplace=True)
    
    train_loader, val_loader = csv_to_torch_loader(data, batch_size, is_imbalanced=False)
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()    
    scheduler = OneCycleLR(optimizer, 
                            max_lr=lr * 10,
                            epochs=max_epochs, 
                            steps_per_epoch=len(train_loader),
                            pct_start=0.3, 
                            div_factor=10)
    for epoch in range(max_epochs):
        train_nni(train_loader, model, DEVICE, criterion, optimizer, scheduler)
        accuracy = val_nni(val_loader, model, DEVICE, criterion)
        nni.report_intermediate_result(accuracy)
    nni.report_final_result(accuracy)
    

# Define search space
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NNI experiment with optional NAS.')
    parser.add_argument('--nas', action='store_true',
                        help='Set to True if you are using NAS, False otherwise')
    args = parser.parse_args()

    if args.nas:
        search_strategy = strategy.Random()
        model_space = CardioMLP(2, 3)
        evaluator = FunctionalEvaluator(nni_nas_evaluate)
        exp = NasExperiment(model_space, evaluator, search_strategy)
        exp.config.max_trial_number = 100
        exp.config.trial_concurrency = 4
        # exp.config.trial_gpu_number = 0   # will not use GPU
        # exp.config.training_service.use_active_gpu = True
        exp.run(port=8080)
        print('Experiment is running. Press Ctrl-C to quit.')
        signal.pause()

    else:    
        search_space = {
            # ── model architecture parameters ─────────────────────────────
            'hidden1':  { '_type': 'choice', '_value': [16, 32, 64, 128, 256] },
            'hidden2':  { '_type': 'choice', '_value': [32, 64, 128, 256]},
            'hidden3':  { '_type': 'choice', '_value': [16, 32, 64]},
            'dropout': {'_type': 'uniform', '_value': [0.0, 0.5]},
            
            "optimizer": {
                "_type": "choice",
                "_value": ["adamw", "adam", "sgd"]
            },

            # ── shared optimiser hyper‑params ──────────────────────────────
            "lr": {
                "_type": "loguniform",
                "_value": [1e-4, 1e-1]
            },
            
            "weight_decay": {
                "_type": "loguniform",
                "_value": [1e-6, 1e-2]
            },

            # ── conditional: only when optimiser ∈ {"adam", "adamw"} ──────
            "betas": {
                "_type": "choice",
                "_value": [[0.9, 0.99], [0.9, 0.999], [0.95, 0.999]],
            },
            
            # ── conditional: only when optimiser == "sgd" ─────────────────
            "momentum": {
                "_type": "uniform",
                "_value": [0.0, 0.99],
                "_condition": {"optimizer": "sgd"}
            }
        }

        # Configure experiment
        experiment = Experiment('local')
        experiment.config.trial_command = 'python run_nni.py'
        experiment.config.trial_code_directory = Path(__file__).parent
        experiment.config.search_space = search_space
        experiment.config.tuner.name = 'TPE'
        experiment.config.max_trial_number = 100
        experiment.config.trial_concurrency = 4

        # Assessor and global limits configuration
        experiment.config.assessor.name = "Medianstop"
        experiment.config.assessor.class_args = {
            "optimize_mode": "maximize",
            "start_step": 5
        }

        # Run it!
        experiment.run(port=8080, wait_completion=False)

        print('Experiment is running. Press Ctrl-C to quit.')
        signal.pause()
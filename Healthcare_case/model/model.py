import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# MLP for HH case
# ---------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, 4)
        self.linear3 = nn.Linear(4, 8)
        self.linear4 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x
    
# ---------------------------
# MLP (fine-tuning version) for HH case
# ---------------------------
class MLPv1(nn.Module):
    def __init__(self, input_dim: int, features: int, dropout: float = 0.5, num_classes: int = 3) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_dim, features),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(features, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    
# ---------------------------
# An updated version of MLP for HH case
# ---------------------------
class MLPv2(nn.Module):
    def __init__(self, input_dim: int, p_dropout: float = 0.50):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 16)
        self.linear6 = nn.Linear(16, 3)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.batchnorm4 = nn.BatchNorm1d(32)
        self.batchnorm5 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.linear4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.linear5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear6(x)
        
        return x
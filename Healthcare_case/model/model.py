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
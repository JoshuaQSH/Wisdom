import torch
import torch.nn as nn
import torch.nn.functional as F

class DaveOrig(nn.Module):
    def __init__(self):
        super(DaveOrig, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(64*2*2, 1164)  # Update this based on the actual flattened size
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = torch.atan(x) * 2  # Apply atan_layer equivalent
        return x

# Instantiate the model
model = DaveOrig()
print(model)

# Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adadelta(model.parameters())

# # Example of how to perform a forward pass and compute loss
# input_tensor = torch.randn(1, 3, 100, 100)  # Example input tensor
# output = model(input_tensor)
# loss = criterion(output, torch.tensor([[1.0]]))  # Example target
# print(loss.item())

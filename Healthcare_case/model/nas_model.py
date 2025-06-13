import nni
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear
import torch.nn as nn

class CardioMLP(ModelSpace):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        # global choices reused across blocks
        self.activation = LayerChoice(
            [nn.ReLU(), nn.GELU(), nn.SiLU()],
            label='activation'
        )
        self.dropout = MutableDropout(nni.choice('dropout', [0.1, 0.2, 0.3, 0.4, 0.5]))

        feature = nni.choice('feature', [16, 32, 64, 128, 256])
        hidden_1 = nni.choice('hidden1', [16, 32, 64, 128, 256])
        hidden_2 = nni.choice('hidden2', [16, 32, 64, 128, 256])
        hidden_3 = nni.choice('hidden3', [16, 32, 64, 128, 256])
        
        self.fc1 = MutableLinear(input_dim, feature)
        self.fc2 = MutableLinear(feature, hidden_1)
        self.fc3 = MutableLinear(hidden_1, hidden_2)
        self.fc4 = MutableLinear(hidden_2, hidden_3)
        self.fc_out = MutableLinear(hidden_3, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation.clone()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation.clone()(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation.clone()(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.activation.clone()(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x
        
import torch
from torch import nn
from torchvision import models

class SimpleCNN(nn.Module):
    """Three‑layer CNN roughly matching the paper's baseline."""

    def __init__(self, num_classes: int = 4, image_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class DCNN(nn.Module):
    """VGG16 backbone with three extra conv layers, referring to the paper."""

    def __init__(self, num_classes: int = 4, 
                 conv_filters: tuple[int, int, int] = (512, 512, 256),
                 dense_units: int = 256):
        super().__init__()
        
        # Load the VGG16 model and use its features
        # Note: VGG16 is pretrained on ImageNet, which is suitable for transfer learning
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.base = nn.Sequential(*list(base.features.children())[:-1])
        
        # Freeze the base model parameters
        for p in self.base.parameters():
            p.requires_grad = False


        layers: list[nn.Module] = []
        in_ch = 512
        for f in conv_filters:
            layers += [
                nn.Conv2d(in_ch, f, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = f
        self.extra = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x):
        x = self.base(x)
        x = self.extra(x)
        x = self.classifier(x)
        return x
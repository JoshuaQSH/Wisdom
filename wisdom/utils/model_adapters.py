from __future__ import annotations

import torch
import torch.nn as nn


_VGG_CFG = {
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}


class CIFARVGG(nn.Module):
    """VGG variant used by common CIFAR pretrained checkpoints."""

    def __init__(self, vgg_name: str = "VGG16", num_classes: int = 100):
        super().__init__()
        self.features = self._make_layers(_VGG_CFG[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for item in cfg:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend(
                    [
                        nn.Conv2d(in_channels, int(item), kernel_size=3, padding=1),
                        nn.BatchNorm2d(int(item)),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = int(item)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class CIFARInputAdapter(nn.Module):
    """Adapt repo-normalized CIFAR tensors to an external model's input scale."""

    def __init__(
        self,
        model: nn.Module,
        target_mean: tuple[float, float, float] | None = None,
        target_std: tuple[float, float, float] | None = None,
        input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        input_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        super().__init__()
        self.model = model
        self.register_buffer("input_mean", torch.tensor(input_mean).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(input_std).view(1, 3, 1, 1))
        if target_mean is None or target_std is None:
            self.target_mean = None
            self.target_std = None
        else:
            self.register_buffer("target_mean", torch.tensor(target_mean).view(1, 3, 1, 1))
            self.register_buffer("target_std", torch.tensor(target_std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x * self.input_std + self.input_mean
        if self.target_mean is None or self.target_std is None:
            return self.model(raw)
        normalized = (raw - self.target_mean) / self.target_std
        return self.model(normalized)

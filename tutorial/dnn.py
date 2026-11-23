"""Tabular quantile DNN with target normalization and pinball-loss training.

``prediction`` preserves the module's train/eval state: callers enable training
mode for MC dropout and evaluation mode for deterministic baseline quantiles.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


DEFAULT_HIDDEN_LAYERS = (64, 32, 16)
DEFAULT_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 0.003


class LinearDropout(nn.Sequential):
    """Three hidden layers and a two-quantile output, in native target units.

    The historical class name and sequential parameter names are retained for
    state-dictionary compatibility.  Defaults reproduce the four-linear-layer
    network exactly; architecture overrides are explicit tuning options.
    """

    def __init__(self, n_features, center=0.0, scale=1.0, *,
                 hidden_layers=DEFAULT_HIDDEN_LAYERS, dropout=DEFAULT_DROPOUT):
        widths = tuple(int(width) for width in hidden_layers)
        if int(n_features) < 1 or not widths or min(widths) < 1:
            raise ValueError("Need positive input and hidden-layer dimensions")
        if not math.isfinite(float(dropout)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        layers, previous = [], int(n_features)
        for width in widths:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, 2))
        super().__init__(*layers)
        self.center, self.scale = center, scale

    @torch.no_grad()
    def prediction(self, data):
        """Return ordered lower/upper quantiles without changing dropout mode."""
        bounds = self(torch.as_tensor(data, dtype=torch.float32)).cpu().numpy()
        return np.sort(bounds * self.scale + self.center, axis=1)


def _training_options(args=None, *, epochs=None, alpha=None, learning_rate=None):
    epochs = getattr(args, "epochs", 500) if epochs is None else epochs
    alpha = getattr(args, "alpha", 0.1) if alpha is None else alpha
    if learning_rate is None:
        learning_rate = getattr(args, "learning_rate", None)
        if learning_rate is None:
            learning_rate = DEFAULT_LEARNING_RATE
    if not isinstance(epochs, (int, np.integer)) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(float(alpha)) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between zero and one")
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    return int(epochs), float(alpha), float(learning_rate)


def fit_standard(x_train, y_train, seed=42, args=None, *, epochs=None,
                 alpha=None, learning_rate=None, model_kwargs=None):
    """Fit the original row-weighted pinball DNN and return training metadata.

    ``args`` accepts the historical runner namespace.  Explicit keyword options
    support the consolidated entry point without depending on a CLI module.
    Feature imputation/scaling must be fitted by the calling training workflow.
    """
    epochs, alpha, learning_rate = _training_options(
        args, epochs=epochs, alpha=alpha, learning_rate=learning_rate)
    x_train, y_train = np.asarray(x_train), np.asarray(y_train)
    if x_train.ndim != 2 or y_train.ndim != 1 or len(x_train) != len(y_train) or not len(y_train):
        raise ValueError("Need a nonempty training feature matrix and matching target vector")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError("Impute nonfinite features before DNN training")
    torch.manual_seed(int(seed))
    center, scale = y_train.mean(), max(y_train.std(), 1e-8)
    model = LinearDropout(x_train.shape[1], center, scale, **(model_kwargs or {}))
    inputs = torch.as_tensor(x_train, dtype=torch.float32)
    targets = torch.as_tensor((y_train - center) / scale, dtype=torch.float32)[:, None]
    quantiles = torch.tensor([alpha / 2, 1 - alpha / 2])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        error = targets - model(inputs)
        loss = torch.maximum(quantiles * error, (quantiles - 1) * error).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model, {
        "final_pinball": float(loss.detach()),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "target_center": float(center),
        "target_scale": float(scale),
    }


# A descriptive alias is convenient in registries; the legacy name remains the
# actual class, so existing sequential state dictionaries keep their key layout.
QuantileDNN = LinearDropout


__all__ = ["LinearDropout", "QuantileDNN", "fit_standard"]

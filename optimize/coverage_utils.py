"""
coverage_utils.py
=================
Layer-stratified coverage computation for YOLO backbone neurons.

Key improvements over the original RQ scripts:
1. **Stratified layers**: early / middle / late groups with separate coverage
2. **More neurons monitored**: top-100 (not top-20) from WISDOM scores
3. **Adaptive thresholds**: per-layer threshold calibrated from clean data
4. **Coverage variability**: std-dev across layers as a discrimination signal
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn
import numpy as np


# ── Layer grouping ──────────────────────────────────────────────────
# YOLO11n backbone layers numbered model.0 … model.22
# We split by depth index:
#   early  = model.0 – model.5   (low-level: edges, colours, textures)
#   middle = model.6 – model.12  (mid-level: parts, motifs)
#   late   = model.13 – model.22 (high-level: objects, context, neck)

# For YOLOb11n so far
def _layer_group(layer_name: str) -> str:
    """Return 'early', 'middle', 'late' based on the model.X index."""
    import re
    m = re.search(r"model\.(\d+)", layer_name)
    if m is None:
        return "late"
    idx = int(m.group(1))
    if idx <= 5:
        return "early"
    elif idx <= 12:
        return "middle"
    else:
        return "late"


def load_top_neurons(
    csv_path: str,
    top_k: int = 100,
    strip_prefix: str = "yolo_model.",
) -> Dict[str, List[int]]:
    """
    Load top-K neurons from a WISDOM scores CSV.

    Returns {raw_layer_name: [neuron_indices]} with the wrapper prefix
    stripped so names match the raw YOLO model.
    """
    df = pd.read_csv(csv_path)
    top_df = df.nlargest(top_k, "Score")
    neurons: Dict[str, List[int]] = {}
    for lname, grp in top_df.groupby("LayerName"):
        raw = lname.replace(strip_prefix, "") if strip_prefix else lname
        neurons[raw] = sorted(grp["NeuronIndex"].tolist())
    return neurons


def load_layerwise_top_neurons(
    csv_path: str,
    per_layer_k: int = 5,
    strip_prefix: str = "yolo_model.",
) -> Dict[str, List[int]]:
    """
    Load top-K neurons **per layer** (layer-wise selection).

    This ensures every scored layer contributes neurons, avoiding the
    early-layer dominance problem of global top-K.
    """
    df = pd.read_csv(csv_path)
    neurons: Dict[str, List[int]] = {}
    for lname, grp in df.groupby("LayerName"):
        raw = lname.replace(strip_prefix, "") if strip_prefix else lname
        top = grp.nlargest(per_layer_k, "Score")
        top = top[top["Score"] > 0]
        if len(top) > 0:
            neurons[raw] = sorted(top["NeuronIndex"].tolist())
    return neurons


# ── Activation collection ───────────────────────────────────────────

class ActivationCollector:
    """
    Efficient hook-based activation collector for specific neurons.

    Supports both per-neuron and full-channel collection.
    """

    def __init__(
        self,
        model: nn.Module,
        target_neurons: Dict[str, List[int]],
        device: str = "cuda:0",
    ):
        self.model = model
        self.target_neurons = target_neurons
        self.device = device
        self._handles: list = []
        self._activations: Dict[str, torch.Tensor] = {}

    def _make_hook(self, layer_name: str, indices: List[int]):
        def hook_fn(module, input, output):
            if output.dim() == 4:
                # Conv: (B, C, H, W) → mean over spatial → pick indices
                spatial_mean = output.mean(dim=(2, 3))  # (B, C)
                self._activations[layer_name] = spatial_mean[:, indices].detach()
            elif output.dim() == 2:
                self._activations[layer_name] = output[:, indices].detach()
        return hook_fn

    def attach(self):
        self.detach()
        for name, mod in self.model.named_modules():
            if name in self.target_neurons:
                indices = self.target_neurons[name]
                h = mod.register_forward_hook(self._make_hook(name, indices))
                self._handles.append(h)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def collect(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward pass and return {layer: (B, n_neurons)} activations."""
        self._activations.clear()
        with torch.no_grad():
            self.model(images.to(self.device))
        return dict(self._activations)


# ── Coverage computation ────────────────────────────────────────────

def calibrate_thresholds(
    model: nn.Module,
    target_neurons: Dict[str, List[int]],
    calibration_images: torch.Tensor,
    device: str = "cuda:0",
    percentile: float = 25.0,
) -> Dict[str, torch.Tensor]:
    """
    Calibrate per-neuron activation thresholds from clean data.

    Uses the P-th percentile of activations so that ~(100-P)% of
    neurons are "active" on clean data — leaving room for perturbation
    to shift coverage.
    """
    collector = ActivationCollector(model, target_neurons, device)
    collector.attach()

    # Accumulate activations
    all_acts: Dict[str, list] = {k: [] for k in target_neurons}
    bs = 4
    for i in range(0, len(calibration_images), bs):
        batch = calibration_images[i:i + bs]
        acts = collector.collect(batch)
        for k, v in acts.items():
            all_acts[k].append(v.cpu())
    collector.detach()

    thresholds: Dict[str, torch.Tensor] = {}
    for k, v_list in all_acts.items():
        cat = torch.cat(v_list, dim=0)  # (N_images, n_neurons)
        # Per-neuron percentile threshold
        thr = torch.quantile(cat.float().abs(), percentile / 100.0, dim=0)
        thresholds[k] = thr
    return thresholds


def compute_stratified_coverage(
    activations: Dict[str, torch.Tensor],
    thresholds: Dict[str, torch.Tensor],
    target_neurons: Dict[str, List[int]],
) -> Dict[str, float]:
    """
    Compute coverage per layer-group and overall.

    Returns dict with keys: 'early', 'middle', 'late', 'overall',
    'variability' (std across groups).
    """
    group_active: Dict[str, int] = {"early": 0, "middle": 0, "late": 0}
    group_total: Dict[str, int] = {"early": 0, "middle": 0, "late": 0}

    for layer_name in target_neurons:
        if layer_name not in activations:
            continue
        act = activations[layer_name]  # (B, n_neurons)
        thr = thresholds.get(layer_name, torch.zeros(act.shape[1]))
        thr = thr.to(act.device)

        # A neuron is "active" if ANY sample in the batch exceeds threshold
        max_act = act.abs().max(dim=0).values  # (n_neurons,)
        active = (max_act > thr).sum().item()
        total = act.shape[1]

        grp = _layer_group(layer_name)
        group_active[grp] += active
        group_total[grp] += total

    coverages = {}
    for g in ("early", "middle", "late"):
        if group_total[g] > 0:
            coverages[g] = group_active[g] / group_total[g]
        else:
            coverages[g] = 0.0

    total_active = sum(group_active.values())
    total_neurons = sum(group_total.values())
    coverages["overall"] = total_active / max(total_neurons, 1)

    grp_vals = [coverages[g] for g in ("early", "middle", "late") if group_total[g] > 0]
    coverages["variability"] = float(np.std(grp_vals)) if len(grp_vals) > 1 else 0.0

    return coverages


def compute_magnitude_change(
    acts_clean: Dict[str, torch.Tensor],
    acts_perturbed: Dict[str, torch.Tensor],
    target_neurons: Dict[str, List[int]],
) -> Dict[str, float]:
    """
    Compute mean relative activation change per layer group.

    Unlike threshold-based coverage, this measures HOW MUCH activations
    shift, capturing subtle changes that don't cross a binary threshold.
    """
    group_change: Dict[str, list] = {"early": [], "middle": [], "late": []}

    for layer_name in target_neurons:
        if layer_name not in acts_clean or layer_name not in acts_perturbed:
            continue
        clean = acts_clean[layer_name].float()   # (B, n_neurons)
        pert = acts_perturbed[layer_name].float()
        # Relative change per neuron (mean over batch)
        diff = (pert - clean).abs().mean(dim=0)  # (n_neurons,)
        base = clean.abs().mean(dim=0).clamp(min=1e-8)
        rel_change = (diff / base).cpu().numpy()

        grp = _layer_group(layer_name)
        group_change[grp].extend(rel_change.tolist())

    result = {}
    for g in ("early", "middle", "late"):
        result[g] = float(np.mean(group_change[g])) if group_change[g] else 0.0
    all_vals = sum(group_change.values(), [])
    result["overall"] = float(np.mean(all_vals)) if all_vals else 0.0
    grp_vals = [result[g] for g in ("early", "middle", "late") if group_change[g]]
    result["variability"] = float(np.std(grp_vals)) if len(grp_vals) > 1 else 0.0
    return result


def compute_flat_coverage(
    activations: Dict[str, torch.Tensor],
    thresholds: Dict[str, torch.Tensor],
) -> float:
    """Simple overall coverage fraction (no stratification)."""
    total_active = 0
    total_neurons = 0
    for layer_name, act in activations.items():
        thr = thresholds.get(layer_name, torch.zeros(act.shape[1]))
        thr = thr.to(act.device)
        max_act = act.abs().max(dim=0).values
        total_active += (max_act > thr).sum().item()
        total_neurons += act.shape[1]
    return total_active / max(total_neurons, 1)

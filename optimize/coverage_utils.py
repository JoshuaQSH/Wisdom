"""
coverage_utils.py
=================
Layer-stratified coverage computation for YOLO backbone neurons.

Supports two coverage modes controlled by ``coverage_mode``:

- **"plain"** (default): Binary threshold coverage — a neuron is "active"
  if its activation exceeds a calibrated percentile threshold.  Fast but
  loses the combinatorial power of the original WISDOM.

- **"cluster"**: Combinatorial cluster coverage — each neuron's activation
  is clustered (KMeans/MeanShift) on a build set, then test samples are
  assigned to their nearest cluster.  Coverage = |unique cluster-state
  tuples| / ∏(cluster sizes per neuron).  This is faithful to the
  original WISDOM IDC methodology.

Key improvements over the original RQ scripts:
1. **Stratified layers**: early / middle / late groups with separate coverage
2. **More neurons monitored**: top-100 (not top-20) from WISDOM scores
3. **Adaptive thresholds**: per-layer threshold calibrated from clean data
4. **Coverage variability**: std-dev across layers as a discrimination signal
5. **Cluster coverage**: optional combinatorial coverage via WISDOM clustering
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


# ── Clustering-based combinatorial coverage ─────────────────────────

class ClusterCoverageComputer:
    """
    Wraps WISDOM's clustering pipeline for combinatorial coverage.

    Workflow:
      1. ``fit(build_images)`` — collect activations on a build set and
         cluster each neuron independently (KMeans by default).
      2. ``coverage(test_images)`` — for each test image, assign every
         monitored neuron to its nearest cluster, then count how many
         unique cluster-state *combinations* were observed.
         Coverage = |seen tuples| / ∏(C_i).

    The build set should be a representative sample of clean images.
    """

    def __init__(
        self,
        model: nn.Module,
        target_neurons: Dict[str, List[int]],
        device: str = "cuda:0",
        method: str = "KMeans",
        use_silhouette: bool = True,
        k_max: int = 5,
        n_clusters: int = 3,
    ):
        self.model = model
        self.target_neurons = target_neurons
        self.device = device
        self.method = method
        self.use_silhouette = use_silhouette
        self.k_max = k_max
        self.n_clusters = n_clusters

        # Populated by fit()
        self.groups: Dict[str, Dict[int, dict]] = {}
        self.cluster_sizes: Dict[str, int] = {}
        self._fitted = False

    def fit(self, build_images: torch.Tensor, batch_size: int = 4) -> None:
        """Cluster each monitored neuron's activations on the build set."""
        from wisdom.clustering.assign import fit_per_neuron

        collector = ActivationCollector(self.model, self.target_neurons, self.device)
        collector.attach()

        # Collect per-neuron activation series
        series: Dict[str, Dict[int, list]] = {
            l: {i: [] for i in idxs}
            for l, idxs in self.target_neurons.items()
        }

        for i in range(0, len(build_images), batch_size):
            batch = build_images[i:i + batch_size]
            acts = collector.collect(batch)
            for lname, act_t in acts.items():
                cpu = act_t.detach().cpu()
                for col, idx in enumerate(self.target_neurons[lname]):
                    series[lname][idx].extend(cpu[:, col].tolist())
        collector.detach()

        # Convert to numpy
        np_series: Dict[str, Dict[int, np.ndarray]] = {}
        for lname, dct in series.items():
            np_series[lname] = {
                idx: np.asarray(vals, dtype=np.float64)
                for idx, vals in dct.items()
            }

        # Fit per-neuron clustering
        params = {"n_clusters": self.n_clusters, "random_state": 42}
        self.groups = fit_per_neuron(
            np_series,
            method=self.method,
            params=params,
            use_silhouette=self.use_silhouette,
            k_max=self.k_max,
        )

        # Build cluster_sizes keyed as "layer:neuron_idx"
        self.cluster_sizes = {}
        for l in self.groups:
            for idx in self.groups[l]:
                key = f"{l}:{idx}"
                self.cluster_sizes[key] = self.groups[l][idx]["centers"].shape[0]
        self._fitted = True

    def _assign_single(
        self, sample_acts: Dict[str, Dict[int, float]]
    ) -> Dict[str, int]:
        """Assign one sample's activations to cluster ids, returning flat dict."""
        from wisdom.clustering.assign import assign_clusters
        assn = assign_clusters(self.groups, sample_acts)
        return {f"{l}:{i}": assn[l][i] for l in assn for i in assn[l]}

    def coverage(
        self, test_images: torch.Tensor, batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Compute per-layer combinatorial cluster coverage, then aggregate.

        With per_layer_k=5 neurons and ~3 clusters each, each layer has
        3^5 = 243 possible combinations — tractable and discriminating.
        We compute coverage per layer, then average within each group
        (early/middle/late) and overall.

        Returns dict with 'early', 'middle', 'late', 'overall',
        'variability'.
        """
        assert self._fitted, "Call fit() before coverage()"
        from wisdom.core.compute import combinations_coverage

        collector = ActivationCollector(self.model, self.target_neurons, self.device)
        collector.attach()

        # Collect per-sample, per-layer assignments
        # layer_assignments[layer_name] = list of dicts, one per image
        layer_assignments: Dict[str, list] = {l: [] for l in self.target_neurons}
        layer_sizes: Dict[str, Dict[str, int]] = {l: {} for l in self.target_neurons}

        for i in range(0, len(test_images), batch_size):
            batch = test_images[i:i + batch_size]
            acts = collector.collect(batch)
            B = batch.shape[0]
            for b in range(B):
                sample_acts: Dict[str, Dict[int, float]] = {}
                for lname in self.target_neurons:
                    if lname not in acts:
                        continue
                    sample_acts[lname] = {}
                    for col, idx in enumerate(self.target_neurons[lname]):
                        sample_acts[lname][idx] = float(acts[lname][b, col].item())

                flat = self._assign_single(sample_acts)

                # Split assignments by layer
                for key, cluster_id in flat.items():
                    lname = key.rsplit(":", 1)[0]
                    if lname not in layer_assignments:
                        continue
                    # Lazy-init per-image dict for this layer
                layer_dicts: Dict[str, Dict[str, int]] = {l: {} for l in self.target_neurons}
                for key, cluster_id in flat.items():
                    lname = key.rsplit(":", 1)[0]
                    if lname in layer_dicts:
                        layer_dicts[lname][key] = cluster_id
                        layer_sizes[lname][key] = self.cluster_sizes[key]

                for lname in self.target_neurons:
                    if layer_dicts[lname]:
                        layer_assignments[lname].append(layer_dicts[lname])

        collector.detach()

        # Compute per-layer combinatorial coverage
        layer_covs: Dict[str, float] = {}
        for lname in self.target_neurons:
            if layer_assignments[lname] and layer_sizes[lname]:
                r, t, _ = combinations_coverage(
                    layer_assignments[lname], layer_sizes[lname]
                )
                layer_covs[lname] = r
            else:
                layer_covs[lname] = 0.0

        # Aggregate by group (early / middle / late)
        group_covs: Dict[str, list] = {"early": [], "middle": [], "late": []}
        for lname, cov in layer_covs.items():
            grp = _layer_group(lname)
            group_covs[grp].append(cov)

        coverages: Dict[str, float] = {}
        all_covs = []
        for g in ("early", "middle", "late"):
            if group_covs[g]:
                coverages[g] = float(np.mean(group_covs[g]))
                all_covs.extend(group_covs[g])
            else:
                coverages[g] = 0.0

        coverages["overall"] = float(np.mean(all_covs)) if all_covs else 0.0
        grp_vals = [coverages[g] for g in ("early", "middle", "late") if coverages[g] > 0]
        coverages["variability"] = float(np.std(grp_vals)) if len(grp_vals) > 1 else 0.0
        return coverages

    def coverage_delta(
        self,
        clean_images: torch.Tensor,
        perturbed_images: torch.Tensor,
        batch_size: int = 4,
    ) -> Dict[str, float]:
        """
        Compute coverage change between clean and perturbed image sets.

        Returns dict with delta_early, delta_middle, delta_late,
        delta_overall, delta_variability.
        """
        cov_clean = self.coverage(clean_images, batch_size)
        cov_pert = self.coverage(perturbed_images, batch_size)
        delta = {}
        for k in ("early", "middle", "late", "overall", "variability"):
            delta[f"clean_{k}"] = cov_clean[k]
            delta[f"pert_{k}"] = cov_pert[k]
            delta[f"delta_{k}"] = abs(cov_pert[k] - cov_clean[k])
        return delta

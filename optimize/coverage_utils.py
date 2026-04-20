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

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger("wisdom_opt")


# ── Layer grouping ──────────────────────────────────────────────────
# YOLO11n backbone layers numbered model.0 … model.22
# Default 3-group split:
#   early  = model.0 – model.5   (low-level: edges, colours, textures)
#   middle = model.6 – model.12  (mid-level: parts, motifs)
#   late   = model.13 – model.22 (high-level: objects, context, neck)

_GROUP_PRESETS: Dict[int, Dict[str, Tuple[int, int]]] = {
    2: {"front": (0, 11), "back": (12, 22)},
    3: {"early": (0, 5), "middle": (6, 12), "late": (13, 22)},
    4: {"early": (0, 4), "mid_early": (5, 9), "mid_late": (10, 15), "late": (16, 22)},
    5: {"early": (0, 3), "mid_early": (4, 7), "middle": (8, 12), "mid_late": (13, 17), "late": (18, 22)},
}

_ACTIVE_GROUPS: Dict[str, Tuple[int, int]] = dict(_GROUP_PRESETS[3])


def set_n_groups(n: int) -> None:
    """Switch the layer-grouping scheme. *n* must be 2, 3, 4 or 5."""
    global _ACTIVE_GROUPS
    if n not in _GROUP_PRESETS:
        raise ValueError(f"n_groups must be one of {sorted(_GROUP_PRESETS)}, got {n}")
    _ACTIVE_GROUPS = dict(_GROUP_PRESETS[n])


def set_custom_groups(groups: Dict[str, Tuple[int, int]]) -> None:
    """Set custom (asymmetric) layer group boundaries.

    *groups* maps group name → (lo_model_idx, hi_model_idx) inclusive.
    Example: ``{"early": (0, 3), "middle": (4, 15), "late": (16, 22)}``
    """
    global _ACTIVE_GROUPS
    _ACTIVE_GROUPS = dict(groups)


def get_group_names() -> Tuple[str, ...]:
    """Return the currently active group names in order."""
    return tuple(_ACTIVE_GROUPS.keys())


def _layer_group(layer_name: str) -> str:
    """Return the group name for *layer_name* based on its model.X index."""
    import re
    m = re.search(r"model\.(\d+)", layer_name)
    if m is None:
        return list(_ACTIVE_GROUPS.keys())[-1]
    idx = int(m.group(1))
    for gname, (lo, hi) in _ACTIVE_GROUPS.items():
        if lo <= idx <= hi:
            return gname
    return list(_ACTIVE_GROUPS.keys())[-1]


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


def load_groupwise_top_neurons(
    csv_path: str,
    per_group_k: int = 5,
    strip_prefix: str = "yolo_model.",
) -> Dict[str, List[int]]:
    """
    Load top-K neurons **per layer group** (early/middle/late).

    Unlike per-layer selection (which picks k neurons from each of ~60
    layers), this picks the k BEST neurons across all layers within each
    group.  This means the selected neurons may span multiple layers
    within a group, and the combinatorial coverage is computed over
    cross-layer neuron interactions — more faithful to the original
    WISDOM IDC methodology.

    Returns {raw_layer_name: [neuron_indices]}.
    """
    df = pd.read_csv(csv_path)
    grouped: Dict[str, list] = {g: [] for g in get_group_names()}

    for _, row in df.iterrows():
        lname = row["LayerName"]
        raw = lname.replace(strip_prefix, "") if strip_prefix else lname
        grp = _layer_group(raw)
        score = float(row["Score"])
        if score > 0:
            grouped[grp].append((raw, int(row["NeuronIndex"]), score))

    neurons: Dict[str, List[int]] = {}
    for g in get_group_names():
        top = sorted(grouped[g], key=lambda x: x[2], reverse=True)[:per_group_k]
        for lname, idx, _ in top:
            neurons.setdefault(lname, []).append(idx)

    return {l: sorted(idxs) for l, idxs in neurons.items()}


def score_layers(
    csv_path: str,
    method: str = "mean_positive",
    strip_prefix: str = "yolo_model.",
) -> List[Tuple[str, float, int, int]]:
    """
    Score each layer by aggregate neuron importance.

    Returns sorted list of (layer_name, score, n_positive, n_total)
    in descending order of *score*.

    *method* controls the aggregation:
      - ``"mean_positive"``: mean of neuron scores where Score > 0
      - ``"sum"``: sum of all neuron scores
      - ``"max"``: max neuron score in the layer
    """
    df = pd.read_csv(csv_path)
    results: List[Tuple[str, float, int, int]] = []
    for lname, grp in df.groupby("LayerName"):
        raw = lname.replace(strip_prefix, "") if strip_prefix else lname
        scores = grp["Score"].values
        positives = scores[scores > 0]
        n_total = len(scores)
        n_pos = len(positives)
        if n_pos == 0:
            results.append((raw, 0.0, 0, n_total))
            continue
        if method == "mean_positive":
            agg = float(positives.mean())
        elif method == "sum":
            agg = float(scores.sum())
        elif method == "max":
            agg = float(scores.max())
        else:
            raise ValueError(f"Unknown layer scoring method: {method}")
        results.append((raw, agg, n_pos, n_total))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def load_important_layer_neurons(
    csv_path: str,
    top_layers: int = 10,
    per_layer_k: int = 5,
    layer_score_method: str = "mean_positive",
    strip_prefix: str = "yolo_model.",
) -> Dict[str, List[int]]:
    """
    Two-stage neuron selection: important layers first, then top neurons.

    Stage 1: Score all layers by aggregate neuron importance, keep top-L.
    Stage 2: From each selected layer, pick top-k neurons by score.

    This concentrates monitoring on the layers where WISDOM importance
    is highest, avoiding thin coverage spread across unimportant layers.
    """
    ranked = score_layers(csv_path, method=layer_score_method,
                          strip_prefix=strip_prefix)

    # Filter to layers with at least 1 positive-score neuron, take top-L
    ranked_pos = [(l, s, np_, nt) for l, s, np_, nt in ranked if s > 0]
    selected = ranked_pos[:top_layers]
    selected_names = {l for l, _, _, _ in selected}

    logger.info("")
    logger.info("  ╔══════════════════════════════════════════════════════╗")
    logger.info("  ║     LAYER IMPORTANCE SELECTION                      ║")
    logger.info("  ║  Method: %-15s  Top-L: %-3d  k: %-3d     ║",
                layer_score_method, top_layers, per_layer_k)
    logger.info("  ╠══════════════════════════════════════════════════════╣")
    logger.info("  ║  Rank  Layer                           Score  #pos  ║")
    for i, (l, s, np_, nt) in enumerate(ranked_pos[:top_layers + 5]):
        marker = " ✓" if l in selected_names else "  "
        logger.info("  ║  %3d%s %-35s %8.1f %3d/%-3d ║",
                     i + 1, marker, l[:35], s, np_, nt)
    n_skipped = len(ranked) - len(ranked_pos)
    if n_skipped > 0:
        logger.info("  ║  ... %d layers with zero score (skipped)          ║",
                     n_skipped)
    logger.info("  ╚══════════════════════════════════════════════════════╝")

    # Stage 2: top-k neurons per selected layer
    df = pd.read_csv(csv_path)
    neurons: Dict[str, List[int]] = {}
    for lname, grp in df.groupby("LayerName"):
        raw = lname.replace(strip_prefix, "") if strip_prefix else lname
        if raw not in selected_names:
            continue
        top = grp.nlargest(per_layer_k, "Score")
        top = top[top["Score"] > 0]
        if len(top) > 0:
            neurons[raw] = sorted(top["NeuronIndex"].tolist())

    total = sum(len(v) for v in neurons.values())
    logger.info("  Selected %d neurons across %d layers (from %d total layers)",
                total, len(neurons), len(ranked))
    return neurons


def load_hybrid_layer_neurons(
    layer_csv: str,
    neuron_csv: str,
    top_layers: int = 10,
    per_layer_k: int = 5,
    layer_score_method: str = "mean_positive",
    strip_prefix: str = "yolo_model.",
) -> Dict[str, List[int]]:
    """
    Hybrid two-stage selection: rank layers from one CSV, pick neurons from another.

    Stage 1: Score layers using *layer_csv* (e.g. WISDOM consensus scores).
    Stage 2: Pick top-k neurons per selected layer using *neuron_csv* (e.g. lgxa).

    This combines WISDOM's layer-level judgment (consensus voting captures
    which layers are structurally important) with lgxa's neuron-level ranking
    (faster, potentially better within-layer discrimination).
    """
    ranked = score_layers(layer_csv, method=layer_score_method,
                          strip_prefix=strip_prefix)
    ranked_pos = [(l, s, np_, nt) for l, s, np_, nt in ranked if s > 0]
    selected = ranked_pos[:top_layers]
    selected_names = {l for l, _, _, _ in selected}

    logger.info("")
    logger.info("  ╔══════════════════════════════════════════════════════╗")
    logger.info("  ║     HYBRID LAYER+NEURON SELECTION                   ║")
    logger.info("  ║  Layer scoring: %-15s (from layer CSV)   ║", layer_score_method)
    logger.info("  ║  Neuron scoring: from neuron CSV                    ║")
    logger.info("  ║  Top-L: %-3d   k: %-3d                               ║",
                top_layers, per_layer_k)
    logger.info("  ╠══════════════════════════════════════════════════════╣")
    logger.info("  ║  Rank  Layer                           Score  #pos  ║")
    for i, (l, s, np_, nt) in enumerate(ranked_pos[:top_layers + 5]):
        marker = " ✓" if l in selected_names else "  "
        logger.info("  ║  %3d%s %-35s %8.1f %3d/%-3d ║",
                     i + 1, marker, l[:35], s, np_, nt)
    logger.info("  ╚══════════════════════════════════════════════════════╝")

    # Stage 2: top-k neurons from neuron_csv within selected layers
    df = pd.read_csv(neuron_csv)
    neurons: Dict[str, List[int]] = {}
    for lname, grp in df.groupby("LayerName"):
        raw = lname.replace(strip_prefix, "") if strip_prefix else lname
        if raw not in selected_names:
            continue
        top = grp.nlargest(per_layer_k, "Score")
        top = top[top["Score"] > 0]
        if len(top) > 0:
            neurons[raw] = sorted(top["NeuronIndex"].tolist())

    total = sum(len(v) for v in neurons.values())
    logger.info("  Selected %d neurons across %d layers (from %d total layers)",
                total, len(neurons), len(ranked))
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
    gnames = get_group_names()
    group_active: Dict[str, int] = {g: 0 for g in gnames}
    group_total: Dict[str, int] = {g: 0 for g in gnames}

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
    for g in gnames:
        if group_total[g] > 0:
            coverages[g] = group_active[g] / group_total[g]
        else:
            coverages[g] = 0.0

    total_active = sum(group_active.values())
    total_neurons = sum(group_total.values())
    coverages["overall"] = total_active / max(total_neurons, 1)

    grp_vals = [coverages[g] for g in gnames if group_total[g] > 0]
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
    gnames = get_group_names()
    group_change: Dict[str, list] = {g: [] for g in gnames}

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
    for g in gnames:
        result[g] = float(np.mean(group_change[g])) if group_change[g] else 0.0
    all_vals = sum(group_change.values(), [])
    result["overall"] = float(np.mean(all_vals)) if all_vals else 0.0
    grp_vals = [result[g] for g in gnames if group_change[g]]
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
        combo_mode: str = "per-layer",
    ):
        self.model = model
        self.target_neurons = target_neurons
        self.device = device
        self.method = method
        self.use_silhouette = use_silhouette
        self.k_max = k_max
        self.n_clusters = n_clusters
        self.combo_mode = combo_mode  # "per-layer" or "per-group"

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

    def _process_activations_to_assignments(
        self,
        activations: Dict[str, torch.Tensor],
        layer_assignments: Dict[str, list],
        layer_sizes: Dict[str, Dict[str, int]],
    ) -> None:
        """Convert batch activations to per-layer cluster assignments (in-place)."""
        B = next(iter(activations.values())).shape[0] if activations else 0
        for b in range(B):
            sample_acts: Dict[str, Dict[int, float]] = {}
            for lname in self.target_neurons:
                if lname not in activations:
                    continue
                sample_acts[lname] = {}
                for col, idx in enumerate(self.target_neurons[lname]):
                    sample_acts[lname][idx] = float(activations[lname][b, col].item())

            flat = self._assign_single(sample_acts)

            layer_dicts: Dict[str, Dict[str, int]] = {l: {} for l in self.target_neurons}
            for key, cluster_id in flat.items():
                lname = key.rsplit(":", 1)[0]
                if lname in layer_dicts:
                    layer_dicts[lname][key] = cluster_id
                    layer_sizes[lname][key] = self.cluster_sizes[key]

            for lname in self.target_neurons:
                if layer_dicts[lname]:
                    layer_assignments[lname].append(layer_dicts[lname])

    def collect_assignments(
        self, images: torch.Tensor, batch_size: int = 4
    ) -> Tuple[Dict[str, list], Dict[str, Dict[str, int]]]:
        """Collect per-layer cluster assignments for a set of images."""
        collector = ActivationCollector(self.model, self.target_neurons, self.device)
        collector.attach()

        layer_assignments: Dict[str, list] = {l: [] for l in self.target_neurons}
        layer_sizes: Dict[str, Dict[str, int]] = {l: {} for l in self.target_neurons}

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            acts = collector.collect(batch)
            self._process_activations_to_assignments(acts, layer_assignments, layer_sizes)

        collector.detach()
        return layer_assignments, layer_sizes

    def coverage_from_assignments(
        self,
        layer_assignments: Dict[str, list],
        layer_sizes: Dict[str, Dict[str, int]],
    ) -> Dict[str, float]:
        """Compute stratified coverage from pre-collected per-layer assignments.

        Dispatches to per-layer or per-group mode based on ``self.combo_mode``.
        """
        if self.combo_mode == "per-group":
            return self._coverage_per_group(layer_assignments, layer_sizes)
        return self._coverage_per_layer(layer_assignments, layer_sizes)

    def _coverage_per_layer(
        self,
        layer_assignments: Dict[str, list],
        layer_sizes: Dict[str, Dict[str, int]],
    ) -> Dict[str, float]:
        """Original per-layer combinatorial coverage (one combo space per layer)."""
        from wisdom.core.compute import combinations_coverage

        layer_covs: Dict[str, float] = {}
        for lname in self.target_neurons:
            if layer_assignments.get(lname) and layer_sizes.get(lname):
                r, t, _ = combinations_coverage(
                    layer_assignments[lname], layer_sizes[lname]
                )
                layer_covs[lname] = r
            else:
                layer_covs[lname] = 0.0

        gnames = get_group_names()
        group_covs: Dict[str, list] = {g: [] for g in gnames}
        for lname, cov in layer_covs.items():
            grp = _layer_group(lname)
            group_covs[grp].append(cov)

        coverages: Dict[str, float] = {}
        all_covs = []
        for g in gnames:
            if group_covs[g]:
                coverages[g] = float(np.mean(group_covs[g]))
                all_covs.extend(group_covs[g])
            else:
                coverages[g] = 0.0

        coverages["overall"] = float(np.mean(all_covs)) if all_covs else 0.0
        grp_vals = [coverages[g] for g in gnames if coverages[g] > 0]
        coverages["variability"] = float(np.std(grp_vals)) if len(grp_vals) > 1 else 0.0
        return coverages

    def _coverage_per_group(
        self,
        layer_assignments: Dict[str, list],
        layer_sizes: Dict[str, Dict[str, int]],
    ) -> Dict[str, float]:
        """Per-group combinatorial coverage — one combo space per group.

        Merges all neuron assignments within a group (early/middle/late)
        into a single combinatorial space, so coverage captures cross-layer
        interactions.  E.g., with 5 neurons × 3 clusters per group, the
        total combinations = 3^5 = 243 per group.
        """
        from wisdom.core.compute import combinations_coverage

        gnames = get_group_names()

        # Partition neuron keys by group
        group_keys: Dict[str, List[str]] = {g: [] for g in gnames}
        all_keys_flat: Dict[str, str] = {}  # "layer:idx" -> group
        for lname in self.target_neurons:
            grp = _layer_group(lname)
            for idx in self.target_neurons[lname]:
                key = f"{lname}:{idx}"
                group_keys[grp].append(key)
                all_keys_flat[key] = grp

        # Determine number of images from any layer's assignment list
        n_images = 0
        for lname in layer_assignments:
            if layer_assignments[lname]:
                n_images = len(layer_assignments[lname])
                break

        # Build per-group merged assignments
        group_assignments: Dict[str, List[Dict[str, int]]] = {
            g: [] for g in gnames
        }
        group_sizes: Dict[str, Dict[str, int]] = {
            g: {} for g in gnames
        }

        # Collect sizes
        for lname in layer_sizes:
            grp = _layer_group(lname)
            for key, sz in layer_sizes[lname].items():
                nkey_grp = all_keys_flat.get(key)
                if nkey_grp:
                    group_sizes[nkey_grp][key] = sz

        # Merge assignments across layers within each group
        for img_i in range(n_images):
            merged: Dict[str, Dict[str, int]] = {
                g: {} for g in gnames
            }
            for lname in self.target_neurons:
                grp = _layer_group(lname)
                if img_i < len(layer_assignments.get(lname, [])):
                    per_layer_dict = layer_assignments[lname][img_i]
                    for key, cid in per_layer_dict.items():
                        if key in all_keys_flat:
                            merged[grp][key] = cid
            for g in gnames:
                group_assignments[g].append(merged[g])

        coverages: Dict[str, float] = {}
        all_covs = []
        for g in gnames:
            if group_assignments[g] and group_sizes[g]:
                r, t, _ = combinations_coverage(
                    group_assignments[g], group_sizes[g]
                )
                coverages[g] = r
                all_covs.append(r)
            else:
                coverages[g] = 0.0

        coverages["overall"] = float(np.mean(all_covs)) if all_covs else 0.0
        grp_vals = [coverages[g] for g in gnames if coverages[g] > 0]
        coverages["variability"] = float(np.std(grp_vals)) if len(grp_vals) > 1 else 0.0
        return coverages

    def coverage(
        self, test_images: torch.Tensor, batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Compute per-layer combinatorial cluster coverage, then aggregate.

        Returns dict with 'early', 'middle', 'late', 'overall', 'variability'.
        """
        assert self._fitted, "Call fit() before coverage()"
        assignments, sizes = self.collect_assignments(test_images, batch_size)
        return self.coverage_from_assignments(assignments, sizes)

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
        for k in list(get_group_names()) + ["overall", "variability"]:
            delta[f"clean_{k}"] = cov_clean[k]
            delta[f"pert_{k}"] = cov_pert[k]
            delta[f"delta_{k}"] = abs(cov_pert[k] - cov_clean[k])
        return delta


# ── Union coverage trackers (for RQ2 set-union methodology) ─────────

class UnionCoverageTracker:
    """
    Incrementally track union (set-union) threshold coverage.

    A neuron is "covered" if its activation exceeds the calibrated threshold
    in ANY image across all updates.  This implements the original WISDOM
    C(D_O ∪ D_I) methodology for plain threshold coverage.

    Usage::

        tracker = UnionCoverageTracker(thresholds, top_neurons)
        for batch_acts in clean_activations:
            tracker.update(batch_acts)
        # fork for a perturbation variant
        variant = tracker.clone()
        for batch_acts in perturbed_activations:
            variant.update(batch_acts)
        C_union = variant.coverage()
    """

    def __init__(
        self,
        thresholds: Dict[str, torch.Tensor],
        target_neurons: Dict[str, List[int]],
    ):
        self.thresholds = thresholds
        self.target_neurons = target_neurons
        self.max_acts: Dict[str, torch.Tensor] = {
            l: torch.full((len(idxs),), float("-inf"))
            for l, idxs in target_neurons.items()
        }

    def update(self, activations: Dict[str, torch.Tensor]) -> None:
        """Update with activations {layer: (B, n_neurons)} from a batch."""
        for l in self.target_neurons:
            if l in activations:
                batch_max = activations[l].float().abs().max(dim=0).values.cpu()
                self.max_acts[l] = torch.max(self.max_acts[l], batch_max)

    def coverage(self) -> Dict[str, float]:
        """Compute stratified union coverage from accumulated max activations."""
        gnames = get_group_names()
        group_active: Dict[str, int] = {g: 0 for g in gnames}
        group_total: Dict[str, int] = {g: 0 for g in gnames}
        for l in self.target_neurons:
            thr = self.thresholds.get(l, torch.zeros(len(self.target_neurons[l])))
            active = (self.max_acts[l] > thr).sum().item()
            total = len(self.target_neurons[l])
            grp = _layer_group(l)
            group_active[grp] += active
            group_total[grp] += total
        result: Dict[str, float] = {}
        for g in gnames:
            result[g] = group_active[g] / max(group_total[g], 1)
        result["overall"] = sum(group_active.values()) / max(sum(group_total.values()), 1)
        grp_vals = [result[g] for g in gnames if group_total[g] > 0]
        result["variability"] = float(np.std(grp_vals)) if len(grp_vals) > 1 else 0.0
        return result

    def clone(self) -> "UnionCoverageTracker":
        """Return a deep copy for forking baseline state."""
        new = UnionCoverageTracker.__new__(UnionCoverageTracker)
        new.thresholds = self.thresholds
        new.target_neurons = self.target_neurons
        new.max_acts = {l: t.clone() for l, t in self.max_acts.items()}
        return new

    def reset(self) -> None:
        for l in self.target_neurons:
            self.max_acts[l].fill_(float("-inf"))


class ClusterUnionTracker:
    """
    Incrementally track union combinatorial cluster coverage.

    Collects cluster-state tuples from each image and computes coverage
    as |unique tuples| / |total possible tuples|, pooled across all
    images ever passed through ``update`` / ``update_from_activations``.

    Usage::

        tracker = ClusterUnionTracker(cluster_comp)
        tracker.update(clean_images)        # or update_from_activations()
        variant = tracker.clone()
        variant.update(perturbed_images)
        C_union = variant.coverage()
    """

    def __init__(self, cluster_comp: ClusterCoverageComputer):
        self.comp = cluster_comp
        self.layer_assignments: Dict[str, list] = {
            l: [] for l in cluster_comp.target_neurons
        }
        self.layer_sizes: Dict[str, Dict[str, int]] = {
            l: {} for l in cluster_comp.target_neurons
        }

    def update_from_activations(self, activations: Dict[str, torch.Tensor]) -> None:
        """Update from pre-collected {layer: (B, n_neurons)} activations."""
        self.comp._process_activations_to_assignments(
            activations, self.layer_assignments, self.layer_sizes
        )

    def update(self, images: torch.Tensor, batch_size: int = 4) -> None:
        """Process images end-to-end (forward pass → cluster assignment)."""
        assignments, sizes = self.comp.collect_assignments(images, batch_size)
        for l in self.layer_assignments:
            self.layer_assignments[l].extend(assignments.get(l, []))
            self.layer_sizes[l].update(sizes.get(l, {}))

    def coverage(self) -> Dict[str, float]:
        return self.comp.coverage_from_assignments(
            self.layer_assignments, self.layer_sizes
        )

    def clone(self) -> "ClusterUnionTracker":
        """Return a deep copy for forking baseline state."""
        new = ClusterUnionTracker.__new__(ClusterUnionTracker)
        new.comp = self.comp
        new.layer_assignments = {
            l: list(v) for l, v in self.layer_assignments.items()
        }
        new.layer_sizes = {l: dict(v) for l, v in self.layer_sizes.items()}
        return new

    def reset(self) -> None:
        for l in self.comp.target_neurons:
            self.layer_assignments[l] = []
            self.layer_sizes[l] = {}


# ── Verbose logging helpers ─────────────────────────────────────────

def _build_group_combo_sets(
    tracker: ClusterUnionTracker,
) -> Dict[str, dict]:
    """Build per-group combo sets from a tracker's layer assignments.

    Merges neuron assignments across layers within each group
    (early/middle/late), mirroring _coverage_per_group logic.

    Returns {group_name: {"seen_set": set, "total": int, "n_neurons": int,
                          "neuron_keys": list}}.
    """
    comp = tracker.comp
    gnames = get_group_names()

    # Partition neuron keys by group
    group_keys: Dict[str, List[str]] = {g: [] for g in gnames}
    all_keys_flat: Dict[str, str] = {}
    for lname in comp.target_neurons:
        grp = _layer_group(lname)
        for idx in comp.target_neurons[lname]:
            key = f"{lname}:{idx}"
            group_keys[grp].append(key)
            all_keys_flat[key] = grp

    # Determine number of images
    n_images = 0
    for lname in tracker.layer_assignments:
        if tracker.layer_assignments[lname]:
            n_images = len(tracker.layer_assignments[lname])
            break

    # Collect sizes per group
    group_sizes: Dict[str, Dict[str, int]] = {g: {} for g in gnames}
    for lname in tracker.layer_sizes:
        grp = _layer_group(lname)
        for key, sz in tracker.layer_sizes[lname].items():
            nkey_grp = all_keys_flat.get(key)
            if nkey_grp:
                group_sizes[nkey_grp][key] = sz

    # Build merged assignments per group (same logic as _coverage_per_group)
    result: Dict[str, dict] = {}
    for g in gnames:
        if not group_sizes[g]:
            result[g] = {"seen_set": set(), "total": 0,
                         "n_neurons": 0, "neuron_keys": []}
            continue

        keys_sorted = sorted(group_sizes[g].keys())
        total_possible = 1
        for k in keys_sorted:
            total_possible *= int(group_sizes[g][k])

        seen = set()
        for img_i in range(n_images):
            merged: Dict[str, int] = {}
            for lname in comp.target_neurons:
                if _layer_group(lname) != g:
                    continue
                if img_i < len(tracker.layer_assignments.get(lname, [])):
                    per_layer_dict = tracker.layer_assignments[lname][img_i]
                    for key, cid in per_layer_dict.items():
                        if key in all_keys_flat:
                            merged[key] = cid
            tup = tuple(merged.get(k, -1) for k in keys_sorted)
            if -1 not in tup:
                seen.add(tup)

        result[g] = {
            "seen_set": seen,
            "total": total_possible,
            "n_neurons": len(group_keys[g]),
            "neuron_keys": group_keys[g],
        }

    return result


def verbose_coverage_breakdown(
    tracker: ClusterUnionTracker,
    label: str = "U_O",
) -> Dict[str, dict]:
    """Log per-layer (or per-group) combo counts for a single tracker.

    Automatically detects per-layer vs per-group mode from
    tracker.comp.combo_mode.

    Returns dict of {unit_name: {seen, total, ratio}} for further analysis.
    """
    from wisdom.core.compute import combinations_coverage

    combo_mode = getattr(tracker.comp, "combo_mode", "per-layer")

    if combo_mode == "per-group":
        # ── Per-group: merge neurons across layers within each group
        group_data = _build_group_combo_sets(tracker)
        details: Dict[str, dict] = {}
        for g in get_group_names():
            gd = group_data[g]
            if gd["total"] > 0:
                seen = len(gd["seen_set"])
                ratio = seen / gd["total"]
                details[g] = {"seen": seen, "total": gd["total"],
                              "ratio": ratio, "n_neurons": gd["n_neurons"]}
                neuron_list = ", ".join(gd["neuron_keys"])
                logger.info(
                    f"  [{label}] group={g}: {seen}/{gd['total']} combos "
                    f"({ratio:.4f}) | n_neurons={gd['n_neurons']} "
                    f"[{neuron_list}]"
                )
            else:
                details[g] = {"seen": 0, "total": 0, "ratio": 0.0,
                              "n_neurons": 0}
        active = [d for d in details.values() if d["total"] > 0]
        overall = np.mean([d["ratio"] for d in active]) if active else 0.0
        logger.info(
            f"  [{label}] Overall: {overall:.6f} "
            f"(mean of {len(active)} groups)"
        )
        return details

    # ── Per-layer (original behavior)
    details = {}
    for lname in tracker.comp.target_neurons:
        assigns = tracker.layer_assignments.get(lname, [])
        sizes = tracker.layer_sizes.get(lname, {})
        if assigns and sizes:
            r, total, _ = combinations_coverage(assigns, sizes)
            seen = int(round(r * total))
            details[lname] = {"seen": seen, "total": total, "ratio": r}
            logger.info(
                f"  [{label}] {lname}: {seen}/{total} combos "
                f"({r:.4f})"
            )
        else:
            details[lname] = {"seen": 0, "total": 0, "ratio": 0.0}
    overall = np.mean([d["ratio"] for d in details.values() if d["total"] > 0])
    logger.info(
        f"  [{label}] Overall: {overall:.6f} "
        f"(mean of {sum(1 for d in details.values() if d['total'] > 0)} layers)"
    )
    return details


def verbose_union_diff(
    baseline: ClusterUnionTracker,
    variant: ClusterUnionTracker,
    label: str = "D_I",
) -> Dict[str, dict]:
    """Log per-layer or per-group combo diff between baseline and variant.

    Automatically detects per-layer vs per-group mode.
    Shows: baseline_seen, union_seen, new_from_variant, total_possible.
    """
    from wisdom.core.compute import combinations_coverage

    combo_mode = getattr(baseline.comp, "combo_mode", "per-layer")

    if combo_mode == "per-group":
        # ── Per-group diff
        base_data = _build_group_combo_sets(baseline)
        var_data = _build_group_combo_sets(variant)
        details: Dict[str, dict] = {}
        total_new = 0
        for g in get_group_names():
            bd = base_data[g]
            vd = var_data[g]
            if bd["total"] == 0:
                details[g] = {"base_seen": 0, "union_seen": 0, "new": 0,
                              "total": 0, "base_ratio": 0.0,
                              "union_ratio": 0.0}
                continue
            base_set = bd["seen_set"]
            var_set = vd["seen_set"]
            union_set = base_set | var_set
            new_combos = var_set - base_set
            base_ratio = len(base_set) / bd["total"]
            union_ratio = len(union_set) / bd["total"]
            details[g] = {
                "base_seen": len(base_set),
                "union_seen": len(union_set),
                "new": len(new_combos),
                "total": bd["total"],
                "base_ratio": base_ratio,
                "union_ratio": union_ratio,
                "n_neurons": bd["n_neurons"],
            }
            total_new += len(new_combos)
            sat = " [SATURATED]" if base_ratio > 0.85 else ""
            neuron_list = ", ".join(bd["neuron_keys"])
            logger.info(
                f"  [{label}] group={g}: base={len(base_set)}, "
                f"union={len(union_set)}, new={len(new_combos)}, "
                f"total={bd['total']} "
                f"({base_ratio:.4f} → {union_ratio:.4f}){sat}\n"
                f"           neurons: [{neuron_list}]"
            )
        active = [d for d in details.values() if d["total"] > 0]
        overall_base = np.mean([d["base_ratio"] for d in active]) if active else 0.0
        overall_union = np.mean([d["union_ratio"] for d in active]) if active else 0.0
        logger.info(
            f"  [{label}] Summary: {len(active)} groups | "
            f"total_new_combos={total_new} | "
            f"coverage: {overall_base:.4f} → {overall_union:.4f} "
            f"(Δ={overall_union - overall_base:+.6f})"
        )
        return details

    # ── Per-layer diff (original behavior)
    details = {}
    total_new = 0
    total_base_seen = 0
    saturated_layers = []
    variant_wins = 0
    no_change = 0

    for lname in baseline.comp.target_neurons:
        b_assigns = baseline.layer_assignments.get(lname, [])
        b_sizes = baseline.layer_sizes.get(lname, {})
        v_assigns = variant.layer_assignments.get(lname, [])
        v_sizes = variant.layer_sizes.get(lname, {})

        if not (b_assigns and b_sizes):
            details[lname] = {"base_seen": 0, "union_seen": 0, "new": 0,
                              "total": 0, "base_ratio": 0.0, "union_ratio": 0.0}
            continue

        keys = sorted(b_sizes.keys())
        total_possible = 1
        for k in keys:
            total_possible *= int(b_sizes[k])

        base_set = set()
        for a in b_assigns:
            tup = tuple(a.get(k, -1) for k in keys)
            if -1 not in tup:
                base_set.add(tup)

        v_set = set()
        for a in v_assigns:
            tup = tuple(a.get(k, -1) for k in keys)
            if -1 not in tup:
                v_set.add(tup)

        union_set = base_set | v_set
        new_combos = v_set - base_set

        base_ratio = len(base_set) / total_possible if total_possible > 0 else 0.0
        union_ratio = len(union_set) / total_possible if total_possible > 0 else 0.0

        details[lname] = {
            "base_seen": len(base_set),
            "union_seen": len(union_set),
            "new": len(new_combos),
            "total": total_possible,
            "base_ratio": base_ratio,
            "union_ratio": union_ratio,
        }

        total_new += len(new_combos)
        total_base_seen += len(base_set)
        if base_ratio > 0.85:
            saturated_layers.append(lname)
        if len(new_combos) > 0:
            variant_wins += 1
        else:
            no_change += 1

        logger.info(
            f"  [{label}] {lname}: base={len(base_set)}, "
            f"union={len(union_set)}, new={len(new_combos)}, "
            f"total={total_possible} "
            f"({base_ratio:.4f} → {union_ratio:.4f})"
            f"{' [SATURATED]' if base_ratio > 0.85 else ''}"
        )

    n_active = sum(1 for d in details.values() if d["total"] > 0)
    overall_base = np.mean([d["base_ratio"] for d in details.values() if d["total"] > 0])
    overall_union = np.mean([d["union_ratio"] for d in details.values() if d["total"] > 0])
    logger.info(
        f"  [{label}] Summary: {n_active} layers | "
        f"total_new_combos={total_new} | "
        f"layers_with_new={variant_wins}/{n_active} | "
        f"no_change={no_change} | "
        f"saturated={len(saturated_layers)} | "
        f"coverage: {overall_base:.4f} → {overall_union:.4f} "
        f"(Δ={overall_union - overall_base:+.6f})"
    )
    return details

# core/wisdom_train.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple

import math
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wisdom.utils.io_cache import save_layer_scores_csv
from wisdom.utils.detection_loader import detect_head_prefixes as infer_detect_head_prefixes
from wisdom.utils.detection_loader import infer_num_classes, normalize_detection_output
from wisdom.attribution.captum_backend import batch_per_layer_scores

# -----------------------------
# Config
# -----------------------------
@dataclass
class WisdomTrainConfig:
    methods: List[str] = field(default_factory=lambda: ["lrp", "ldl", "lig"])
    device: str = "cuda:0"
    voting_weights: Optional[List[float]] = None
    voting_mode: str = "fine-grained"  # "fine-grained" | "coarse"
    selection_mode: str = "global"  # "global" | "per-group"
    pruning_augmentations: Optional[List[Dict]] = None
    out_csv: Optional[str] = None
    # YOLO-specific: when True, use detection surrogate loss instead of CE
    is_yolo: bool = False
    num_classes: int = 80  # COCO classes for YOLO


# -----------------------------
# Helper functions
# -----------------------------
def _is_trainable_module(m: nn.Module) -> bool:
    return isinstance(m, (nn.Conv2d, nn.Linear))

def _trainable_modules(model: nn.Module) -> Tuple[List[str], List[nn.Module]]:
    names, mods = [], []
    for n, m in model.named_modules():
        if _is_trainable_module(m):
            names.append(n)
            mods.append(m)
    return names, mods

def _criterion():
    return nn.CrossEntropyLoss()

def _eval_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    """Evaluate classification loss for a batch."""
    model.eval().to(device)
    x = x.to(device); y = y.to(device)
    with torch.no_grad():
        out = model(x)
        loss = _criterion()(out, y).item()
    return loss

def _eval_loss_yolo(model: nn.Module, x: torch.Tensor, device: str) -> float:
    """Surrogate loss for YOLO: negative sum of class confidences.
    When neurons are pruned, confidence drops → loss rises."""
    model.eval().to(device)
    x = x.to(device)
    with torch.no_grad():
        preds = normalize_detection_output(model(x), num_classes=infer_num_classes(model))
        cls_scores = preds[:, 4:, :]  # (B, nc, A)
        return -cls_scores.sum().item()

def _voting_init(layer_scores: Dict[str, torch.Tensor],
                 trainable_names: List[str],
                 trainable_mods: List[nn.Module],
                 excluded_layer: Optional[str] = None,
                 excluded_prefixes: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    if layer_scores:
        return layer_scores
    for lname, m in zip(trainable_names, trainable_mods):
        if excluded_layer and lname == excluded_layer:
            continue
        if excluded_prefixes and any(lname.startswith(p) for p in excluded_prefixes):
            continue
        if isinstance(m, nn.Conv2d):
            layer_scores[lname] = torch.zeros(m.out_channels, dtype=torch.float32)
        elif isinstance(m, nn.Linear):
            layer_scores[lname] = torch.zeros(m.out_features, dtype=torch.float32)
    return layer_scores

def _voting_neurons(layer_index_pairs: List[Tuple[str, int]],
                    layer_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Assign rank points (higher rank gets more points) like prepare_data.voting_neurons.
    Input list is assumed sorted DESC by importance; we intentionally enumerate reversed.
    """
    for rank, (layer_name, neuron_index) in enumerate(reversed(layer_index_pairs), start=1):
        if layer_name in layer_scores and 0 <= neuron_index < layer_scores[layer_name].numel():
            layer_scores[layer_name][neuron_index] += rank
    return layer_scores

def _weighted_top_neurons(important_neurons_dict: Dict[str, List[Tuple[str, float, int]]],
                          loss_gains: Dict[str, float],
                          top_k: int) -> List[Tuple[Tuple[str, int], float]]:
    """
    Same spirit as prepare_data.weighted_top_neurons:
      - normalize loss_gains to weights,
      - accumulate weight * score per (layer, idx),
      - return top_k ((layer, idx), weighted_score).
    """
    total_gain = sum(max(0.0, g) for g in loss_gains.values()) or 1.0
    weights = {m: max(0.0, g) / total_gain for m, g in loss_gains.items()}
    weighted_scores: Dict[Tuple[str, int], float] = {}
    for method, triples in important_neurons_dict.items():
        w = weights.get(method, 0.0)
        if w == 0.0:
            continue
        for (layer_name, score, idx) in triples:
            key = (layer_name, int(idx))
            weighted_scores[key] = weighted_scores.get(key, 0.0) + float(score) * w
    sorted_neurons = sorted(weighted_scores.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_neurons[:top_k]


# -----------------------------
# Attribution (per-batch, Captum-friendly)
# -----------------------------
def _compute_batch_importance_captum(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-layer importance on ONE batch, reduced to per-neuron vectors.
    Returns {layer_name: tensor(#neurons,)}.
    """
    try:
        from captum.attr import (
            LayerLRP, LayerDeepLift, LayerIntegratedGradients, LayerGradientShap,
            LayerFeatureAblation, LayerActivation, InternalInfluence,
            LayerGradientXActivation, LayerGradCam
        )
    except Exception as e:
        raise RuntimeError(
            "Captum is required for _compute_batch_importance_captum. Please install captum or "
            "wire this to your attribution/registry."
        ) from e

    name2ctor = {
        "lrp": LayerLRP,
        "ldl": LayerDeepLift,
        "lig": LayerIntegratedGradients,
        "lgs": LayerGradientShap,
        "lfa": LayerFeatureAblation,
        "la":  LayerActivation,
        "ii":  InternalInfluence,
        "lgxa": LayerGradientXActivation,
        "lgc": LayerGradCam,
    }
    key = method.lower()
    if key not in name2ctor:
        raise ValueError(f"Unknown Captum method '{method}'. Options: {list(name2ctor.keys())}")

    model = model.to(device).eval()
    images = images.to(device); labels = labels.to(device)

    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for lname, layer in model.named_modules():
            if not _is_trainable_module(layer):
                continue
            A = name2ctor[key](model, layer)
            if key == "la":
                attr = A.attribute(images)
            elif key in ("ldl", "lgs"):
                attr = A.attribute(images, baselines=torch.zeros_like(images), target=labels)
            else:
                attr = A.attribute(images, target=labels)

            # reduce to per-neuron vector
            if attr.dim() == 4:  # (B,C,H,W)
                vec = attr.sum(dim=(0, 2, 3)).detach().cpu()
            else:                # (B,F)
                vec = attr.sum(dim=0).detach().cpu()
            out[lname] = vec

    return out


def _select_top_neurons_all(
    importance_scores_dict: Dict[str, torch.Tensor],
    top_m_neurons: int,
    filter_layer: Optional[str] = None,
    filter_prefixes: Optional[List[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, float, int]]]:
    """
    Flatten all layers' importance and pick top-M across layers (optionally excluding final layer
    or layers matching given prefixes, e.g. detection head).
    Returns:
      - indices_by_layer: {layer: 1D LongTensor of selected indices}
      - selected_triplets: list of (layer_name, score, idx) sorted desc by score
    """
    flattened: List[Tuple[str, float, int]] = []
    for layer_name, scores in importance_scores_dict.items():
        if filter_layer and layer_name == filter_layer:
            continue
        if filter_prefixes and any(layer_name.startswith(p) for p in filter_prefixes):
            continue
        if scores.dim() == 1:
            for idx, s in enumerate(scores):
                flattened.append((layer_name, float(s.item()), int(idx)))
        else:
            # Should not happen because we reduce per layer, but keep safe
            mean_attr = scores.mean(dim=tuple(range(1, scores.dim())))
            for idx, s in enumerate(mean_attr):
                flattened.append((layer_name, float(s.item()), int(idx)))

    flattened.sort(key=lambda x: x[1], reverse=True)
    selected = flattened if top_m_neurons == -1 else flattened[:top_m_neurons]

    by_layer: Dict[str, List[int]] = {}
    for layer_name, _, idx in selected:
        by_layer.setdefault(layer_name, []).append(idx)

    indices_by_layer = {
        layer: torch.tensor(sorted(idxs), dtype=torch.long) for layer, idxs in by_layer.items()
    }
    return indices_by_layer, selected


# Layer-group boundaries for per-group selection (YOLOv11 architecture)
_GROUP_BOUNDS = {
    "early": (0, 5),
    "middle": (6, 12),
    "late": (13, 22),
}

def _layer_group_from_name(layer_name: str) -> str:
    """Determine early/middle/late group from layer name like 'yolo_model.model.7.cv1.conv'."""
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


def _select_top_neurons_per_group(
    importance_scores_dict: Dict[str, torch.Tensor],
    top_m_per_group: int,
    filter_layer: Optional[str] = None,
    filter_prefixes: Optional[List[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, float, int]]]:
    """
    Select top-M neurons **per layer group** (early/middle/late).

    This ensures balanced representation across network depth, avoiding
    the early-layer dominance that occurs with global top-M selection.

    Returns same format as ``_select_top_neurons_all``.
    """
    # Flatten all neurons with their group tags
    grouped: Dict[str, List[Tuple[str, float, int]]] = {
        "early": [], "middle": [], "late": [],
    }
    for layer_name, scores in importance_scores_dict.items():
        if filter_layer and layer_name == filter_layer:
            continue
        if filter_prefixes and any(layer_name.startswith(p) for p in filter_prefixes):
            continue
        grp = _layer_group_from_name(layer_name)
        if scores.dim() == 1:
            for idx, s in enumerate(scores):
                grouped[grp].append((layer_name, float(s.item()), int(idx)))
        else:
            mean_attr = scores.mean(dim=tuple(range(1, scores.dim())))
            for idx, s in enumerate(mean_attr):
                grouped[grp].append((layer_name, float(s.item()), int(idx)))

    # Select top-M per group, combine
    selected: List[Tuple[str, float, int]] = []
    for grp in ("early", "middle", "late"):
        grp_sorted = sorted(grouped[grp], key=lambda x: x[1], reverse=True)
        selected.extend(grp_sorted[:top_m_per_group])

    # Re-sort combined result DESC by score (for voting)
    selected.sort(key=lambda x: x[1], reverse=True)

    by_layer: Dict[str, List[int]] = {}
    for layer_name, _, idx in selected:
        by_layer.setdefault(layer_name, []).append(idx)

    indices_by_layer = {
        layer: torch.tensor(sorted(idxs), dtype=torch.long)
        for layer, idxs in by_layer.items()
    }
    return indices_by_layer, selected


# -----------------------------
# Pruning backends (new style)
# -----------------------------
class _WeightsGuard:
    def __init__(self, model: nn.Module):
        self.model = model
        self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    def restore(self):
        self.model.load_state_dict(self.state, strict=True)

def _mask_prune_once(model: nn.Module, selection: Dict[str, List[int]], x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    from wisdom.pruning.mask_pruning import mask_model_neurons
    handle = mask_model_neurons(model, selection)
    try:
        return _eval_loss(model, x, y, device)
    finally:
        try: handle.remove()
        except Exception: pass

def _weights_prune_once(model: nn.Module, selection: Dict[str, List[int]], x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    from wisdom.pruning.weights_pruning import prune_model_neurons
    guard = _WeightsGuard(model)
    try:
        prune_model_neurons(model, selection)
        return _eval_loss(model, x, y, device)
    finally:
        guard.restore()


# -----------------------------
# YOLO-specific helpers
# -----------------------------
def _compute_yolo_importance(
    wrapper: nn.Module,
    images: torch.Tensor,
    method: str,
    device: str,
    num_classes: int = 80,
    exclude_detect_head: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-layer importance for a YOLO model via the YOLOWrapper.
    Uses the wrapper (which outputs (B, nc)) so Captum can attribute
    to a class target.

    When exclude_detect_head=True, skips all layers in the detection head
    (model.23.*) to avoid pruning the final detection layers – analogous
    to excluding the classifier in classification networks.
    """
    try:
        from captum.attr import (
            LayerGradientXActivation, LayerIntegratedGradients,
            LayerGradientShap, LayerActivation,
        )
    except ImportError as e:
        raise RuntimeError("Captum is required") from e

    # Methods known to work with YOLO (LRP and DeepLift fail)
    name2ctor = {
        "lgxa": LayerGradientXActivation,
        "lig":  LayerIntegratedGradients,
        "lgs":  LayerGradientShap,
        "la":   LayerActivation,
    }

    key = method.lower()
    if key not in name2ctor:
        # Fallback: use gradient magnitude
        return _gradient_importance(wrapper, images, device, num_classes)

    wrapper = wrapper.to(device).eval()
    images = images.to(device)
    # Target the most common class (0) for attribution
    target = torch.zeros(images.size(0), dtype=torch.long, device=device)

    out: Dict[str, torch.Tensor] = {}
    detect_head_prefixes = infer_detect_head_prefixes(wrapper)
    for lname, layer in wrapper.named_modules():
        if not _is_trainable_module(layer):
            continue
        if exclude_detect_head and any(lname.startswith(p) for p in detect_head_prefixes):
            continue
        A = name2ctor[key](wrapper, layer)
        if key == "la":
            attr = A.attribute(images)
        elif key == "lgs":
            attr = A.attribute(images, baselines=torch.zeros_like(images), target=target)
        else:
            attr = A.attribute(images, target=target)
        if attr.dim() == 4:
            vec = attr.sum(dim=(0, 2, 3)).detach().cpu()
        else:
            vec = attr.sum(dim=0).detach().cpu()
        out[lname] = vec
    return out


def _gradient_importance(
    model: nn.Module,
    images: torch.Tensor,
    device: str,
    num_classes: int = 80,
    exclude_detect_head: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Fallback importance: gradient magnitude w.r.t. each layer's parameters.
    Works with any model architecture.
    """
    model = model.to(device).eval()
    images = images.to(device).requires_grad_(False)

    # Enable grad for parameters
    for p in model.parameters():
        p.requires_grad_(True)

    out_logits = model(images)
    if isinstance(out_logits, (tuple, list)):
        out_logits = out_logits[0]
    # Sum all outputs as the target scalar
    scalar = out_logits.sum()
    scalar.backward()

    scores: Dict[str, torch.Tensor] = {}
    detect_head_prefixes = infer_detect_head_prefixes(model, wrapper_prefix="")
    for lname, m in model.named_modules():
        if not _is_trainable_module(m):
            continue
        if exclude_detect_head and any(lname.startswith(p) for p in detect_head_prefixes):
            continue
        w = m.weight
        if w.grad is not None:
            if isinstance(m, nn.Conv2d):
                vec = w.grad.abs().sum(dim=(1, 2, 3)).detach().cpu()
            else:
                vec = w.grad.abs().sum(dim=1).detach().cpu()
            scores[lname] = vec

    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(False)
    return scores


def _yolo_prune_eval(
    model: nn.Module,
    selection: Dict[str, List[int]],
    images: torch.Tensor,
    device: str,
    prune_mode: str,
) -> float:
    """Prune, evaluate YOLO surrogate loss, then restore."""
    if prune_mode == "mask":
        from wisdom.pruning.mask_pruning import mask_model_neurons
        handle = mask_model_neurons(model, selection)
        try:
            return _eval_loss_yolo(model, images, device)
        finally:
            try:
                handle.remove()
            except Exception:
                pass
    else:
        guard = _WeightsGuard(model)
        try:
            from wisdom.pruning.weights_pruning import prune_model_neurons
            prune_model_neurons(model, selection)
            return _eval_loss_yolo(model, images, device)
        finally:
            guard.restore()


# -----------------------------
# Main Trainer
# -----------------------------
class ConsensusWisdom:
    """
    Multi-method voting + optional pruning to identify important neurons.
    Steps per batch:
      1) Get important neurons for each attribution method.
      2) Identify optimal method by pruning its top neurons and measuring loss gain.
      3) Initialize voting buffers on first batch.
      4) Update votes:
         - fine-grained: pick TOP-K neurons across methods via weighted_top_neurons (weights from loss gains),
           then vote them by rank (like prepare_data.voting_neurons).
         - coarse: take only the optimal method's neurons and vote by rank.
      5) After all batches, save CSV.

    For YOLO models, set cfg.is_yolo=True.  The model must be the raw
    torch Module (``model.model`` from Ultralytics), and the wrapper is
    built internally so that Captum receives a classifier-like interface.
    """

    def __init__(self, model: nn.Module, device: str = "cuda:0"):
        self.model = model
        self.device = device
        self.model.eval().to(device)
        self.trainable_names, self.trainable_mods = _trainable_modules(model)

    # -------- public API --------
    def fit(
        self,
        train_loader: DataLoader,
        cfg: WisdomTrainConfig,
        top_m_neurons: int,
        final_layer: Optional[str] = None,
        prune_mode: str = "mask",  # "mask" | "weights"
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 50,
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Returns (layer_scores, csv_path).

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to a ``.pt`` file for saving/resuming progress.  If the
            file already exists, training resumes from the saved batch
            index.  A checkpoint is written every *checkpoint_every*
            batches.
        checkpoint_every : int
            How often (in batches) to save a checkpoint.  Default 50.
        """
        import os
        assert cfg.out_csv, "Please provide cfg.out_csv to save layer scores."
        final_layer = final_layer or (self.trainable_names[-1] if self.trainable_names else None)

        # For YOLO, exclude all detection head layers (model.23.*) instead
        # of just the single final layer – analogous to excluding the
        # classifier head in classification networks.
        detect_head_prefix_list: Optional[List[str]] = None
        if cfg.is_yolo:
            detect_head_prefix_list = infer_detect_head_prefixes(self.model)
            final_layer = None  # prefix-based exclusion replaces single-layer

        # For YOLO, create a wrapper for Captum attribution
        wrapper = None
        if cfg.is_yolo:
            from wisdom.utils.yolo_wrapper import YOLOWrapper
            wrapper = YOLOWrapper(self.model, num_classes=cfg.num_classes or infer_num_classes(self.model))
            wrapper.eval().to(self.device)
            # Remap trainable layer names to wrapper namespace
            wrapper_names, wrapper_mods = _trainable_modules(wrapper)

        layer_scores: Dict[str, torch.Tensor] = {}
        init_done = False

        # Checkpoint resume
        start_batch = 0
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            layer_scores = {k: v.clone() for k, v in ckpt["layer_scores"].items()}
            start_batch = ckpt["batch_idx"] + 1
            init_done = bool(layer_scores)
            print(f"[WISDOM] Resuming from checkpoint batch {start_batch}")

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            if batch_idx < start_batch:
                continue
            if cfg.is_yolo:
                # YOLO dataloaders may yield (images,) or (images, targets)
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                labels = torch.zeros(images.size(0), dtype=torch.long)  # dummy
            else:
                images, labels = batch

            # 1) Important neurons per method on THIS batch
            important_neurons_dict: Dict[str, List[Tuple[str, float, int]]] = {}
            for method in cfg.methods:
                if cfg.is_yolo:
                    imp = _compute_yolo_importance(
                        wrapper, images, method, cfg.device, cfg.num_classes
                    )
                else:
                    imp = batch_per_layer_scores(
                        model=self.model,
                        images=images,
                        labels=labels,
                        device=cfg.device,
                        method=method,
                        target_layers=None,
                    )
                # Select top neurons: global or per-group
                if cfg.selection_mode == "per-group" and cfg.is_yolo:
                    _, selected_triplets = _select_top_neurons_per_group(
                        imp, top_m_per_group=max(1, top_m_neurons // 3),
                        filter_layer=final_layer,
                        filter_prefixes=detect_head_prefix_list,
                    )
                else:
                    _, selected_triplets = _select_top_neurons_all(
                        imp, top_m_neurons=top_m_neurons,
                        filter_layer=final_layer,
                        filter_prefixes=detect_head_prefix_list,
                    )
                important_neurons_dict[method] = selected_triplets

            # 2) Identify optimal method by loss gain after pruning
            if cfg.is_yolo:
                base_loss = _eval_loss_yolo(self.model, images, cfg.device)
            else:
                base_loss = _eval_loss(self.model, images, labels, cfg.device)

            loss_gains: Dict[str, float] = {}
            for method, triplets in important_neurons_dict.items():
                selection: Dict[str, List[int]] = {}
                for (layer_name, _score, idx) in triplets:
                    # Map wrapper names back to raw model names for pruning
                    prune_name = layer_name
                    if cfg.is_yolo and layer_name.startswith("yolo_model."):
                        prune_name = layer_name[len("yolo_model."):]
                    selection.setdefault(prune_name, []).append(int(idx))

                if cfg.is_yolo:
                    pruned_loss = _yolo_prune_eval(
                        self.model, selection, images, cfg.device, prune_mode
                    )
                elif prune_mode == "mask":
                    pruned_loss = _mask_prune_once(self.model, selection, images, labels, cfg.device)
                elif prune_mode == "weights":
                    pruned_loss = _weights_prune_once(self.model, selection, images, labels, cfg.device)
                else:
                    raise ValueError("prune_mode must be 'mask' or 'weights'")

                loss_gains[method] = pruned_loss - base_loss

            optimal_method = max(loss_gains, key=loss_gains.get)

            # 3) Voting buffers
            if not init_done:
                if cfg.is_yolo:
                    # Use wrapper layer names for scoring
                    layer_scores = _voting_init(
                        layer_scores, wrapper_names, wrapper_mods,
                        excluded_layer=final_layer,
                        excluded_prefixes=detect_head_prefix_list,
                    )
                else:
                    layer_scores = _voting_init(
                        layer_scores, self.trainable_names, self.trainable_mods,
                        excluded_layer=final_layer,
                    )
                init_done = True

            # 4) Vote according to mode
            if cfg.voting_mode == "coarse":
                opt_triplets = important_neurons_dict[optimal_method]
                opt_triplets = sorted(opt_triplets, key=lambda t: t[1], reverse=True)
                layer_index_pairs = [(layer, idx) for (layer, _score, idx) in opt_triplets]
                _voting_neurons(layer_index_pairs, layer_scores)
            else:
                effective_top_k = top_m_neurons
                if cfg.selection_mode == "per-group" and cfg.is_yolo:
                    # Each group contributes top_m//3 → total ≈ top_m
                    effective_top_k = max(1, top_m_neurons // 3) * 3
                top_across = _weighted_top_neurons(
                    important_neurons_dict, loss_gains, top_k=effective_top_k
                )
                layer_index_pairs = [pair for (pair, _wscore) in top_across]
                _voting_neurons(layer_index_pairs, layer_scores)

            # Periodic checkpoint
            if checkpoint_path and (batch_idx + 1) % checkpoint_every == 0:
                torch.save(
                    {"layer_scores": layer_scores, "batch_idx": batch_idx},
                    checkpoint_path,
                )

        # 5) Save CSV
        # Remove checkpoint after successful completion
        if checkpoint_path and os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)
        out_csv = save_layer_scores_csv(layer_scores, cfg.out_csv)
        return layer_scores, out_csv

# wisdom/attribution/captum_backend.py
from typing import Dict, Tuple, List, Iterable, Optional, Callable
import torch
from torch.utils.data import DataLoader
from captum.attr import (
    LayerConductance, LayerActivation, InternalInfluence, LayerGradientXActivation,
    LayerGradCam, LayerDeepLift, LayerDeepLiftShap, LayerGradientShap,
    LayerIntegratedGradients, LayerFeatureAblation, LayerLRP
)

from .base import AttributionMethod
from ..utils.common import reduce_to_neuron_vector  # helper to collapse conv (C,H,W)->(C,)


ATTRS = {
    "lc": LayerConductance, 
    "la": LayerActivation, 
    "ii": InternalInfluence,
    "lgxa": LayerGradientXActivation, 
    "lgc": LayerGradCam, 
    "ldl": LayerDeepLift,
    "ldls": LayerDeepLiftShap, 
    "lgs": LayerGradientShap, 
    "lig": LayerIntegratedGradients,
    "lfa": LayerFeatureAblation, 
    "lrp": LayerLRP,
}

def per_layer_scores(model: torch.nn.Module,
                     dataloader: DataLoader,
                     device: str,
                     method: str) -> Dict[str, torch.Tensor]:
    """Return {layer_name: relevance per neuron (C,) or (out_channels,)} over the dataloader."""
    model = model.to(device).eval()
    scores: Dict[str, torch.Tensor] = {}
    # discover candidate layers
    layers: List[Tuple[str, torch.nn.Module]] = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
    ]
    for name, layer in layers:
        A = ATTRS[method](model, layer)
        agg = None
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if method in ("ldl", "ldls", "lgs"):
                attr = A.attribute(images, baselines=torch.zeros_like(images), target=labels)
            elif method == "la":
                attr = A.attribute(images)
            else:
                attr = A.attribute(images, target=labels)
            if attr.dim() == 4:
                part = attr.sum(dim=(0,2,3)).detach().cpu()
            else:
                part = attr.sum(dim=0).detach().cpu()
            agg = part if agg is None else (agg + part)
        if isinstance(layer, torch.nn.Linear):
            scores[name] = agg / len(dataloader.dataset)
        else:
            scores[name] = agg / len(dataloader.dataset)
    return scores


def _accumulate_layer_attr(
    A_obj, call_attr: Callable, model, layer, dataloader, device, target_is_label=True
) -> torch.Tensor:
    agg = None
    model.eval()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device) if target_is_label else None
        # call the provided function to compute attributions
        attr = call_attr(A_obj, images, labels)
        # collapse batch then spatial dims to per-neuron vector
        # attr shapes: (B, C) or (B, C, H, W)
        if attr.dim() == 4:
            vec = attr.sum(dim=(0,2,3)).detach().cpu()
        else:
            vec = attr.sum(dim=0).detach().cpu()
        agg = vec if agg is None else (agg + vec)
    return agg / len(dataloader.dataset)


class CaptumLayerAttribution(AttributionMethod):
    """
    Generic wrapper for Captum's per-layer attribution classes
    """
    def __init__(self, name: str, ctor, call):
        self.name = name.lower()
        self._ctor = ctor
        self._call = call

    def _candidate_layers(self, model) -> Iterable[tuple[str, torch.nn.Module]]:
        for n, m in model.named_modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                yield n, m

    def attribute_model(
        self, model, dataloader, device="cuda:0", target_layers: Optional[Iterable[str]] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        model = model.to(device)
        names = set(target_layers) if target_layers else None
        out: Dict[str, torch.Tensor] = {}
        for lname, layer in self._candidate_layers(model):
            if names and lname not in names:
                continue
            A = self._ctor(model, layer)
            out[lname] = _accumulate_layer_attr(A, self._call, model, layer, dataloader, device)
        return out

    def attribute_layer(self, model, layer_name, dataloader, device="cuda:0", **kwargs) -> torch.Tensor:
        model = model.to(device)
        layer = dict(model.named_modules())[layer_name]
        A = self._ctor(model, layer)
        return _accumulate_layer_attr(A, self._call, model, layer, dataloader, device)

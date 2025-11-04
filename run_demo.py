import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import yaml
import logging
import copy
import time
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from captum.attr import LayerLRP, LayerGradientXActivation, LayerActivation
from captum.attr import visualization as viz
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# from zennit.composites import EpsilonPlus
# from zennit.attribution import Gradient
# from lxt.efficient import monkey_patch

data_yaml = 'eterry/data.yaml'  # path to data.yaml
device = 'cuda'                    # cuda or cpu
imgsz = 640                        # inference size (pixels)
batch = 8                          # batch size
global_topk = 16                   # select top-K neurons globally
weights_path = '/scratch/staff/lrr550/yolo-dev/yolov5-eterry-detection/runs/train/exp/weights/best.pt'

# ---------------------------
# Helper
# ---------------------------
def configure_logging(level='info', enable_logging=False):
    if not enable_logging:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        return logging.getLogger(__name__)
    log_level = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "crit": logging.CRITICAL,
        }.get(level.lower(), logging.INFO)
        
    start_ms = int(time.time() * 1000)
    timestamp = time.strftime("%Y%m%d‑%H%M%S", time.localtime(start_ms / 1000))
    logfile = f"debugmode-{timestamp}.log"

    logger = logging.getLogger("Wisdom")
    logger.setLevel(log_level)
        
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_model(load_model_path='./models_info/saved_models/lenet_CIFAR10_whole.pth'):
    module_name = []
    module = []
    model = torch.load(load_model_path, weights_only=False)
    
    # Alternatively, to get all submodule names (including nested ones)
    for name, layer in model.named_modules():
        module_name.append(name)
        module.append(layer)

    return model, module_name, module

def get_trainable_modules_main(model, prefix=''):
    
    trainable_module = []
    trainable_module_name = []
    
    def get_trainable_modules(model, prefix=''):
        for name, layer in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)) and any(p.requires_grad for p in layer.parameters()):
                trainable_module_name.append(full_name)
                trainable_module.append(layer)
            get_trainable_modules(layer, full_name)
    get_trainable_modules(model)
    return trainable_module, trainable_module_name

def replace_silu_with_relu(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_silu_with_relu(child)
    return module

# ---------------------------
# Data
# ---------------------------
class SimpleImageFolder(Dataset):
    def __init__(self, images_dir: str, size: int = 640):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.paths = sorted([p for e in exts for p in glob.glob(os.path.join(images_dir, e))])
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return x, p

def collate(batch):
    xs, ps = zip(*batch)
    return torch.stack(xs, 0), ps


# ---------------------------
# Model & forward target
# ---------------------------
def load_yolov5(device: str):
    # raw nn.Module (no AutoShape), keeps grads & raw outputs
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False)
    # /scratch/staff/lrr550/yolo-dev/yolov5-eterry-detection/runs/train/exp/weights/best.pt
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, trust_repo=True, force_reload=True, autoshape=False)
    model.to(device).eval()
    return model

@torch.enable_grad()
def forward_scores(model: nn.Module, x: torch.Tensor, target_class: int | None) -> torch.Tensor:
    """
    Returns a per-sample scalar for Captum.
    Using max class logit across anchors; or a fixed class if provided.
    """
    y = model(x)
    if isinstance(y, (list, tuple)):  # older v5 may return a list
        y = y[0]
    # y: [B, N, 5+nc]
    cls = y[..., 5:]
    if target_class is None:
        per_anchor = cls.max(dim=-1).values  # [B, N]
    else:
        per_anchor = cls[..., int(target_class)]  # [B, N]
    return per_anchor.max(dim=-1).values  # [B]

# ---------------------------
# Layer selection (Conv & Linear)
# ---------------------------
def list_target_layers(model: nn.Module) -> List[Tuple[str, nn.Module, str]]:
    """
    Returns (name, module, kind) with kind in {'conv','linear'}.
    """
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            out.append((name, m, 'conv'))
        elif isinstance(m, nn.Linear):
            out.append((name, m, 'linear'))
    return out

# ---------------------------
# Attribution per layer (Conv channels or Linear neurons)
# ---------------------------
@torch.no_grad()
def _shape_info(m: nn.Module, x: torch.Tensor) -> Tuple[int, ...]:
    _y = m(x)
    return tuple(_y.shape)

def lrp_scores_layer(
    model: nn.Module,
    layer: nn.Module,
    kind: str,
    loader: DataLoader,
    device: str,
    train_limit_batches: int | None = None
) -> np.ndarray:
    """
    Aggregation = sum over batch (and spatial for conv) of |LRP|.
    """
    grad_explainer = LayerGradientXActivation(lambda z: forward_scores(model, z, 0), layer)

    agg = None
    steps = 0
    for xb, yb in loader:
        xb = xb.to(device).requires_grad_(True)
        attr = grad_explainer.attribute(xb)  # shape: conv->[B,C,H,W], linear->[B, out]
        breakpoint()
        if kind == 'conv':
            batch_scores = attr.abs().sum(dim=(0, 2, 3))  # [C]
        else:  # linear
            batch_scores = attr.abs().sum(dim=0)          # [out]
        agg = batch_scores if agg is None else agg + batch_scores
        steps += 1
        if train_limit_batches and steps >= train_limit_batches:
            break
    return agg.detach().float().cpu().numpy()

# ---------------------------
# Activation collectors for selected neurons
# ---------------------------
def collect_activations(
    model: nn.Module,
    layer: nn.Module,
    kind: str,
    indices: List[int],
    loader: DataLoader,
    device: str,
    limit_batches: int | None = None
) -> Dict[int, List[float]]:
    """
    For each selected neuron:
      conv: mean over HxW for target channel per sample
      linear: the pre-activation output (layer output) for that neuron per sample
    """
    feats: Dict[str, torch.Tensor] = {}
    def hook(_m, _in, out):
        feats['y'] = out.detach()

    h = layer.register_forward_hook(hook)
    store = {i: [] for i in indices}
    steps = 0

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            _ = model(xb)
            y = feats['y']  # conv: [B,C,H,W] ; linear: [B, out]
            if kind == 'conv':
                yred = y.mean(dim=(2, 3))  # [B, C]
            else:
                yred = y                   # [B, out]
            for idx in indices:
                store[idx].extend(yred[:, idx].float().cpu().tolist())
            steps += 1
            if limit_batches and steps >= limit_batches:
                break

    h.remove()
    return store


# ---------------------------
# Clustering per neuron
# ---------------------------
def cluster_per_neuron(
    vals: Dict[int, List[float]],
    use_silhouette: bool = True,
    min_k: int = 2,
    max_k: int = 5,
    fixed_k: int = 3,
    random_state: int = 0
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for idx, seq in vals.items():
        X = np.asarray(seq, dtype=np.float32).reshape(-1, 1)
        if use_silhouette and X.shape[0] >= 10:
            best = None
            best_score = -1e9
            for k in range(min_k, max_k + 1):
                km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
                lab = km.labels_
                if len(set(lab)) > 1:
                    sc = silhouette_score(X, lab)
                else:
                    sc = -1e9
                if sc > best_score:
                    best_score, best = sc, km
            km = best if best is not None else KMeans(n_clusters=fixed_k, n_init=10, random_state=random_state).fit(X)
        else:
            km = KMeans(n_clusters=fixed_k, n_init=10, random_state=random_state).fit(X)
        cents = np.sort(km.cluster_centers_.reshape(-1))
        out[idx] = {"k": len(cents), "centroids": cents}
    return out


# ---------------------------
# Coverage computation (exact or MC)
# ---------------------------
def coverage_exact(
    tuple_stream: List[Tuple[int, ...]],
    space_sizes: List[int]
) -> float:
    total = 1
    for s in space_sizes:
        total *= s
    seen = set(tuple_stream)
    return len(seen) / max(1, total)

def coverage_mc(
    tuple_stream: List[Tuple[int, ...]],
    space_sizes: List[int],
    num_samples: int = 20000,
    seed: int = 0
) -> float:
    # Estimate coverage by sampling combos uniformly from the space.
    rng = np.random.default_rng(seed)
    total = 1
    for s in space_sizes:
        total *= s
    if total <= 0:
        return 0.0
    sample = [
        tuple(rng.integers(0, s, size=1)[0] for s in space_sizes)
        for _ in range(min(num_samples, total))
    ]
    seen = set(tuple_stream)
    hit = sum(1 for t in sample if t in seen)
    return hit / len(sample)

# ---------------------------
# Pruning pipeline
# ---------------------------
def prune_model():
    pass

# ---------------------------
# Visualization
# ---------------------------
def visualize_neuron_clusters():
    pass

# ---------------------------
# Pipeline
# ---------------------------
def run_pipeline(
    data_yaml: str,
    device: str = "cuda",
    imgsz: int = 640,
    batch: int = 8,
    global_topk: int = 16,
):
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    train_images = cfg["train"]
    train_loader = DataLoader(SimpleImageFolder(train_images, imgsz), batch_size=batch, shuffle=False, num_workers=4, collate_fn=collate)
    test_loader  = DataLoader(SimpleImageFolder(train_images, imgsz), batch_size=batch, shuffle=False, num_workers=4, collate_fn=collate)
    model = load_yolov5(device)

    # 1) score all Conv & Linear layers
    layer_list = list_target_layers(model)  # (name, module, kind)
    all_rows = []
    for lname, layer, kind in layer_list:
        print(f"Scoring layer: {lname} ({kind})")
        scores = lrp_scores_layer(
            model, layer, kind, train_loader, device, 4
        )
        # Save per-layer CSV
        df = pd.DataFrame({
            "LayerName": [lname]*len(scores),
            "Kind": [kind]*len(scores),
            "NeuronIndex": list(range(len(scores))),
            "Score": scores.tolist()
        })
        # print(f"  -> layer {lname} | top scores: {np.sort(scores)[-5:]}")
        all_rows.append(df)
    
    big = pd.concat(all_rows, ignore_index=True)
    big.to_csv("yolov5_neuron_scores.csv", index=False)
    print(f"[Score] saved all neuron scores -> yolov5_neuron_kind_scores.csv")
    # 2) global Top-K selection (optionally cap per layer to avoid single layer dominating)
    
    per_layer_cap = None  # e.g. 4
    if per_layer_cap is not None:
        big["abs_score"] = big["Score"].abs()
        big = big.sort_values("abs_score", ascending=False)
        # per-layer cap:
        kept = []
        per_layer_count: Dict[str, int] = {}
        for _, row in big.iterrows():
            key = f"{row.LayerName}|{row.Kind}"
            if per_layer_count.get(key, 0) < per_layer_cap:
                kept.append(row)
                per_layer_count[key] = per_layer_count.get(key, 0) + 1
            if len(kept) >= global_topk:
                break
        top_df = pd.DataFrame(kept)
    else:
        top_df = big.reindex(big["Score"].abs().sort_values(ascending=False).index).head(global_topk)

    top_df = top_df.reset_index(drop=True)
    top_df_path = "global_topk_kind_neurons.csv"
    top_df.to_csv(top_df_path, index=False)
    print(f"[Select] saved global Top-K -> {top_df_path}")

    # 3) collect activations for selected neurons (train set)
    # group by layer
    selected_map: Dict[Tuple[str, str], List[int]] = {}
    for _, r in top_df.iterrows():
        key = (r["LayerName"], r["Kind"])
        selected_map.setdefault(key, []).append(int(r["NeuronIndex"]))

    clusters_per_neuron: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for (lname, kind), idxs in selected_map.items():
        layer = dict(model.named_modules())[lname]
        print(f"[Collect] {lname} ({kind}) channels: {idxs}")
        vals = collect_activations(
            model, layer, kind, idxs, train_loader, device, limit_batches=16
        )
        clust = cluster_per_neuron(vals, use_silhouette=False, fixed_k=2)
        # Save centroids
        rows = []
        for ch, info in clust.items():
            clusters_per_neuron[(lname, kind, ch)] = info
            rows.append({"LayerName": lname, "Kind": kind, "NeuronIndex": ch, "k": info["k"], "centroids": ";".join([f"{c:.6f}" for c in info["centroids"]])})
        pd.DataFrame(rows).to_csv(f"{lname.replace('.', '_')}_{kind}_clusters.csv", index=False)

    # 4) coverage on test set (joint space of all selected neurons)
    # prepare ordered list of (lname, kind, idx) and space sizes
    sel_neurons = []
    space_sizes = []
    for _, r in top_df.iterrows():
        key = (r["LayerName"], r["Kind"], int(r["NeuronIndex"]))
        sel_neurons.append(key)
        space_sizes.append(int(clusters_per_neuron[key]["k"]))

    print(f"[Coverage] selected neurons: {len(sel_neurons)}, space sizes={space_sizes}")

    # create test tuples
    tuples: List[Tuple[int, ...]] = []
    # Pre-build hooks per layer to avoid repeated registration
    hooks = []
    feat_map: Dict[str, torch.Tensor] = {}
    def make_hook(key_name: str):
        def _hook(_m, _in, out):
            feat_map[key_name] = out.detach()
        return _hook

    # register one hook per involved layer
    involved_layers = {}
    for lname, kind, _ in sel_neurons:
        involved_layers[lname] = (dict(model.named_modules())[lname], kind)
    for lname, (layer, kind) in involved_layers.items():
        hooks.append(layer.register_forward_hook(make_hook(lname)))

    with torch.no_grad():
        steps = 0
        for xb, _ in test_loader:
            xb = xb.to(device)
            _ = model(xb)
            B = xb.size(0)
            # compute per-sample tuple over selected neurons
            # cache reduced activations per involved layer
            red: Dict[str, torch.Tensor] = {}
            for lname, (_, kind) in involved_layers.items():
                fm = feat_map[lname]
                if kind == 'conv':
                    red[lname] = fm.mean(dim=(2, 3))  # [B, C]
                else:
                    red[lname] = fm                    # [B, out]
            for b in range(B):
                idxs = []
                for lname, kind, nidx in sel_neurons:
                    val = float(red[lname][b, nidx].item())
                    cents = clusters_per_neuron[(lname, kind, nidx)]["centroids"]
                    near = int(np.argmin(np.abs(cents - val)))
                    idxs.append(near)
                tuples.append(tuple(idxs))
            steps += 1

    # remove hooks
    for h in hooks:
        h.remove()

    cov = coverage_exact(tuples, space_sizes)

    # Save tuples (optional & heavy)
    pd.DataFrame({"tuple": [",".join(map(str, t)) for t in tuples]}).to_csv("test_tuple_indices.csv", index=False)
    summary = {
        "coverage": cov,
        "num_selected_neurons": len(sel_neurons),
    }
    pd.DataFrame([summary]).to_csv("e2e_coverage_summary.csv", index=False)
    print(f"[DONE] | saved -> {'e2e_coverage_summary.csv'}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    run_pipeline(
        data_yaml=data_yaml,
        device=device,
        imgsz=imgsz,
        batch=batch,
        global_topk=global_topk
    )

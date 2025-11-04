from pathlib import Path
import sys
from copy import deepcopy
from tkinter.font import names
from tqdm import tqdm
import os, csv, tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import pandas as pd
import yaml
import numpy as np
from ultralytics import YOLO

from captum.attr import (
    LayerConductance, LayerActivation, InternalInfluence, LayerGradientXActivation,
    LayerGradCam, LayerDeepLift, LayerDeepLiftShap, LayerGradientShap,
    LayerIntegratedGradients, LayerFeatureAblation, LayerLRP
)

from wisdom.pruning.mask_pruning import mask_model_neurons   # returns handles
from wisdom.pruning.weights_pruning import prune_model_neurons  # in-place permanent

yolo_dir = Path("/scratch/staff/lrr550/Wisdom_dev_trans/Wisdom/standalone/ultralytics_yolov5").resolve()
if str(yolo_dir) not in sys.path:
    sys.path.insert(0, str(yolo_dir))

from val import run as yolo_val_run
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_dataset,
    check_yaml,
    colorstr,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.general import non_max_suppression


data_yaml = "/scratch/staff/lrr550/Wisdom_dev_trans/Wisdom/standalone/data/coco128.yaml"
weights_path = "./weights/yolov5s.pt"
imgsz = 640
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_ids = [0, 1, 2, 3]
batch_size = 1
iou = 0.65
conf = 0.001

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

# DDP
def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank, world_size

def cleanup_distributed():
    dist.destroy_process_group()


# Basic utility
def unwrap_yolov5(m: nn.Module) -> nn.Module:
    base = getattr(m, "model", m)
    # base = getattr(base, "model", base)
    return base

def iter_target_layers(model: nn.Module) -> Iterable[Tuple[str, nn.Module, str]]:
    modules = list(model.named_modules())
    for name, layer in modules:
    # for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            yield name, layer, "conv"
        elif isinstance(layer, nn.Linear):
            yield name, layer, "linear"

def load_importance_csv(csv_path: str) -> Dict[str, List[Tuple[int, float]]]:
    df = pd.read_csv(csv_path)
    df["NeuronIndex"] = df["NeuronIndex"].astype(int)
    return {
        lname: list(zip(g["NeuronIndex"].tolist(), g["Score"].astype(float).tolist()))
        for lname, g in df.groupby("LayerName")
    }

def write_importance_csv(rows: List[Tuple[str, int, float]], out_csv: str):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["LayerName", "NeuronIndex", "Score"])
        w.writerows(rows)

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

# @torch.enable_grad()
# def forward_scores(model, x, target_class_idx):
#     # x is a batch tensor (B, C, H, W)
#     results_list = model(x)  # returns list of Results objects (one per image)
#     # Process each Results
#     scores = []
#     for res in results_list:
#         # Use res.boxes or res.boxes.conf and res.boxes.cls
#         boxes = res.boxes  # Boxes object
#         confs = boxes.conf
#         cls_ids = boxes.cls
#         # Now pick the target class
#         mask = (cls_ids == target_class_idx)
#         # Example: compute sum/confidence for that class
#         score = confs[mask].sum().item()
#         scores.append(score)
#     return torch.tensor(scores, device=x.device)

def compute_importance_for_layer(
    attribution_name: str, model: nn.Module, layer: nn.Module, kind: str, loader, max_batches: int | None = None, device='cuda:0'
) -> torch.Tensor:
    
    # Set up Captum attribution method
    key = attribution_name.lower()
    if key not in ATTRS:
        raise ValueError(f"Unknown Captum method '{attribution_name}'. Options: {list(ATTRS.keys())}")
    attribution_method = ATTRS[key]
    
    model.eval()
    # expl = attribution_method(lambda z: forward_scores(model, z, 0), layer)
    expl = attribution_method(lambda z: forward_scores(model, z, 0), layer)
    agg = None; steps = 0
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        if not torch.is_tensor(imgs): 
            continue
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        imgs = imgs.requires_grad_(True)
        attr = expl.attribute(imgs)
        # attr = expl.attribute(imgs.to(device))
        score = attr.abs().sum(dim=(0,2,3)) if kind == "conv" else attr.abs().sum(dim=0)
        agg = score if agg is None else agg + score
        steps += 1
        if max_batches and steps >= max_batches:
            break
    if agg is None:
        raise RuntimeError("No batches processed for attribution.")
    return agg.detach().cpu()

def compute_all_importances_csv(model: nn.Module, loader, out_csv: str, max_batches: int | None = 50, attribution_name='lgxa', device='cuda:0'):
    # model = unwrap_yolov5(model).to(device).eval()
    for p in model.parameters(): 
        p.requires_grad_(True)

    rows = []
    for lname, layer, kind in iter_target_layers(model):
        imp = compute_importance_for_layer(attribution_name, model, layer, kind, loader, max_batches=max_batches, device=device)
        torch.cuda.empty_cache()
        for idx, val in enumerate(imp.tolist()):
            rows.append((lname, idx, float(val)))
    write_importance_csv(rows, out_csv)
    return out_csv

def compute_importance_for_layer_ddp(model, layer, loader: DataLoader, device, max_batches=None):
    expl = LayerGradientXActivation(
        lambda z: forward_scores(model, z, 0),
        layer,
        device_ids=[device.index]
    )
    model.eval()
    agg = None
    steps = 0
    for imgs, *_ in loader:
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        imgs.requires_grad_(True)
        attr = expl.attribute(imgs)
        # scoring depending on layer kind, assume conv for example
        score = attr.abs().sum(dim=(0,2,3))
        if agg is None:
            agg = score
        else:
            agg = agg + score
        steps += 1
        if max_batches and steps >= max_batches:
            break
        del attr
        torch.cuda.empty_cache()
    return agg.detach().cpu()

def compute_all_importances_csv_ddp(model, rank, world_size, device, train_loader, max_batches=50):
    layers = []
    for name, layer in model.module.named_modules():
        if isinstance(layer, nn.Conv2d):
            layers.append((name, layer))
    # Only process a subset of layers per rank to split work
    layers_for_this_rank = layers[rank::world_size]
    rows = []
    for (lname, layer) in layers_for_this_rank:
        imp = compute_importance_for_layer(model, layer, train_loader, device, max_batches=max_batches)
        for idx, val in enumerate(imp.tolist()):
            rows.append((lname, idx, float(val)))
    # Each rank writes its own CSV (or gather and write in rank 0)
    out_csv = f"importances_rank{rank}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = torch.csv.writer if hasattr(torch.csv, 'writer') else None
        w = csv.writer(f)
        w.writerow(["LayerName","NeuronIndex","Score"])
        w.writerows(rows)
    if rank == 0:
        print("Rank 0 done. You may merge CSVs from all ranks.")
    cleanup_distributed()

def cal_scores(data_yaml, attribution_name='lgxa', yolo_name="yolov5"):    
    data = "/scratch/staff/lrr550/Wisdom_dev_trans/Wisdom/standalone/data/coco128.yaml"
    if yolo_name == 'yolov5':
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, autoshape=False, trust_repo=True)
    else:
        print("With YOLOv11 nano")
        model = YOLO("yolo11n.pt")
    
    # device, rank, world_size = setup_distributed()
    device = 'cuda:0'
    model = model.to(device)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)
    
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    data_dict = yaml.safe_load(open(check_yaml(data_yaml), "r"))
    train_path, val_path = data_dict["train"], data_dict["val"]
    train_loader, dataset = create_dataloader(
            train_path,
            imgsz,
            batch_size,
            stride=32, # 32 is the default stride for yolov5
            single_cls=False,
            hyp=None,
            augment=False,
            cache=False,
            rect=False, # Use rectangular training. Defaults to False.
            rank=-1,
            # rank=rank,
            image_weights=False, # Use weighted image selection for training. Defaults to False.
            quad=False, # Use quadrilateral training. Defaults to False.
            shuffle=True,
            seed=42)
    
    out_csv_name = f"{yolo_name}_neuron_scores_{attribution_name}.csv"
    out_csv = compute_all_importances_csv(model=model, loader=train_loader, out_csv=out_csv_name, max_batches=50, attribution_name=attribution_name, device=device)
    print(f"Important scores saved: {out_csv}")
    
if __name__ == "__main__":
    demo_attr = ["lc", "la", "ii", "lgxa", "lgc", "ldl", "ldls", "lgs", "lig", "lfa", "lrp"]
    data_yaml = "/scratch/staff/lrr550/Wisdom_dev_trans/Wisdom/standalone/data/coco128.yaml"
    cal_scores(data_yaml, attribution_name=demo_attr[6], yolo_name="yolov5")

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
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt

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
from train import run as yolo_train_run
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
top_k_neurons = 8
imgsz = 640
device = "cuda:2" if torch.cuda.is_available() else "cpu"
# device = "cpu"
batch_size = 2
iou = 0.65
conf = 0.001
prune_list = [100, 200, 300, 400, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1500, 2000, 2500, 3000, 4000]
results_csv = Path("./logs/prune_demo/neuron_prune_results_4.csv")
device_ids = [0,1,2,3] 

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

# attribution_name = "la"


# ---------------------------------
# Visualization of strategies
# ---------------------------------
def viz_strategy(k_values, strategies, results):        
    ks = sorted(k_values)
    for strategy in strategies:
        means50 = [results[(strategy,k)]["mean_map50"] for k in ks]
        stds50  = [results[(strategy,k)]["std_map50"]  for k in ks]
        plt.errorbar(ks, means50, yerr=stds50, label=f"{strategy} mAP50")
    plt.xlabel("# pruned neurons")
    plt.ylabel("mAP50")
    plt.legend()
    plt.title("mAP50 vs pruned neurons")
    plt.figure()
    for strategy in strategies:
        means95 = [results[(strategy,k)]["mean_map95"] for k in ks]
        stds95  = [results[(strategy,k)]["std_map95"]  for k in ks]
        plt.errorbar(ks, means95, yerr=stds95, label=f"{strategy} mAP95")
    plt.xlabel("# pruned neurons")
    plt.ylabel("mAP95")
    plt.legend()
    plt.title("mAP95 vs pruned neurons")
    plt.savefig("neuron_prune_strategies.pdf", bbox_inches="tight", dpi=1200, format="pdf")

# ---------------------------------
# Utility Functions
# ---------------------------------

def unwrap_yolov5(m: nn.Module) -> nn.Module:
    base = getattr(m, "model", m)
    # base = getattr(base, "model", base)
    return base

def iter_target_layers(model: nn.Module) -> Iterable[Tuple[str, nn.Module, str]]:
    for name, layer in model.named_modules():
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

# ---------------------------------
# Standalone prune map generators
# ---------------------------------
def load_scores(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure correct types
    df['NeuronIndex'] = df['NeuronIndex'].astype(int)
    return df

def build_prune_map_random(df, k):
    sel = df.sample(n=k, replace=False)
    prune_map = {}
    for _, row in sel.iterrows():
        prune_map.setdefault(row['LayerName'], []).append(row['NeuronIndex'])
    return prune_map

def build_prune_map_prune_top_k(df, k):
    sel = df.sort_values('Score', ascending=False).head(k)
    prune_map = {}
    for _, row in sel.iterrows():
        prune_map.setdefault(row['LayerName'], []).append(row['NeuronIndex'])
    return prune_map

def build_prune_map_prune_bottom_k(df, k):
    sel = df.sort_values('Score', ascending=True).head(k)
    prune_map = {}
    for _, row in sel.iterrows():
        prune_map.setdefault(row['LayerName'], []).append(row['NeuronIndex'])
    return prune_map

def build_prune_map_keep_top_k(df, k):
    df_sorted = df.sort_values('Score', ascending=False)
    topk = set(zip(df_sorted.head(k)['LayerName'], df_sorted.head(k)['NeuronIndex']))
    prune_map = {}
    for _, row in df_sorted.iterrows():
        if (row['LayerName'], row['NeuronIndex']) not in topk:
            prune_map.setdefault(row['LayerName'], []).append(row['NeuronIndex'])
    return prune_map

# ---------------------------------
# Simple evaluation function
# ---------------------------------
def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    # correct_class = labels[:, 0:1] == detections[:, 5:6]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

@torch.no_grad()
def evaluate_yolov5(model, data, dataloader, device='cuda', conf_thres=0.001, iou_thres=0.6, plots=False, single_cls=False, half=True, save_txt=False, save_dir=Path("")):
    model.eval()
    data = check_dataset(data)
    seen = 0
    max_det = 300
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    
    s = ("%22s" + "%11s" * 7) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP95", "mAP50-95")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    tp, fp, p, r, f1, mp, mr, map50, map95, ap50, ap95, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        im = im.to(device, non_blocking=True)
        targets = targets.to(device)
        im = im.half() if half else im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape
        preds, train_out = model(im, augment=False), None
        
        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        lb = []
        preds = non_max_suppression(preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det)
        
        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
        
        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred
        
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap95, ap = ap[:, 0], ap[:, 1], ap.mean(1)  # AP@0.5, AP@0.95, AP@0.5:0.95
        mp, mr, map50, map95, map = p.mean(), r.mean(), ap50.mean(), ap95.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 5  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map95, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING: no labels found in set, can not compute metrics without labels")
        
    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        
    # Return results
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else "" 
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")        
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    
    metrics = {
        "Precision": float(mp),
        "Recall": float(mr),
        "mAP50": float(map50),
        "mAP95": float(map95),
        "mAP50-95": float(map)
    }
    print(f"Precision: {mp:.4f}, Recall: {mr:.4f}, mAP50: {map50:.4f}, mAP95: {map95:.4f}, mAP50-95: {map:.4f}")

    return metrics
    # return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps

        
# Actual pruning function
def prune_model_by_neurons(model, prune_map):
    for name, module in model.named_modules():
        if name in prune_map:
            indices = prune_map[name]
            # Only handling Conv2d (you can extend to Linear or others if needed)
            if isinstance(module, torch.nn.Conv2d):
                with torch.no_grad():
                    # zero out the weights of these output channels
                    module.weight[indices, :, :, :] = 0
                    if module.bias is not None:
                        module.bias[indices] = 0
            else:
                print(f"Warning: layer {name} is {type(module)}, skip pruning by neurons for this type")
    return model

# ---------------------------------
# Saving the model for evaluation
# ---------------------------------   

def _as_1d_tensor(x):
    if isinstance(x, torch.Tensor): return x.reshape(-1)
    if isinstance(x, (list, tuple)): return torch.tensor(list(x))
    return torch.tensor([int(x)])

def normalize_yolo_attrs(model: nn.Module, data_yaml: str | None = None):
    """
    Make the cloned YOLOv5 Model look like a normal checkpoint-loaded Model:
      - model.nc: int
      - model.names: list[str] of length nc
      - model.stride: 1D tensor (e.g., [8,16,32])
      - Detect head's nc matches model.nc
    """
    # try to read names/nc from YAML if provided
    names_from_yaml = None
    if data_yaml and Path(data_yaml).exists():
        with open(data_yaml, "r") as f:
            dd = yaml.safe_load(f)
        if isinstance(dd.get("names"), list) and len(dd["names"]) > 0:
            names_from_yaml = dd["names"]

    # stride
    if hasattr(model, "stride"):
        model.stride = _as_1d_tensor(getattr(model, "stride"))
    else:
        try:
            det = model.model[-1]
            s = getattr(det, "stride", [32])
            model.stride = _as_1d_tensor(s)
        except Exception:
            model.stride = torch.tensor([32])

    # detect head
    det = None
    try:
        det = model.model[-1]
    except Exception:
        pass

    # nc / names
    nc = getattr(model, "nc", None)
    if nc is None or not isinstance(nc, int) or nc <= 0:
        if det is not None and hasattr(det, "nc") and isinstance(det.nc, int) and det.nc > 0:
            nc = det.nc
        elif names_from_yaml is not None:
            nc = len(names_from_yaml)
        else:
            nc = 80  # safe fallback

    model.nc = int(nc)

    names = getattr(model, "names", None)
    if not isinstance(names, list) or len(names) != model.nc:
        if names_from_yaml is not None and len(names_from_yaml) == model.nc:
            model.names = names_from_yaml
        else:
            model.names = [str(i) for i in range(model.nc)]

    # keep detect head consistent
    if det is not None and hasattr(det, "nc"):
        det.nc = model.nc

def strip_all_hooks(module: nn.Module):
    """Remove all forward/backward hooks from a module tree (in-place)."""
    for mod in module.modules():
        # clear dicts of registered hooks
        if hasattr(mod, "_forward_pre_hooks"): 
            mod._forward_pre_hooks.clear()
        if hasattr(mod, "_forward_hooks"):     
            mod._forward_hooks.clear()
        if hasattr(mod, "_backward_hooks"):    
            mod._backward_hooks.clear()

def build_eval_weights_from_model(model: nn.Module, selection: dict | None, pruning_mode: str) -> str:
    # base = unwrap_yolov5(model)
    eval_model = deepcopy(model).cpu().eval()
    strip_all_hooks(eval_model)

    # Ensure the callable fuse() method is not shadowed by an instance bool
    if hasattr(eval_model, "fuse") and not callable(getattr(eval_model, "fuse")):
        # optional: preserve it under a different name
        try:
            setattr(eval_model, "is_fused_flag", getattr(eval_model, "fuse"))
        except Exception:
            pass
        delattr(eval_model, "fuse")

    # If you masked the live model, mirror it here with permanent zeroing
    if selection:
        if pruning_mode == "mask":
            from wisdom.pruning.weights_pruning import prune_model_neurons
            prune_model_neurons(eval_model, selection)
        elif pruning_mode == "permanent":
            pass
        else:
            raise ValueError("pruning_mode must be 'mask' or 'permanent'")

    # Ensure YOLOv5-specific attrs are correct
    normalize_yolo_attrs(eval_model)
    
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    # Save exactly what YOLOv5 attempt_load expects
    torch.save({"model": eval_model}, tmp.name)
    return tmp.name

# ---------------------------------
# Gradient-based scoring
# ---------------------------------
# @torch.no_grad()
# def forward_scores(model: nn.Module, x: torch.Tensor, class_idx: int = 0) -> torch.Tensor:
#     # A simple scalar objective that propagates through the full model.
#     out = model(x)
#     if isinstance(out, (list, tuple)):
#         return sum(t.float().sum() for t in out if torch.is_tensor(t))
#     return out.float().sum()

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

def compute_importance_for_layer(
    attribution_name: str, model: nn.Module, layer: nn.Module, kind: str, loader, max_batches: int | None = None
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

def compute_all_importances_csv(model: nn.Module, loader, out_csv: str, max_batches: int | None = 50, attribution_name='lgxa'):
    # model = unwrap_yolov5(model).to(device).eval()
    for p in model.parameters(): 
        p.requires_grad_(True)

    rows = []
    for lname, layer, kind in iter_target_layers(model):
        imp = compute_importance_for_layer(attribution_name, model, layer, kind, loader, max_batches=max_batches)
        for idx, val in enumerate(imp.tolist()):
            rows.append((lname, idx, float(val)))
    write_importance_csv(rows, out_csv)
    return out_csv

# ---------------------------------
# Top-k pruning selection
# ---------------------------------
def topk_selection_per_layer(imp_map: Dict[str, List[Tuple[int, float]]], k_per_layer: int) -> Dict[str, List[int]]:
    selection: Dict[str, List[int]] = {}
    for lname, pairs in imp_map.items():
        # pairs: List[(neuron_idx, score)]
        top = sorted(pairs, key=lambda x: -x[1])[:k_per_layer]
        selection[lname] = [i for (i, s) in top]
    return selection

def apply_mask_then_eval(model: nn.Module, selection: Dict[str, List[int]]):
    handles = mask_model_neurons(model, selection)
    return handles

def apply_permanent_prune(model: nn.Module, selection: Dict[str, List[int]]):
    prune_model_neurons(model, selection)


# ---------------------------------
# Evaluation
# ---------------------------------
def yolo_val_map(weights_pt: str, project="logs/val", name="coco_eval"):
    yolov5_dir = Path("yolov5")
    sys.path.append(str(yolov5_dir))

    yolo_val_run(
        data=data_yaml,
        weights=[weights_pt],
        imgsz=imgsz,
        batch_size=batch_size,
        iou_thres=iou,
        conf_thres=conf,
        device=device,
        project=project,
        name=name,
        save_json=True,
        plots=False
    )
    # Read summary row
    results_csv = Path(project) / name / "results.csv"
    df = pd.read_csv(results_csv)
    last = df.iloc[-1]
    return {
        "precision": float(last["P"]),
        "recall": float(last["R"]),
        "mAP50": float(last["mAP50"]),
        "mAP95": float(last["mAP95"]),
        "mAP50-95": float(last["mAP50-95"]),
        "results_csv": str(results_csv),
    }

def save_tmp_weights_from_model(model: nn.Module) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    torch.save({"model": model}, tmp.name)  # YOLOv5 attempt_load can handle this
    return tmp.name

# ---------------------------------
# Entry Point
# ---------------------------------
def run_neuron_importance_experiment(
    *,
    coco_data_yaml: str,              # points to train2017 and val2017 (e.g., coco128.yaml adjusted)
    weights_path: str,                # e.g., 'yolov5s.pt' (COCO-pretrained)
    device: str = "cuda:0",
    use_csv_importance: bool = True,
    importance_csv_in: str | None = None,
    compute_max_batches: int | None = 50,  # used if computing Captum scores
    k_per_layer: int = 8,
    pruning_mode: str = "mask",       # 'mask' or 'permanent'
    out_dir: str = "neuron_eval_out",
    val_imgsz: int = 640,
    val_batch: int = 16,
) -> str:
    """
    Returns path to results CSV with columns:
    [mode, k_per_layer, pruning_mode, precision, recall, mAP50, mAP50-95, weights_used]
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, autoshape=False, trust_repo=True)
    model = model.to(device)
    
    # model = unwrap_yolov5(m).to(device).eval()
    for p in model.parameters(): 
        p.requires_grad_(True)

    # 2) Dataloaders for COCO2017
    data_dict = yaml.safe_load(open(check_yaml(data_yaml), "r"))
    train_path, val_path = data_dict["train"], data_dict["val"]
    print(f"Train images: {train_path}, Val images: {val_path}")
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
        image_weights=False, # Use weighted image selection for training. Defaults to False.
        quad=False, # Use quadrilateral training. Defaults to False.
        shuffle=True,
        seed=42)
    
    val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size,
            stride=32,
            single_cls=False,
            hyp=None,
            augment=False,
            cache=False,
            rect=False,
            rank=-1)[0]
    
    train_labels = np.concatenate(dataset.labels, 0)
    mlc = int(train_labels[:, 0].max())  # max label class
    print(f"{len(dataset)} images, {len(train_labels)} labels, {mlc} classes")
    
    # 3) Importance map
    if use_csv_importance:
        assert importance_csv_in is not None and Path(importance_csv_in).exists(), "CSV path missing."
        imp_map = load_importance_csv(importance_csv_in)   # {layer: [(idx,score),...]}
        importance_csv_used = importance_csv_in
    else:
        # compute with Captum on train2017 subset
        importance_csv_used = str(Path(out_dir) / "coco_train2017_importance.csv")
        compute_all_importances_csv(model, train_loader, importance_csv_used, max_batches=compute_max_batches)
        imp_map = load_importance_csv(importance_csv_used)

    print("Importance map loaded/computed.")
    
    # 4) Build selection (Top-K per layer)
    selection = topk_selection_per_layer(imp_map, k_per_layer=k_per_layer)

    # 5) Apply pruning & evaluate mAP on val2017
    tmp_weights_path = None
    
    try:
        tmp_weights_path  = build_eval_weights_from_model(model, selection, pruning_mode=pruning_mode)        
        # YOLOv5 validator reads dataset from YAML (must point to val2017)
        print(f"Evaluating {pruning_mode} pruned model with weights from: {tmp_weights_path}")
        metrics = yolo_val_map(weights_pt=tmp_weights_path, project="logs/runs_val", name=f"coco_prune_k{k_per_layer}_{pruning_mode}")

    finally:
        if tmp_weights_path and Path(tmp_weights_path).exists():
            try: 
                os.remove(tmp_weights_path)
            except Exception: 
                pass

    # 6) Save experiment row
    results_csv = str(Path(out_dir) / "prune_eval_results.csv")
    exists = Path(results_csv).exists()
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["mode","k_per_layer","pruning_mode","precision","recall","mAP50","mAP50-95","importance_csv","weights_used","val_results_csv"])
        w.writerow([
            "csv" if use_csv_importance else "captum",
            k_per_layer,
            pruning_mode,
            metrics["precision"], metrics["recall"], metrics["mAP50"], metrics["mAP50-95"],
            importance_csv_used,
            weights_path,
            metrics["results_csv"]
        ])
    return results_csv

# ---------------------------------
# Unit Test / Demo
# ---------------------------------

def simple_test():
    results = yolo_val_run(
        data="/scratch/staff/lrr550/Wisdom_dev_trans/Wisdom/standalone/data/coco128.yaml",
        weights=["/scratch/staff/lrr550/Wisdom_dev_trans/Wisdom/standalone/ultralytics_yolov5/yolov5s.pt"],
        imgsz=640,
        batch_size=16,
        device=0,
        iou_thres=0.65,
        conf_thres=0.001,
        project="logs/val",
        name="external_eval",
        save_json=True,
        plots=False,
    )
    # return results

# ---------------------------------
# Entry Point for pruning demo
# ---------------------------------

def prune_demo(data_yaml, csv_file: str, k_value: List[int] = [10, 20, 50], results_path=Path("prune_results.csv"), use_csv_importance=True, attribution_name='lgxa', total_runs: int = 5, epochs: int = 3, name: str = "fine_tune_demo", project: str = "./logs/prune_demo"):
    
    data = "/scratch/staff/lrr550/Wisdom_dev_trans/Wisdom/standalone/data/coco128.yaml"
    # model_orig = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, autoshape=False, trust_repo=True)
    # model_orig = model_orig.to(device)
    data_dict = yaml.safe_load(open(check_yaml(data_yaml), "r"))
    train_path, val_path = data_dict["train"], data_dict["val"]
    
    if use_csv_importance:
        assert csv_file is not None and Path(csv_file).exists(), "CSV path missing."
    else:
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
            image_weights=False, # Use weighted image selection for training. Defaults to False.
            quad=False, # Use quadrilateral training. Defaults to False.
            shuffle=True,
            seed=42)
        out_csv_name = f"yolov5_neuron_scores_{attribution_name}.csv"
        csv_file = compute_all_importances_csv(model=model_orig, loader=train_loader, out_csv=out_csv_name, max_batches=50, attribution_name=attribution_name)

    # Load scores
    df = load_scores(csv_file)

    # Basic setup
    results = {}
    val_loader = create_dataloader(
                val_path,
                imgsz,
                batch_size,
                stride=32,
                single_cls=False,
                hyp=None,
                augment=False,
                cache=False,
                rect=False,
                rank=-1)[0]
    
    # Write CSV header if not exists
    write_header = not results_path.exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Type", "Precision", "Std_Precision", "Recall", "Std_Recall", "mAP50", "Std_mAP50", "mAP95", "Std_mAP95", "mAP50-95", "Std_mAP50-95"])

        for k in k_value:
            for strategy_name, map_fn in [
                    ('baseline', None),
                    ('random', build_prune_map_random),
                    ('prune_top_k', build_prune_map_prune_top_k),
                    ('prune_bottom_k', build_prune_map_prune_bottom_k),
                    ('keep_top_k', build_prune_map_keep_top_k),
                ]:
                metrics_list = []
                for i in range(total_runs):
                    print(f"=== Run {i+1}/{total_runs} ===")
                    yolo_train_run(data=data, imgsz=640, weights=weights_path, epochs=epochs, name=name, project=project, batch_size=16, device=device)
                    model_orig = torch.hub.load('ultralytics/yolov5', 'custom', path=project+name+".pt", autoshape=False, trust_repo=True)
                    model_orig = model_orig.to(device)
                    
                    print(f"=== Running strategy {strategy_name} with k={k} ===")
                    
                    if strategy_name == 'baseline':
                        metrics = evaluate_yolov5(model_orig, data, val_loader, device=device, conf_thres=0.001, iou_thres=0.6, plots=False, single_cls=False, half=False, save_txt=False, save_dir=Path("./logs/prune_demo"))
                        metrics_list.append(metrics)
                        continue
                    
                    prune_map = map_fn(df, k)
                    model = deepcopy(model_orig)
                    model = prune_model_by_neurons(model, prune_map)
                    # model, data, dataloader, device='cuda', conf_thres=0.001, iou_thres=0.6, plots=False, single_cls=False, half=True
                    metrics = evaluate_yolov5(model, data, val_loader, device=device, conf_thres=0.001, iou_thres=0.6, plots=False, single_cls=False, half=False, save_txt=False, save_dir=Path("./logs/prune_demo"))
                    metrics_list.append(metrics)
                
                mean_precision = np.mean([m["Precision"] for m in metrics_list])
                std_precision = np.std([m["Precision"] for m in metrics_list] )
                mean_recall = np.mean([m["Recall"] for m in metrics_list])
                std_recall = np.std([m["Recall"] for m in metrics_list])
                mean_map50 = np.mean([m["mAP50"] for m in metrics_list])
                std_map50 = np.std([m["mAP50"] for m in metrics_list] )
                mean_map95 = np.mean([m["mAP95"] for m in metrics_list])
                std_map95 = np.std([m["mAP95"] for m in metrics_list] )
                mean_map50_95 = np.mean([m["mAP50-95"] for m in metrics_list])
                std_map50_95 = np.std([m["mAP50-95"] for m in metrics_list] )
                results[(strategy_name, k)] = {
                    "mean_precision": mean_precision, "std_precision": std_precision,
                    "mean_recall": mean_recall, "std_recall": std_recall,
                    "mean_map50": mean_map50, "std_map50": std_map50,
                    "mean_map95": mean_map95, "std_map95": std_map95,
                    "mean_map50_95": mean_map50_95, "std_map50_95": std_map50_95
                }
                breakpoint()
                writer.writerow([f"{strategy_name}_{k}", results["mean_precision"], results["std_precision"], results["mean_recall"], results["std_recall"], results["mean_map50"], results["std_map50"], results["mean_map95"], results["std_map95"], results["mean_map50_95"], results["std_map50_95"]])
                f.flush()

        viz_strategy(k_value, ['baseline', 'random', 'prune_top', 'prune_bottom', 'keep_top'], results)

if __name__ == "__main__":
    # simple_test()
    # run_neuron_importance_experiment(
    #     coco_data_yaml=data_yaml,
    #     weights_path=weights_path,
    #     device=device,
    #     use_csv_importance=True,
    #     importance_csv_in="yolov5_neuron_scores.csv",
    #     k_per_layer=top_k_neurons,
    #     pruning_mode="mask",  # or "permanent"
    #     out_dir="neuron_eval_out"
    # )
    demo_csv = "./neuron_eval_out/yolov5_neuron_scores_ii.csv"
    demo_attr = ["lc", "la", "ii", "lgxa", "lgc", "ldl", "ldls", "lgs", "lig", "lfa", "lrp"]
    # prune_demo(data_yaml, demo_csv, prune_list, results_path=results_csv, use_csv_importance=True, attribution_name=demo_attr[3])
    prune_demo(data_yaml, demo_csv, prune_list, results_path=results_csv, use_csv_importance=True, attribution_name=demo_attr[3], total_runs=2, epochs=2, name="fine_tune_temp", project="./logs/prune_demo2")

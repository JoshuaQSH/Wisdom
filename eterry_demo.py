# run_yolo_demo.py
import argparse
import os
import random
import time
import math
import json
import logging
import yaml
import glob
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from captum.attr import LayerGradientXActivation
from captum.attr import visualization as viz

from wisdom.core.wisdom import WisdomIDC
from wisdom.core.wisdom import WisdomConfig, ClusteringConfig
from wisdom.core.wisdom_train import ConsensusWisdom, WisdomTrainConfig

from wisdom.utils.io_cache import read_layer_scores_csv
from wisdom.utils.common import get_trainable_modules_main, get_model
from wisdom.utils.visulization import viz_topk_neurons_score


from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.plots import Annotator, colors
import cv2

data_yaml = 'eterry/data.yaml'  # path to data.yaml
device = 'cuda'                    # cuda or cpu
imgsz = 640                        # inference size (pixels)
batch = 4                          # batch size
global_topk = 16                   # select top-K neurons globally
weights_path = '/scratch/staff/lrr550/yolo-dev/yolov5-eterry-detection/runs/train/exp/weights/best.pt'
gaussian_STD = 0.5
TOPK = 0.05  # Top-k fraction of pixels to perturb (5%)
cmap = ['PuBuGn', 'Greens', 'Purples', 'Reds', 'Blues', 'YlGn', 'summer', 'cool', 'bwr']

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

# Small configuration
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
    
class YOLOFolder(Dataset):
    def __init__(self, images_dir: str, size: int = 640):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.img_paths = sorted([p for e in exts for p in glob.glob(os.path.join(images_dir, e))])
        if not self.img_paths:
            raise FileNotFoundError(f"No images under {images_dir}")
        self.size = size

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        img_path = Path(self.img_paths[i])
        # load & resize (replace with your transform)
        img = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0

        # load labels from matching txt
        labels_path = str(img_path).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        targets = []
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                for line in f:
                    c, x, y, w, h = map(float, line.split())
                    targets.append([int(c), x, y, w, h])  # normalized
        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0,5), dtype=torch.float32)

        return img, targets, str(img_path)

def yolo_collate(batch):
    imgs, targets, paths = zip(*batch)
    imgs = torch.stack(imgs, 0)                      # [B,3,H,W]
    # keep targets as a list of [Ni,5] tensors
    return imgs, list(targets), list(paths)

def collate(batch):
    xs, ps = zip(*batch)
    return torch.stack(xs, 0), ps

def load_yolov5(device: str):
    # raw nn.Module (no AutoShape), keeps grads & raw outputs
    # /scratch/staff/lrr550/yolo-dev/yolov5-eterry-detection/runs/train/exp/weights/best.pt
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, trust_repo=True, force_reload=True, autoshape=False)
    model.to(device).eval()
    return model

def wisdom_importance_scores(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df.Score != 0]
    layer2score = defaultdict(dict)
    for _, row in df.iterrows():
        layer2score[row.LayerName][int(row.NeuronIndex)] = float(row.Score)
    return layer2score

def add_gaussian_noise(imgs: torch.Tensor, mask: torch.Tensor,
                       mean: float = 0.0, std: float = 0.01):
    if std <= 0:
        out = imgs
    else:
        noise = torch.randn_like(imgs) * std + mean
        if mask.dtype is torch.bool:
            mask_f = mask.to(imgs.dtype)
        else:
            mask_f = mask
        # arithmetic selection (no boolean requirement)
        out = imgs * (1 - mask_f) + (imgs + noise) * mask_f
    return out.clamp(imgs.min(), imgs.max())

def register_hooks(model, csv_path):
    """
    For every layer that appears in the CSV we grab its output.
    Returns list of handles and a dict that will fill up per batch.
    """
    layer2score = wisdom_importance_scores(csv_path)
    activations = defaultdict(list)
    handles = []

    for name, module in model.named_modules():
        if name in layer2score:
            scores_for_layer = layer2score[name]

            def _make_hook(layer_name, score_dict):
                def _hook(_, __, out):
                    # out shape: (B, C, H, W) for conv; (B, C) for FC
                    acts = out.detach()
                    # conv → (B, C, H, W); fc → (B, C, 1, 1)
                    if acts.dim() == 2:
                        acts = acts.unsqueeze(-1).unsqueeze(-1)
                    score_vec = torch.zeros(acts.size(1), device=acts.device)
                    for idx, s in score_dict.items():
                        if idx < score_vec.size(0):
                            score_vec[idx] = s
                    weighted = acts * score_vec.view(1, -1, 1, 1)
                    activations[layer_name].append(weighted)
                return _hook

            handles.append(module.register_forward_hook(
                _make_hook(name, scores_for_layer)))
    return handles, activations

# -----------------------------------------------------------
# Data wrapper and generator - old version
# -----------------------------------------------------------
class LabelToIntDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dataset):
        # e.g., TensorDataset(image_tensor, label_tensor)
        self.dataset = tensor_dataset  
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # convert label from Tensor to int
        return x, y.item()  
    
    def __len__(self):
        return len(self.dataset)

def build_mask_wisdom(heat: torch.Tensor, k: float = 0.02, exclude_imp: bool = True) -> torch.Tensor:
    """
    heat: (B, 1, H, W) – relevance per pixel
    returns boolean mask where top-k fraction is True.
    """
    B, _, H, W = heat.shape
    flat = heat.view(B, -1).abs()
    kth = math.ceil((flat.size(1) * k))
    idx = flat.topk(kth, dim=1).indices
    
    # Important-based mask
    mask_imp_flat = torch.zeros_like(flat, dtype=torch.bool)
    mask_imp_flat = mask_imp_flat.scatter_(1, idx, True)
    mask_imp = mask_imp_flat.view(B, 1, H, W)
    
    # Random-based mask
    if exclude_imp:
        avail_mask = (~mask_imp_flat)
        # rand_idx = torch.multinomial(avail_mask, kth, replacement=False)
        rand_scores = torch.rand_like(flat, dtype=torch.float32)
        rand_scores[~avail_mask] = float('-inf')
        rand_idx = rand_scores.topk(kth, dim=1).indices
        mask_rand = torch.zeros_like(mask_imp_flat, dtype=torch.bool).scatter_(1, rand_idx, True)
        mask_rand = mask_rand.view_as(mask_imp)
    else:
        rand_scores = torch.rand_like(flat, dtype=torch.float32)
        rand_idx = rand_scores.topk(kth, dim=1).indices
        mask_rand = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, rand_idx, True)
        mask_rand = mask_rand.view_as(mask_imp)
        
    return mask_imp, mask_rand

def pertub_sets_demo_wisdom(logger, model, loader, device, csv_path, k, std):
    """
    Returns tensors (U_I, U_R, y) for a dataset, a WISDOM-based version.
    """
    handles, acts_dict = register_hooks(model, csv_path)
    U_I, U_R, y = [], [], []
    # for inputs, labels in tqdm(loader, desc=f"Attribution Gradients x Activations"):
    for inputs, labels, paths in tqdm(loader, desc=f"Attribution Gradients x Activations"):
        # inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.to(device)
        for ac_ in acts_dict:
            acts_dict[ac_].clear()
        
        # Forward pass to get activations
        preds = model(inputs)
        
        
        # Merge all layers: upsample to input resolution & sum
        heat = torch.zeros_like(inputs[:, :1]) # (B,1,H,W)
        for layer_name, batch_list in acts_dict.items():
            if not batch_list:
                continue
            acts = batch_list[0] # (B,C,h,w)
            if acts.size(2) != inputs.size(2):
                acts = F.interpolate(acts, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
            
            # sum across channels → (B,1,H,W)
            layer_heat = acts.sum(1, keepdim=True)
            heat += layer_heat
        
        mask_imp, mask_rand = build_mask_wisdom(heat, k=k, exclude_imp=True)
        
        # WISDOM-based important pixels
        inputs_I = add_gaussian_noise(inputs, mask_imp, std=std)

        # Random-based pixels
        inputs_R = add_gaussian_noise(inputs, mask_rand, std=std)
        
        top_16_layers = ['model.model.2.m.0.cv1.conv', 'model.model.2.cv1.conv', 'model.model.0.conv', 'model.model.2.m.0.cv1.conv', 'model.model.0.conv']
        for index, layer in enumerate(top_16_layers):
            viz_attr(model, inputs, layer_name=layer, tag=str(index))
            logger.info(f"[viz] saved to yolov5_layer_attr_heatmap_{index}.png")
        
        viz_attr_diff(logger, inputs[0], inputs_I[0], cmap=cmap[3], alpha=0.8, tag='importance')
        viz_attr_diff(logger, inputs[0], inputs_R[0], cmap=cmap[3], alpha=0.8, tag='random')

        break

def wisdom_prune(model, trainloader, logger):
    logger.info("ConsensusWisdom...")
    trainer = ConsensusWisdom(model, device=device)
    cfg = WisdomTrainConfig(
            methods=["la","ldl","lig"], # 3 options as examples here
            device=device,
            voting_mode="fine-grained",  # or "coarse"
            out_csv="wisdom_layer_scores_yolov5.csv",
        )
    
    layer_scores, out_csv = trainer.fit(
            train_loader=trainloader,
            cfg=cfg,
            top_m_neurons=10,
            final_layer=None,        # or "fc" / your last trainable layer name to exclude
            prune_mode="mask",       # "mask" (reversible) or "weights" (restored per batch)
        )

def last_conv_before_detect(model: nn.Module) -> tuple[str, nn.Module]:
    """
    Returns (name, module) of the last nn.Conv2d in the graph (which,
    in YOLOv5, is effectively the last conv before the Detect head).
    """
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            last = (name, mod)
    if last is None:
        raise RuntimeError("No Conv2d layers found.")
    return last

def viz_attr(model, input_img, layer_name, tag='1'):
    # layer_name, layer = last_conv_before_detect(model)
    # print(f"Using last conv layer: {layer_name}")
    layer = dict(model.named_modules())[layer_name]
    # explainer = LayerGradientXActivation(model, layer)
    explainer = LayerGradientXActivation(lambda z: forward_scores(model, z, 0), layer)

    attr = explainer.attribute(input_img)  
    attr = attr[0].detach().cpu()
    # attr_2d = attr.abs().sum(dim=0)           # [H,W]
    # attr_2d = (attr_2d - attr_2d.min()) / (attr_2d.max() - attr_2d.min() + 1e-8)
    original_image = np.transpose(input_img[0].cpu().numpy(), (1, 2, 0))  # [H,W,3]
    fig, ax = viz.visualize_image_attr(
        np.transpose(attr.numpy(), (1, 2, 0)),
        original_image=original_image,
        method="heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title="YOLOv5 Conv Layer Attribution",
    )

    fig.savefig(f"logs/yolov5_layer_attr_heatmap_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def viz_attr_diff(logger, img_orig, img_pert, cmap="bwr", alpha=0.8, tag='random'):
    
    # tensors -> HWC numpy
    o = img_orig.cpu().detach().numpy().transpose(1,2,0)
    p = img_pert.cpu().detach().numpy().transpose(1,2,0)

    d = np.abs(p - o)

    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].imshow(o.squeeze() if o.shape[2]==1 else o)
    ax[0].set_title("Original")
    ax[1].imshow(p.squeeze() if p.shape[2]==1 else p)
    ax[1].set_title(f"Perturbed ({tag})")
    ax[2].imshow(o.squeeze() if o.shape[2]==1 else o, alpha=1-alpha)
    ax[2].imshow(d.squeeze() if d.shape[2]==1 else d,
                 cmap=cmap, alpha=alpha)
    ax[2].set_title(f"Overlay ({tag})")
    for a in ax: a.axis("off")

    fig.tight_layout()
    out = f"logs/YOLOv5_TOP_{TOPK}_{tag}_diff.pdf"
    fig.savefig(out, dpi=1200, format='pdf', bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[viz] saved to {out}")

def wisdom_diff_viz(model, testloader, csv_file, logger):
    pertub_sets_demo_wisdom(logger, model, testloader, device, csv_file, TOPK, gaussian_STD)


def wisdom_end2end(model, trainloader, testloader, logger, csv_file):
    results_dict = {}
    if os.path.exists(csv_file):
        layer_scores = read_layer_scores_csv(csv_file)
        viz_topk_neurons_score(csv_file, top_k=global_topk)
        logger.info("Layer scores loaded from %s", csv_file)
    else:
        raise ValueError(f"Layer scores file {csv_file} not found.")
    
    cluster = ClusteringConfig(method="KMeans", params={"random_state": 42, "n_clusters": 2}, use_silhouette=False, k_max=10)
    cfg = WisdomConfig(top_m_neurons=global_topk, test_all_classes=True, cache_path=".wisdom_cache")
    idc = WisdomIDC(model, impl="LRP", cfg=cfg, cluster=cluster)
    selected = idc.select_top_neurons(layer_scores, exclude_last=None)
    logger.info("Selected top-%d neurons globally.", global_topk)
    idc.fit_clusters(trainloader, selected, device=device)
    coverage_rate, total_combination, max_coverage = idc.coverage(testloader, selected, device=device)
    logger.info("Attribution Method: %s", "WISDOM")
    logger.info("Total coverage combinations: %d", total_combination)
    logger.info("Max Coverage (the best we can achieve): %.6f%%", max_coverage * 100)
    logger.info("[WISDOM] Coverage Rate: %.6f%%", coverage_rate * 100)

def run():
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    train_images = cfg["train"]
    # trainloader = DataLoader(SimpleImageFolder(train_images, imgsz), batch_size=batch, shuffle=False, num_workers=4, collate_fn=collate)
    # testloader  = DataLoader(SimpleImageFolder(train_images, imgsz), batch_size=batch, shuffle=False, num_workers=4, collate_fn=collate)
    testloader = DataLoader(YOLOFolder(train_images, size=imgsz),
                    batch_size=batch, shuffle=False, num_workers=4,
                    collate_fn=yolo_collate)
    
    logger = configure_logging()
    model = load_yolov5(device)
    logger.info("Model: %s, Dataset: %s, Topk: %s", "YOLOv5s", "ETERRY", global_topk)
    wisdom_diff_viz(model, testloader, csv_file="yolov5_neuron_scores.csv", logger=logger)
    # wisdom_end2end(model, trainloader, testloader, logger, csv_file="yolov5_neuron_scores.csv")

if __name__ == "__main__":
    run()
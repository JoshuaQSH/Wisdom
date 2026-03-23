#!/usr/bin/env python
"""
sanity_check_rq2.py – Visual sanity check for RQ2 perturbation
==============================================================
Produces 5 images from a single COCO image:
  1. Original (clean)
  2. Random noise on OBJECT region
  3. Random noise on BACKGROUND region
  4. Importance-guided noise on OBJECT region
  5. Importance-guided noise on BACKGROUND region

"Importance-guided" means: compute the gradient of the summed
important-neuron activations w.r.t. the input pixels, then perturb
along that gradient direction (where the model is most sensitive).
This contrasts with "random" which perturbs uniformly.

Usage:
    python optimize/sanity_check_rq2.py \
        --weights weights/yolo11n.pt \
        --img-dir standalone/data/coco/images/val2017 \
        --csv-file neuron_eval_out/wisdom_yolo11n_scores_5000.csv \
        --out-dir results/rq2_sanity
"""
from __future__ import annotations
import argparse, os, sys, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimize.coverage_utils import load_layerwise_top_neurons


# ── Helpers ──────────────────────────────────────────────────────────

def load_single_image(img_dir: str, imgsz: int = 320, index: int | None = None):
    """Load one COCO image, return (tensor [1,3,H,W], PIL image, filename)."""
    paths = sorted(Path(img_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg files in {img_dir}")
    if index is None:
        index = random.randint(0, len(paths) - 1)
    p = paths[index % len(paths)]
    pil = Image.open(p).convert("RGB")
    tensor = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])(pil).unsqueeze(0)  # (1,3,H,W)
    return tensor, pil, p.name


def get_object_mask(model, image_tensor, device, imgsz, conf_thresh=0.25):
    """Binary mask (1,1,H,W): 1 inside detected objects, 0 background."""
    with torch.no_grad():
        out = model(image_tensor.to(device))
        preds = out[0] if isinstance(out, (tuple, list)) else out
    mask = torch.zeros(1, 1, imgsz, imgsz, device=device)
    cls_max = preds[0, 4:, :].max(dim=0).values
    confident = cls_max > conf_thresh
    if confident.sum() > 0:
        boxes = preds[0, :4, confident]
        for j in range(boxes.shape[1]):
            cx, cy, w, h = boxes[:, j]
            w, h = w * 1.3, h * 1.3
            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = min(imgsz, int(cx + w / 2))
            y2 = min(imgsz, int(cy + h / 2))
            mask[0, 0, y1:y2, x1:x2] = 1.0
    return mask


def perturb_random(image, mask, std=0.15):
    """Add random Gaussian noise where mask==1."""
    noise = torch.randn_like(image) * std
    return (image + noise * mask.cpu()).clamp(0, 1)


def importance_gradient(model, image_tensor, top_neurons, device):
    """
    Compute gradient of important-neuron activations w.r.t. input.

    Returns a gradient tensor same shape as image_tensor, indicating
    which input pixels most influence the important neurons.
    """
    x = image_tensor.clone().to(device).requires_grad_(True)

    # Attach hooks to capture activations of important neurons
    activations = {}
    handles = []
    for name, mod in model.named_modules():
        if name in top_neurons:
            indices = top_neurons[name]
            def _hook(layer_name, idxs):
                def fn(module, inp, out):
                    if out.dim() == 4:
                        activations[layer_name] = out[:, idxs, :, :].mean(dim=(2, 3)).sum()
                    elif out.dim() == 2:
                        activations[layer_name] = out[:, idxs].sum()
                return fn
            handles.append(mod.register_forward_hook(_hook(name, indices)))

    model.zero_grad()
    model(x)

    # Sum all important-neuron activations
    total = sum(activations.values())
    total.backward()

    for h in handles:
        h.remove()

    grad = x.grad.detach().cpu()  # (1,3,H,W)
    return grad


def perturb_importance(image, mask, grad, eps=0.15):
    """
    Perturb along gradient direction where mask==1.

    Normalises the gradient to unit norm per-pixel, then scales by eps.
    This focuses perturbation on the pixels the important neurons care about.
    """
    g = grad.clone()
    # Normalise gradient to unit L2 per pixel
    norm = g.norm(dim=1, keepdim=True).clamp(min=1e-8)
    g_normed = g / norm
    perturbation = g_normed * eps
    return (image + perturbation * mask.cpu()).clamp(0, 1)


def tensor_to_pil(tensor):
    """Convert (1,3,H,W) or (3,H,W) tensor to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def draw_mask_overlay(image_tensor, mask_tensor, color=(255, 0, 0), alpha=0.3):
    """Create a PIL image with semi-transparent mask overlay."""
    img = tensor_to_pil(image_tensor)
    arr = np.array(img).astype(np.float32)
    m = mask_tensor[0, 0].cpu().numpy()  # (H, W)
    overlay = np.zeros_like(arr)
    overlay[:, :] = color
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - alpha * m) + overlay[:, :, c] * (alpha * m)
    return Image.fromarray(arr.astype(np.uint8))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/yolo11n.pt")
    p.add_argument("--img-dir", default="standalone/data/coco/images/val2017")
    p.add_argument("--csv-file", default="neuron_eval_out/wisdom_yolo11n_scores_5000.csv")
    p.add_argument("--out-dir", default="results/rq2_sanity")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--per-layer-k", type=int, default=5)
    p.add_argument("--noise-std", type=float, default=0.15)
    p.add_argument("--image-index", type=int, default=None,
                   help="Index of image to use (random if not set)")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    from ultralytics import YOLO
    yolo = YOLO(a.weights)
    model = yolo.model.eval().to(a.device)

    top_neurons = load_layerwise_top_neurons(a.csv_file, per_layer_k=a.per_layer_k)
    total_n = sum(len(v) for v in top_neurons.values())
    print(f"Monitoring {total_n} neurons across {len(top_neurons)} layers")

    # Load image (try a few if first has no detections)
    for attempt in range(20):
        idx = a.image_index if a.image_index is not None else random.randint(0, 4999)
        img_tensor, pil_img, fname = load_single_image(a.img_dir, a.imgsz, idx)
        obj_mask = get_object_mask(model, img_tensor, a.device, a.imgsz)
        obj_frac = obj_mask.mean().item()
        if obj_frac > 0.05:
            break
        if a.image_index is not None:
            print(f"Warning: image {fname} has only {obj_frac:.1%} object coverage")
            break
        print(f"  Skipping {fname} (obj fraction {obj_frac:.1%}), trying another...")
    else:
        print("Could not find image with sufficient detections")
        return

    bg_mask = 1.0 - obj_mask
    print(f"Image: {fname} (object fraction: {obj_frac:.1%})")

    # Compute importance gradient
    print("Computing importance gradient...")
    grad = importance_gradient(model, img_tensor, top_neurons, a.device)

    # Generate 4 perturbed images
    rand_obj = perturb_random(img_tensor, obj_mask, std=a.noise_std)
    rand_bg = perturb_random(img_tensor, bg_mask, std=a.noise_std)
    imp_obj = perturb_importance(img_tensor, obj_mask, grad, eps=a.noise_std)
    imp_bg = perturb_importance(img_tensor, bg_mask, grad, eps=a.noise_std)

    # Save images
    images = {
        "0_original": img_tensor,
        "1_random_object": rand_obj,
        "2_random_background": rand_bg,
        "3_importance_object": imp_obj,
        "4_importance_background": imp_bg,
    }

    for label, t in images.items():
        out_path = os.path.join(a.out_dir, f"{label}.png")
        tensor_to_pil(t).save(out_path)
        print(f"  Saved {out_path}")

    # Save mask overlay for reference
    overlay = draw_mask_overlay(img_tensor, obj_mask)
    overlay_path = os.path.join(a.out_dir, "0_mask_overlay.png")
    overlay.save(overlay_path)
    print(f"  Saved {overlay_path}")

    # Save gradient heatmap
    grad_mag = grad[0].norm(dim=0).numpy()  # (H, W)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    grad_img = Image.fromarray((grad_mag * 255).astype(np.uint8))
    grad_path = os.path.join(a.out_dir, "0_gradient_heatmap.png")
    grad_img.save(grad_path)
    print(f"  Saved {grad_path}")

    # Print pixel-level difference stats
    print(f"\n{'Perturbation':<30} {'L2 diff':>10} {'Max diff':>10}")
    print("-" * 52)
    for label, t in images.items():
        if "original" in label:
            continue
        diff = (t - img_tensor).abs()
        l2 = diff.norm().item()
        mx = diff.max().item()
        print(f"  {label:<28} {l2:>10.4f} {mx:>10.4f}")

    print(f"\nAll outputs saved to {a.out_dir}/")


if __name__ == "__main__":
    main()

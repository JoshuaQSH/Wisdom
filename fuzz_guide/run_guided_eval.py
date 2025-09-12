#!/usr/bin/env python3
"""
run_guided_eval.py

Reproduce TABLE XII and add:
"Comparative Evaluation of Generated Inputs with StableDiffusion / BigGAN using Wisdom-guided"

For each generator root and each coverage method, this script:
  1) Builds a coverage baseline on real ImageNet (train) using your repo loaders.
  2) Greedily selects an "ACCEPT" subset from the provided generated "ORIGIN" images:
       accept img if criterion.gain(img) > 0, then criterion.update(...)
  3) Computes:
       - #Faults [Before/After]            (w.r.t. provided labels)
       - IS [Origin/Accept]                (torchmetrics InceptionScore, splits=10)
       - FID [Origin/Accept] vs real set   (torchmetrics FrechetInceptionDistance, feature=2048)
       - ΔFID = FID_accept - FID_origin
  4) Writes a CSV row per (generator, method).

Notes:
- Uses your IDC wrappers (Wisdom/DeepImportance) from fuzz_idc.py which wrap src.wisdom/src.deepidc.
- Uses src.nlc_coverage for NC/KMNC/... and Surprise coverage methods.
- Image transforms follow ImageNet norms; metrics are computed on uint8 (0..255) as in fuzzer_core.

Example:
python ./fuzz_guide/run_guided_eval.py \
  --imagenet-root /data/shenghao/dataset/ImageNet/ \
  --imagenet-index ./images/imagenet_class_index.json \
  --gen-root /data/shenghao/dataset/samples --generator BigGAN \
  --saved-model ./models_info/saved_models/resnet18_IMAGENET_patched_whole.pth \
  --device cuda:0 \
  --methods Wisdom \
  --wisdom-csv ./saved_files/pre_csv/resnet18_imagenet.csv \
  --out-csv guided_eval_results.csv
  
python ./fuzz_guide/run_guided_eval.py \
  --dataset CIFAR10 \
  --cifar-root ./datasets/CIFAR10 \
  --imagenet-root /data/shenghao/dataset/ImageNet/ \
  --imagenet-index ./images/imagenet_class_index.json \
  --gen-root ./datasets/Inpainting --generator StableDiffusion \
  --saved-model ./models_info/saved_models/vgg16_CIFAR10_whole.pth \
  --methods Wisdom \
  --wisdom-csv ./saved_files/pre_csv/vgg16_cifar10.csv \
  --device cuda:0 \
  --out-csv ./guided_eval_results_cifar10.csv
"""

import os
import json
import csv
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader

# ---- your repo imports
from src.utils import get_model, load_ImageNet, make_path, get_trainable_modules_main  # type: ignore
import src.nlc_tool as nlc_tool  # for layer sizes (when needed)
import src.nlc_coverage as coverage  # NC/KMNC/... interface
from fuzz_idc import DeepImportance, Wisdom  # IDC wrappers built on src.deepidc/src.wisdom

# torchmetrics for IS/FID
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


# -------------------------- helpers --------------------------
# ---- CIFAR10 constants ----
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
C10_NAME2IDX = {n: i for i, n in enumerate(CIFAR10_CLASSES)}

def cifar10_default_transform() -> T.Compose:
    return T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),  # [0,1]
    ])
    
class CIFARFolder(Dataset):
    """
    Real/reference CIFAR10 folder loader:
      <cifar_root>/train/<class>/*.png (your preprocessed layout)
    """
    def __init__(self, root: str):
        train_dir = os.path.join(root, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Expected CIFAR10 'train' folder at {train_dir}")
        self.samples: List[Tuple[str,int]] = []
        self.tx = cifar10_default_transform()
        for cls in CIFAR10_CLASSES:
            d = os.path.join(train_dir, cls)
            if not os.path.isdir(d): 
                print(f"[WARN] CIFAR10: missing class dir: {d}")
                continue
            for fn in os.listdir(d):
                if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp")):
                    self.samples.append((os.path.join(d, fn), C10_NAME2IDX[cls]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        x = default_loader(p)
        return self.tx(x), y

class GeneratedFolderCIFAR10(Dataset):
    """
    Generated CIFAR10 set:
      <gen_root>/<class>/*.(png|jpg|jpeg|bmp|webp)
    """
    def __init__(self, root: str):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Generated CIFAR10 root not found: {root}")
        self.samples: List[Tuple[str,int]] = []
        self.tx = cifar10_default_transform()
        for cls in sorted(os.listdir(root)):
            p = os.path.join(root, cls)
            if not os.path.isdir(p):
                continue
            key = cls.strip().lower()
            if key not in C10_NAME2IDX:
                print(f"[WARN] Skip class '{cls}' (not a CIFAR10 class).")
                continue
            y = C10_NAME2IDX[key]
            for fn in os.listdir(p):
                if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp")):
                    self.samples.append((os.path.join(p, fn), y))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        x = default_loader(p)
        return self.tx(x), y

def imagenet_default_transform(image_size: int = 128) -> T.Compose:
    # Same spatial size assumption as your ImageNet runs (see run_fuzz.py assertions)
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),  # [0,1] float
    ])

def _normalize_for_model(x: torch.Tensor, dataset: str) -> torch.Tensor:
    if dataset.lower() == "cifar10":
        norm = T.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2471, 0.2435, 0.2616))
    else:
        norm = T.Normalize((0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225))
    return norm(x)

def _to_uint8_4d(x: torch.Tensor) -> torch.Tensor:
    # x: BCHW in [0,1] or [0,255]
    if x.max() <= 1.0:
        x = x * 255.0
    return x.round().clamp(0, 255).to(torch.uint8)

def compute_is_fid_uint8(real_bchw_u8: torch.Tensor, fake_bchw_u8: torch.Tensor, device: torch.device) -> Dict[str, float]:
    """
    real_bchw_u8, fake_bchw_u8: BCHW, uint8 [0..255]
    IS: on fake only (splits=10); FID: real vs fake (feature=2048)
    """
    inception = InceptionScore(splits=10).to(device)
    fid = FrechetInceptionDistance(feature=2048).to(device)

    with torch.no_grad():
        # IS (fake)
        for i in range(0, fake_bchw_u8.size(0), 64):
            inception.update(fake_bchw_u8[i:i+64].to(device))
        IS_mean = float(inception.compute()[0].item())

        # FID (real + fake)
        for i in range(0, real_bchw_u8.size(0), 64):
            fid.update(real_bchw_u8[i:i+64].to(device), real=True)
        for i in range(0, fake_bchw_u8.size(0), 64):
            fid.update(fake_bchw_u8[i:i+64].to(device), real=False)
        FID_val = float(fid.compute().item())

    return {"IS": IS_mean, "FID": FID_val}

def load_imagenet_index(idx_path: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Returns:
      name2idx: lowercased human label -> idx
      idx2wnid: str(idx) -> wnid
    """
    with open(idx_path, "r") as f:
        idx = json.load(f)
    name2idx = {}
    idx2wnid = {}
    for k, (wnid, human) in idx.items():
        name2idx[human.lower()] = int(k)
        idx2wnid[k] = wnid
    return name2idx, idx2wnid

def normalize_label_name(s: str) -> str:
    # folder names are now snake_case; map to space form then lowercase
    return s.replace("_", " ").strip().lower()

class GeneratedFolder(Dataset):
    """
    Reads generated images in structure:
      root/
        <class_name>/
          *.png|*.jpg|*.jpeg

    Uses imagenet_class_index.json to map <class_name> -> class index (best-effort).
    If a name isn't in the map (due to synonym differences), you can provide a CSV mapping via --labels-csv.
    """
    def __init__(self, root: str, name2idx: Dict[str, int], override_map: Optional[Dict[str,int]],
                 image_size: int = 128):
        self.root = root
        self.samples: List[Tuple[str,int]] = []
        tx = imagenet_default_transform(image_size)
        self.tx = tx

        # scan folders
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Generated root not found: {root}")
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if not os.path.isdir(p): 
                continue
            key = normalize_label_name(d)
            if override_map and key in override_map:
                cls_idx = override_map[key]
            else:
                cls_idx = name2idx.get(key, None)
            if cls_idx is None:
                # skip unknown class names; report once
                print(f"[WARN] Skip class '{d}' (no mapping in imagenet_class_index nor overrides).")
                continue
            # collect images
            for fn in os.listdir(p):
                if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp")):
                    self.samples.append((os.path.join(p, fn), cls_idx))

        if len(self.samples) == 0:
            print(f"[WARN] No images found under {root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = default_loader(path)  # PIL
        x = self.tx(img)  # [0,1] float
        return x, y


@torch.no_grad()
def count_faults(model: torch.nn.Module, loader: DataLoader, dataset: str, device: torch.device) -> int:
    model.eval().to(device)
    faults = 0
    for x, y in loader:
        x = x.to(device)
        y = torch.as_tensor(y, device=device, dtype=torch.long)
        x = _normalize_for_model(x, dataset)
        logits = model(x)
        pred = logits.argmax(1)
        faults += int((pred != y).sum().item())
    return faults

def gather_images(loader: DataLoader, limit: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: BCHW in [0,1]
      Y: labels (long)
    """
    xs, ys = [], []
    n = 0
    for x, y in loader:
        xs.append(x)
        ys.append(torch.as_tensor(y, dtype=torch.long))
        n += x.size(0)
        if limit is not None and n >= limit:
            break
    if len(xs) == 0:
        return torch.empty(0,3,1,1), torch.empty(0, dtype=torch.long)
    X = torch.cat(xs, 0)
    Y = torch.cat(ys, 0)
    if limit is not None and X.size(0) > limit:
        X = X[:limit]
        Y = Y[:limit]
    return X, Y

def build_accept_set_greedy(
    method_name: str,
    criterion,
    origin_loader: DataLoader,
    dataset: str,
    surprise_needs_label: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterate origin images in mini-batches and accept if gain > 0.
    Returns accepted (BCHW [0,1]) and labels (long).
    """
    accepted_x, accepted_y = [], []
    for x, y in origin_loader:
        x = x.to(device)
        y = torch.as_tensor(y, device=device)
        # coverage.calculate expects:
        #  - IDC wrappers: Tensor or DataLoader; labels not needed
        #  - Surprise coverage (LSC/DSC/MDSC): (x, y) tuple
        if surprise_needs_label:
            # data = (x, y)
            cove = criterion.calculate(x, y)
        else:
            # data = x
            cove = criterion.calculate(x)
        # cove = criterion.calculate(data)
        gain = criterion.gain(cove)
        if gain is None:
            # If a method returns None when no positive gain, treat as 0
            gain = 0.0
        if isinstance(gain, tuple):
            # NLC returns (delta, layers_to_update) or None
            delta = gain[0]
            accept = (delta > 0)
        else:
            accept = (gain > 0)

        if accept:
            criterion.update(cove, gain)
            accepted_x.append(x.detach().cpu())
            accepted_y.append(y.detach().cpu())
        # else: reject

    if len(accepted_x) == 0:
        return torch.empty(0,3,1,1), torch.empty(0, dtype=torch.long)
    return torch.cat(accepted_x, 0), torch.cat(accepted_y, 0)


# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser()
    # data / model
    ap.add_argument("--imagenet-root", required=True, help="Path to ImageNet root (expects train/val like your utils).")
    ap.add_argument("--imagenet-index", required=True, help="imagenet_class_index.json")
    ap.add_argument("--saved-model", required=True, help="Path to your saved model (as used by src.utils.get_model).")
    ap.add_argument("--dataset", choices=["ImageNet", "CIFAR10"], required=True,
                help="Which dataset the run targets; controls transforms, class count, hyperparams.")
    ap.add_argument("--cifar-root", default="", help="Path to CIFAR10 root that contains train/ with class folders")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--image-size", type=int, default=128)
    # generated sets
    ap.add_argument("--gen-root", action="append", default=[], help="Root of a generated dataset (class-subdirs). Use multiple.")
    ap.add_argument("--generator", action="append", default=[], help="Name tag for the above gen-root (same order).")
    # coverage methods
    ap.add_argument("--methods", nargs="+", default=["Wisdom", "DeepImportance", "KMNC", "NC"])
    ap.add_argument("--wisdom-csv", default="", help="Wisdom top-neuron CSV (required if Wisdom in methods).")
    ap.add_argument("--top-m", type=int, default=10)
    ap.add_argument("--n-clusters", type=int, default=2)
    # optional label override
    ap.add_argument("--labels-csv", default="", help="Optional CSV: class_name,idx to override imagenet_class_index mapping.")
    # output
    ap.add_argument("--out-csv", default="./guided_eval_results.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert len(args.gen_root) == len(args.generator), "Use --gen-root and --generator with the same count."

    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # 1) Model
    model, _, _ = get_model(args.saved_model)  # your utility
    model.eval().to(device)
    _, trainable_names = get_trainable_modules_main(model)
    final_layer_name = trainable_names[-1] if len(trainable_names) else None

   
    batch_size = 4

    # Load ImageNet as in your repo so transforms are consistent
    # train_loader, val_loader, train_ds, val_ds, classes = load_ImageNet(
    #     batch_size=batch_size, root=args.imagenet_root, num_workers=4, use_val=False,
    #     label_path=args.imagenet_index
    # )
    
    # 2) Real ImageNet loaders
    if args.dataset == "ImageNet":
        assert args.imagenet_root, "--imagenet-root is required for ImageNet"
        # Ensure ImageNet assertions consistent with your repo
        # (image_size is controlled by --image-size, default set earlier to 128)
        # make_path(os.path.dirname(args.out_csv))
        train_loader, val_loader, train_ds, val_ds, classes = load_ImageNet(
            batch_size=batch_size, root=args.imagenet_root, num_workers=4, use_val=False,
            label_path=args.imagenet_index
        )
        # generator mapping by imagenet_class_index.json
        name2idx, _ = load_imagenet_index(args.imagenet_index)
        override_map = None
        if args.labels_csv and os.path.exists(args.labels_csv):
            override_map = {}
            with open(args.labels_csv, "r") as f:
                r = csv.reader(f)
                for row in r:
                    if not row: continue
                    cname, idx = row[0].strip(), int(row[1])
                    override_map[normalize_label_name(cname)] = idx

        # per-dataset knobs
        num_classes = 1000
        image_size = args.image_size  # 128 in your pipeline
        cc_hyper = 1000
        
     # 2) Real CIFAR10 loaders
    else:
        assert args.cifar_root, "--cifar-root is required for CIFAR10"
        # Real/reference loader from folderized CIFAR10
        cifar_train = CIFARFolder(args.cifar_root)
        train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False, num_workers=2)
        classes = CIFAR10_CLASSES[:]  # 10 classes
        # generator mapping is trivial (folder name -> fixed index)
        name2idx = C10_NAME2IDX
        override_map = None

        num_classes = 10
        image_size = 32
        cc_hyper = 10  # matches your run_fuzz.py

    # 3) Name mapping for generated folders
    name2idx, _ = load_imagenet_index(args.imagenet_index)
    override_map = None
    if args.labels_csv and os.path.exists(args.labels_csv):
        override_map = {}
        with open(args.labels_csv, "r") as f:
            r = csv.reader(f)
            for row in r:
                if not row: continue
                cname, idx = row[0].strip(), int(row[1])
                override_map[normalize_label_name(cname)] = idx

    # 4) coverage hyper map (aligned to run_fuzz.py defaults)
    hyper_map = {
        'NLC': None,
        'NC': 0.5,
        'KMNC': 100,
        'SNAC': None,
        'NBC': None,
        'TKNC': 10,
        'TKNP': 50,
        'CC': cc_hyper,
        'LSC': 10,
        'DSC': 0.1,
        'MDSC': 10,
        'DeepImportance': [args.top_m, args.n_clusters],
        'Wisdom': [args.top_m, args.n_clusters],
    }

    # CSV header
    header = ["Generator", "Method",
              "#Faults Before", "#Faults After",
              "IS Origin", "IS Accept",
              "FID Origin", "FID Accept", "ΔFID"]
    write_header = not os.path.exists(args.out_csv)
    out_f = open(args.out_csv, "a", newline="")
    writer = csv.writer(out_f)
    if write_header:
        writer.writerow(header)

    # Iterate each generator root
    for gen_root, gen_name in zip(args.gen_root, args.generator):
        # Build origin dataset / loader for generated imgs
        if args.dataset == "ImageNet":
            origin_ds = GeneratedFolder(gen_root, name2idx, override_map, image_size=image_size)
        else:
            origin_ds = GeneratedFolderCIFAR10(gen_root)
        # origin_ds = GeneratedFolder(gen_root, name2idx, override_map, image_size=args.image_size)
        if len(origin_ds) == 0:
            print(f"[WARN] No origin images in {gen_root}; skipping.")
            continue
        origin_loader = DataLoader(origin_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        # sample a reference real set ≈ same count for FID
        N = len(origin_ds)
        real_imgs, _ = gather_images(train_loader, limit=N)
        if real_imgs.dim() == 0 or real_imgs.size(0) < 2:
            print("[WARN] Too few real images collected for FID; FID will be NaN.")
        real_uint8 = _to_uint8_4d(real_imgs)

        # Precompute origin tensors/labels (for stats and faults)
        origin_imgs, origin_labels = gather_images(origin_loader)
        origin_uint8 = _to_uint8_4d(origin_imgs)

        # origin faults
        faults_before = count_faults(model, origin_loader, dataset="ImageNet", device=device)

        # IS/FID for origin
        metrics_origin = compute_is_fid_uint8(real_uint8, origin_uint8, device)
        IS_origin = metrics_origin["IS"]
        FID_origin = metrics_origin["FID"]
        
        # Run each method
        # We will (re)build coverage baseline per method, to keep results independent
        # Surprise coverage methods need (x,y) labels
        for method in args.methods:
            print(f"\n[Eval] {gen_name} with {method}")

            # Fresh baseline: layer sizes (needed by nlc_coverage)
            # (run_fuzz.py gets sizes by a random forward; we do the same)
            try:
                probe_x, _probe_y = next(iter(train_loader))
            except StopIteration:
                raise RuntimeError("train_loader is empty; cannot probe layer sizes.")
            probe_x = probe_x.to(device)
            layer_size_dict = nlc_tool.get_layer_output_sizes(model, probe_x)

            # Instantiate criterion
            if method in ("Wisdom", "DeepImportance"):
                if method == "Wisdom":
                    if not args.wisdom_csv:
                        raise ValueError("Wisdom selected but --wisdom-csv not provided.")
                    criterion = Wisdom(
                        model, hyper_map["Wisdom"][0], hyper_map["Wisdom"][1],
                        "KMeans", train_loader, args.wisdom_csv, device=device
                    )
                else:
                    # DeepImportance uses LRP to pick neurons on the train loader
                    criterion = DeepImportance(
                        model, hyper_map["DeepImportance"][0], hyper_map["DeepImportance"][1],
                        "KMeans", train_loader, final_layer_name, device=device
                    )
                # IDC wrappers don't require assess(); clusters are fitted in their ctors.

                surprise = False  # IDC doesn't need labels
            else:
                # SOTA neuron coverage methods from src.nlc_coverage
                ctor = getattr(coverage, method)
                # Surprise coverage needs (x,y)
                surprise = method in ("LSC", "DSC", "MDSC")
                if surprise:
                    # For surprise coverage, num_class is for KDE/SA; ImageNet=1000
                    criterion = ctor(model, device, layer_size_dict, hyper=hyper_map[method],
                                     min_var=1e-5, num_class=1000)
                else:
                    criterion = ctor(model, device, layer_size_dict, hyper=hyper_map[method])

                # Build baseline on train set
                criterion.build(train_loader)
                # For most coverage methods, also assess(train) to init "current"
                if method not in ('CC', 'TKNP', 'LSC', 'DSC', 'MDSC'):
                    criterion.assess(train_loader)

            # Greedy accept set
            accept_imgs, accept_labels = build_accept_set_greedy(
                method, criterion, origin_loader,
                dataset="ImageNet",
                surprise_needs_label=surprise,
                device=device
            )
            # If nothing accepted, put tiny placeholders to avoid crashes
            if accept_imgs.size(0) < 1:
                IS_accept, FID_accept, delta_fid = math.nan, math.nan, math.nan
                faults_after = len(origin_ds)  # pessimistic (or set equal to before)
            else:
                faults_after = count_faults(
                    model,
                    DataLoader(list(zip(accept_imgs, accept_labels)), batch_size=batch_size),
                    dataset="ImageNet", device=device
                )
                accept_uint8 = _to_uint8_4d(accept_imgs)
                metrics_accept = compute_is_fid_uint8(real_uint8, accept_uint8, device)
                IS_accept = metrics_accept["IS"]
                FID_accept = metrics_accept["FID"]
                delta_fid = FID_accept - FID_origin if (not math.isnan(FID_accept) and not math.isnan(FID_origin)) else math.nan

            # write row
            row = [
                gen_name, method,
                int(faults_before), int(faults_after),
                f"{IS_origin:.4f}" if not math.isnan(IS_origin) else "",
                f"{IS_accept:.4f}" if not (isinstance(IS_accept,float) and math.isnan(IS_accept)) else "",
                f"{FID_origin:.4f}" if not math.isnan(FID_origin) else "",
                f"{FID_accept:.4f}" if not (isinstance(FID_accept,float) and math.isnan(FID_accept)) else "",
                f"{delta_fid:.4f}" if not (isinstance(delta_fid,float) and math.isnan(delta_fid)) else "",
            ]
            writer.writerow(row)
            print(" -> Row:", row)

    out_f.close()
    print(f"\nSaved results to: {args.out_csv}")


if __name__ == "__main__":
    main()

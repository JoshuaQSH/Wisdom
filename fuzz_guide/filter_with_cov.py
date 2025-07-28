import argparse
import os
import csv
import datetime
import random
import torchvision
import torch
import numpy as np
import math
import glob
import gc
import copy

from torchvision import transforms as T
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from fuzz_dataloader import CustomFolderFuzzDataset

import src.nlc_coverage as coverage
import src.nlc_tool as tool
from fuzz_idc import DeepImportance, Wisdom 

from src.utils import make_path, get_model, get_trainable_modules_main, load_ImageNet

import src.nlc_coverage as coverage
import src.nlc_tool as tool
from fuzz_idc import DeepImportance, Wisdom 
import fuzz_dataloader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _to_uint8(x):
    return (x.mul(255).clamp(0,255).to(torch.uint8))
    
def compute_metrics(fake_imgs, real_imgs, model, device="cpu"):
    """
    fake_imgs : list[Tensor(C,H,W)] in [0,1]
    real_imgs : list[Tensor(C,H,W)] in [0,1]  (1 000 test seeds – FID anchor)
    model     : torch.nn.Module (eval mode)
    returns   : dict with keys IS, FID, ENT, NCLASS
    """
    fake = torch.stack(fake_imgs)
    real = torch.stack(real_imgs)
    iscore = InceptionScore(feature=2048, store_features=False).to(device)
    for i in range(0, len(fake), 64):
        iscore.update(_to_uint8(fake[i:i+64]).to(device))
    IS = iscore.compute()[0].item()
    if len(fake) < 2 or len(real) < 2:
        FID = math.nan
    else:
        fid = FrechetInceptionDistance(feature=2048).to(device)
        for i in range(0, len(real), 64):
            fid.update(_to_uint8(real[i:i+64]).to(device), real=True)
        for i in range(0, len(fake), 64):
            fid.update(_to_uint8(fake[i:i+64]).to(device), real=False)
        FID = fid.compute().item()
    norm = (T.Normalize((0.4914,0.4822,0.4465),
                        (0.2471,0.2435,0.2616))
            if fake.shape[-1]==32 else
            T.Normalize((0.485,0.456,0.406),
                        (0.229,0.224,0.225)))
    hist = torch.zeros(model.fc.out_features, device=device)
    with torch.no_grad():
        for i in range(0, len(fake), 64):
            y = model(norm(fake[i:i+64].to(device))).argmax(1)
            hist += torch.bincount(y, minlength=len(hist))
    prob = hist.cpu() / hist.sum()
    ENT = float(-(prob*torch.log(prob+1e-12)).sum())
    NCLASS = int((hist>0).sum())
    return {"IS":IS, "FID":FID, "ENT":ENT, "NCLASS":NCLASS}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["CIFAR10","ImageNet"])
    parser.add_argument('--data-path', type=str, default='./datasets/', help='Path to the data directory.')
    parser.add_argument("--model", required=True, default="resnet18", help="Model name, e.g. resnet18")
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training.')
    parser.add_argument("--criterion",required=True,
                choices=["NC","KMNC","NBNC","SNAC","TKNC",
                            "NLC","LSC","DSC","MDSC",
                            "DeepImportance","Wisdom","Random"])
    parser.add_argument("--fake-root",required=True,
                help="Folder with generated images to be filtered")
    parser.add_argument("--label-csv",default=None,
                help="Optional csv filename,label for fault counting")
    parser.add_argument("--wisdom-csv",default="./wisdom_topk.csv")
    parser.add_argument("--output-csv",default="./filter_results.csv")
    args = args.parse_args()
    return args

def prepare_data_model(args):
    model_path = os.getenv("HOME") + args.saved_model
    model, module_name, module = get_model(model_path)
    
    if args.dataset == 'ImageNet':
        assert args.image_size == 128
        assert args.num_class <= 1000
    elif args.dataset == 'CIFAR10':
        assert args.image_size == 32
        assert args.num_class <= 10
    
    model.to(args.device)
    model.eval()
    input_size = (1, args.nc, args.image_size, args.image_size)
    random_data = torch.randn(input_size).to(args.device)
    layer_size_dict = tool.get_layer_output_sizes(model, random_data)
    
    if args.dataset == 'CIFAR10':
        # data_set = fuzz_dataloader.CIFAR10FuzzDataset(args, split='test')
        data_set  = fuzz_dataloader.TorchvisionCIFAR10FuzzDataset(args, root=args.data_path, split="test")
        TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = fuzz_dataloader.get_loader(args)
    elif args.dataset == 'ImageNet':
        # data_set = fuzz_dataloader.ImageNetFuzzDataset(args, image_dir=args.data_path, label2index_file='./datasets/imagenet_labels.json', split='val')
        data_set = fuzz_dataloader.TorchImageNetFuzzDataset(args, root=args.data_path, split="val")
        train_loader, test_loader, train_dataset, val_dataset, classes = load_ImageNet(batch_size=args.batch_size, root=args.data_path, num_workers=2, use_val=False, label_path='./datasets/imagenet_labels.json')
        TOTAL_CLASS_NUM = len(classes)
    
    # TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = fuzz_dataloader.get_loader(args)
    image_list, label_list = data_set.build()
    image_numpy_list = data_set.to_numpy(image_list)
    label_numpy_list = data_set.to_numpy(label_list, False)
    
    del image_list
    del label_list
    gc.collect()
    
    return model, layer_size_dict, TOTAL_CLASS_NUM, train_loader, test_loader, test_loader, image_numpy_list, label_numpy_list


def main():
    args = parse_args()
    set_seed(42)
    hyper_map = {
        'NLC': None,
        'NC': 0.75,
        'KMNC': 100,
        'SNAC': None,
        'NBC': None,
        'TKNC': 10,
        'TKNP': 50,
        'CC': 10 if args.dataset == 'CIFAR10' else 1000,
        'LSA': 10,
        'DSA': 0.1,
        'MDSA': 10,
        'DeepImportance': [10, 2], # top_m_neurons, n_clusters
        'Wisdom': [10, 2] # top_m_neurons, n_clusters
    }
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    
    
    # Prepare data and model
    model, layer_size_dict, num_class, train_loader, test_loader, seed_loader, image_numpy_list, label_numpy_list = prepare_data_model(args)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)
    final_layer = trainable_module_name[-1]
    
    # Coverage method
    if args.criterion in ['LSC', 'DSC', 'MDSC']:
        criterion = getattr(coverage, args.criterion)(model, device, layer_size_dict, hyper=hyper_map[args.criterion], min_var=1e-5, num_class=num_class)
    else:
        if args.criterion == 'Random':
            # Random will inherit the NC criterion
            criterion = getattr(coverage, 'NC')(model, device, layer_size_dict, hyper=hyper_map['NC'])
        elif args.criterion == 'DeepImportance':
            criterion = DeepImportance(model, hyper_map[args.criterion][0], hyper_map[args.criterion][1], "KMeans", train_loader, final_layer, device)
        elif args.criterion == 'Wisdom':
            criterion = Wisdom(model, hyper_map[args.criterion][0], hyper_map[args.criterion][1], "KMeans", train_loader, args.wisdom_csv)
            breakpoint()
        else:
            criterion = getattr(coverage, args.criterion)(model, device, layer_size_dict, hyper=hyper_map[args.criterion])
    
    
    if args.criterion not in ['DeepImportance', 'Wisdom']:
        criterion.build(train_loader)
    if args.criterion not in ['CC', 'TKNP', 'LSC', 'DSC', 'MDSC', 'DeepImportance', 'Wisdom']:
        criterion.assess(train_loader)
    
    # ---------- 0.  loaders & model --------------------------------------
    if args.dataset == 'CIFAR10':
        # data_set = fuzz_dataloader.CIFAR10FuzzDataset(args, split='test')
        data_set  = fuzz_dataloader.TorchvisionCIFAR10FuzzDataset(args, root=args.data_path, split="test")
        TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = fuzz_dataloader.get_loader(args)
    elif args.dataset == 'ImageNet':
        # data_set = fuzz_dataloader.ImageNetFuzzDataset(args, image_dir=args.data_path, label2index_file='./datasets/imagenet_labels.json', split='val')
        data_set = fuzz_dataloader.TorchImageNetFuzzDataset(args, root=args.data_path, split="val")
        train_loader, test_loader, train_dataset, val_dataset, classes = load_ImageNet(batch_size=args.batch_size, root=args.data_path, num_workers=2, use_val=False, label_path='./datasets/imagenet_labels.json')
        TOTAL_CLASS_NUM = len(classes)
    
    # TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = fuzz_dataloader.get_loader(args)
    image_list, label_list = data_set.build()
    real_imgs_numpy_list = data_set.to_numpy(image_list)
    real_labels_numpy_list = data_set.to_numpy(label_list, False)
    

    # fake set to evaluate / filter
    dargs = argparse.Namespace(custom_root=args.fake_root,
                            label_csv=args.label_csv,
                            image_size=32 if args.dataset=="CIFAR10" else 224)
    fake_ds = CustomFolderFuzzDataset(dargs)
    fake_imgs_all, fake_lbls_all = fake_ds.build()
    
    before_is, before_fid, before_num_class = compute_metrics(engine.params, real_imgs_numpy_list, model)



    
    # ---------- 3. filter loop -------------------------------------------
    accepted_imgs, accepted_lbls = [], []
    for img, lbl in zip(fake_imgs_all, fake_lbls_all):
        with torch.no_grad():
            gain = criterion.gain(criterion.calculate(img.unsqueeze(0).to(device)))
        if gain > 0:
            criterion.update({"ratio":criterion.current+gain}, gain)
            accepted_imgs.append(img)
            accepted_lbls.append(lbl)

    # ---------- 4. metrics AFTER filtering -------------------------------
    if accepted_imgs:
        m_after = compute_metrics(accepted_imgs, real_imgs, model, device)
        faults_after = sum(int(
            model((accepted_imgs[i].to(device)-0.5*0.5)/0.5).argmax().item()!=accepted_lbls[i]
        ) for i in range(len(accepted_lbls))) if args.label_csv else "NA"
    else:
        m_after = dict(IS=float("nan"),FID=float("nan"),
                    ENT=float("nan"),NCLASS=0)
        faults_after = "NA"

    # ---------- 5. write / print table -----------------------------------
    row = [datetime.datetime.now().isoformat(timespec="seconds"),
        args.criterion,
        f"{faults_before}->{faults_after}",
        f"{m_before['IS']:.3f}->{m_after['IS']:.3f}",
        f"{m_before['FID']:.2f}->{m_after['FID']:.2f}",
        m_after["NCLASS"],
        f"{m_before['ENT']:.3f}->{m_after['ENT']:.3f}"]
    header = ["timestamp","criterion",
            "Faults","IS","FID","#Class","Entropy"]

    write_head = not os.path.exists(args.output_csv)
    with open(args.output_csv,"a",newline="") as f:
        w=csv.writer(f); 
        if write_head: w.writerow(header)
        w.writerow(row)

    print("\n".join(f"{h}: {v}" for h,v in zip(header,row)))
    
    # Filter images based on coverage
    accepted_imgs, accepted_lbls = [], []
    for img, lbl in zip(image_numpy_list, label_numpy_list):
        with torch.no_grad():
            gain = cov.gain(cov.calculate(img.unsqueeze(0).to(args.device)))
        if gain > 0:
            cov.update({"ratio":cov.current+gain}, gain)
            accepted_imgs.append(img)
            accepted_lbls.append(lbl)

    # Compute metrics after filtering
    if accepted_imgs:
        m_after = compute_metrics(accepted_imgs, real_imgs, model, args.device)
        faults_after = sum(int(
            model((accepted_imgs[i].to(args.device)-0.5*0.5)/0.5).argmax().item()!=accepted_lbls[i]
        ) for i in range(len(accepted_lbls))) if args.label_csv else "NA"
    else:
        m_after = dict(IS=float("nan"),FID=float("nan"),
                       ENT=float("nan"),NCLASS=0)
        faults_after = "NA"

    # Write results to CSV
    row = [datetime.datetime.now().isoformat(timespec="seconds"),
           args.criterion,
           f"{faults_before}->{faults_after}",
           f"{m_before['IS']:.3f}->{m_after['IS']:.3f}",
           f"{m_before['FID']:.2f}->{m_after['FID']:.2f}",
           m_after["NCLASS"],
           f"{m_before['ENT']:.3f}->{m_after['ENT']:.3f}"]
    
    with open(args.output_csv,"a",newline="") as f:
        w=csv.writer(f); 
        if write_head: w.writerow(header)
        w.writerow(row)
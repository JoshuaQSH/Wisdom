import copy
import gc
import os
import argparse
import random
import csv

import torch
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

from PIL import Image
import torchvision.transforms as T
import numpy as np

from src.utils import make_path, get_model, get_trainable_modules_main, load_ImageNet
import src.nlc_coverage as coverage
import src.nlc_tool as tool

from fuzzer_core import Fuzzer, Parameters
import fuzz_dataloader

from fuzz_idc import DeepImportance, Wisdom 

"""
Usage:
- Random baseline fuzzing for CIFAR10 dataset using VGG16 model.
$ python ./fuzz_guide/run_fuzz.py --dataset CIFAR10 --model vgg16 --saved-model /torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth --criterion NC --output-dir ./fuzz_guide/fuzz_outputs/ --log-dir ./logs --seed 42 --device 'cuda:0'
- Coverage method guided fuzzing is not used in this script, but can be enabled by setting the --guided flag.
$ python ./fuzz_guide/run_fuzz.py --dataset CIFAR10 --model vgg16 --saved-model /torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth --criterion NC --output-dir ./fuzz_guide/fuzz_outputs/ --guided --log-dir ./logs --seed 42 --device 'cuda:0'

Hyperparameters:
- alpha: pixel-budget ratio, the smaller the easier to be accepted as valid output, default is 0.2
- beta: pixel-budget magnitude: the maximum per-pixel absolute change allowed when the ratio test fails, default is 0.4
"""

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Fuzz - Random baseline driver")
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                            choices=['CIFAR10', 'ImageNet'])
    parser.add_argument('--data-path', type=str, default='./datasets/', help='Path to the data directory.')
    parser.add_argument('--model', type=str, default='vgg16', help='Model name.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training.')
    parser.add_argument('--saved-model', type=str, default='/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth', 
                        help='Saved model name.')
    
    # coverage method (ignored in random mode)
    parser.add_argument('--criterion', type=str, default='NC', 
                            choices=['Random', 'NLC', 'NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'CC',
                    'LSC', 'DSC', 'MDSC', 'DeepImportance', 'Wisdom'])
    parser.add_argument('--wisdom-csv', type=str, default='./saved_files/pre_csv/vgg16_cifar10.csv',
                        help='CSV file for Wisdom heuristic (if applicable).')
    
    # mode
    parser.add_argument('--guided', action='store_true', help='Use guided fuzzing.')
    parser.add_argument('--genai-only', action='store_true', help='Generative AI for generation [StableDiffusion or BigGAN].')

    # I/O paths
    parser.add_argument('--output-dir', type=str, default='./fuzz_guide/fuzz_outputs/')
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    base_args = parser.parse_args()
    args = Parameters(base_args)
    return  args

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

def to_uint8(batch_float):
    return (batch_float.mul(255).clamp(0, 255).to(torch.uint8))

def compute_metrics(params, seed_imgs, model):
    device = next(model.parameters()).device
    
    # 1) Collect ALL newly accepted mutations (*.jpg dumped by Fuzzer)
    import glob

    file_glob = os.path.join(params.image_dir, "*_new.jpg")
    paths = sorted(glob.glob(file_glob))
    if not paths:
        print("No mutated images were dumped – cannot compute IS/FID.")
        return
    to_tensor = T.Compose([T.ToTensor()])
    mutated = torch.stack([to_tensor(Image.open(p)) for p in paths])

    # 2) --- Inception Score --------------------------------------------
    iscore = InceptionScore(feature=2048).to(device)
    for i in range(0, len(mutated), 64):
        iscore.update(to_uint8(mutated[i:i+64]).to(device))
    IS = iscore.compute()[0].item() 

    # 3) --- FID  (against the 1 000 seeds) ------------------------------
    fid = FrechetInceptionDistance(feature=2048).to(device)
    
    # real = seeds (already in tensor form, [0,1])
    seeds_np = np.array(seed_imgs)
    seed_stack = torch.tensor(seeds_np, dtype=torch.uint8).permute(0, 3, 1, 2)  # (1000,3,H,W)
    # seed_stack = torch.stack(seed_imgs)            # (1000,3,H,W)
    for i in range(0, len(seed_stack), 64):
        fid.update(seed_stack[i:i+64].to(device), real=True)
    # fake = mutations
    for i in range(0, len(mutated), 64):
        fid.update(to_uint8(mutated[i:i+64]).to(device), real=False)
    FID = fid.compute().item()

    # 4) --- Entropy (model’s top‑1 predictions) -------------------------
    ent_loader = torch.utils.data.DataLoader(mutated, batch_size=64, shuffle=False)
    hist = torch.zeros(params.num_class, dtype=torch.float64)
    with torch.no_grad():
        for x in ent_loader:
            y = model(x.to(device)).argmax(1)
            hist += torch.bincount(y.cpu(), minlength=params.num_class)
    prob = hist / hist.sum()
    ENT = -(prob * torch.log(prob + 1e-12)).sum().item()

    # 5) --- #Classes -------------------------------------------------
    num_classes_hit = int((hist > 0).sum().item())

    # 6) --- Print summary ----------------------------------------------
    print(f"\n=== Quality metrics ===")
    print(f"Inception Score (IS): {IS:6.3f}")
    print(f"Fréchet Inception Distance: {FID:6.3f}")
    print(f"Prediction-Entropy (nats): {ENT:6.3f}\n")
    print(f"#Classes hit (top-1): {num_classes_hit:6d}\n")

    return IS, FID, ENT, num_classes_hit

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    set_seed(args.seed)
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
    args.exp_name = ('%s-%s-%s' % (args.dataset, args.model, args.criterion))
    print(args.exp_name)
    make_path(args.output_dir)
    make_path(args.output_dir + args.exp_name)
    args.image_dir = args.output_dir + args.exp_name + '/image/'
    args.coverage_dir = args.output_dir + args.exp_name + '/coverage/'
    args.log_dir = args.output_dir + args.exp_name + '/log/'
    make_path(args.image_dir)
    make_path(args.coverage_dir)
    make_path(args.log_dir)
    
    model, layer_size_dict, num_class, train_loader, test_loader, seed_loader, image_numpy_list, label_numpy_list = prepare_data_model(args)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)
    final_layer = trainable_module_name[-1]

    # Coverage method
    if args.use_sc:
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
    
    initial_coverage = copy.deepcopy(criterion.current)
    print('Initial Coverage: %f' % initial_coverage)
    engine = Fuzzer(args, criterion, guided=args.guided)
    engine.run(image_numpy_list, label_numpy_list)
    engine.exit()
    IS, FID, ENT, NCLASS = compute_metrics(engine.params, image_numpy_list, model)

    mutated_mode = "GenAI" if args.genai_only else "Normal"
    csv_path = os.path.join(args.output_dir, f"fuzz_results_{args.dataset}_{args.model}_{mutated_mode}.csv")
    header   = ["dataset", "model", "criterion",
                "coverage_gain", "faults", "outputs", "faults/outputs",
                "IS", "FID", "Entropy", "Classes"]
    
    num_faults  = engine.num_ae
    num_outputs = engine.delta_batch if engine.delta_batch else 1
    faults_per_output = num_faults / num_outputs
    coverage_inc = (criterion.current - engine.initial_coverage if args.guided else 0.0)
    row = [args.dataset, args.model, args.criterion,
           f"{coverage_inc:.6f}", num_faults, num_outputs,
           f"{faults_per_output:.6f}",
           f"{IS:.3f}" if IS is not None else "",
           f"{FID:.3f}" if FID is not None else "",
           f"{ENT:.3f}" if ENT is not None else "",
           f"{NCLASS:d}"  if NCLASS is not None else "",]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"Results saved to {csv_path}")

# [vgg16-cifar10, resnet18-cifar10, resnet18-imagenet]
# python ./fuzz_guide/run_fuzz.py --dataset CIFAR10 --data-path ./datasets/CIFAR10/ --model vgg16 --saved-model /torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth --criterion Wisdom --output-dir ./fuzz_guide/fuzz_outputs/ --log-dir ./logs --seed 42 --device 'cuda:0' --guided
# python ./fuzz_guide/run_fuzz.py --dataset CIFAR10 --data-path ./datasets/CIFAR10/ --model resnet18 --saved-model /torch-deepimportance/models_info/saved_models/resnet18_CIFAR10_whole.pth --criterion Wisdom --output-dir ./fuzz_guide/fuzz_outputs/ --log-dir ./logs --seed 42 --device 'cuda:0' --guided
# python ./fuzz_guide/run_fuzz.py --dataset ImageNet --data-path /data/shenghao/dataset/ImageNet/ --model resnet18 --saved-model /torch-deepimportance/models_info/saved_models/resnet18_IMAGENET_patched_whole.pth --criterion Random --output-dir ./fuzz_guide/fuzz_outputs/ --log-dir ./logs --seed 42 --device 'cuda:0' --guided
if __name__ == '__main__':
    main()
    print("Fuzzing completed.")
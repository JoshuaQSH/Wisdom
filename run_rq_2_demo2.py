import random
import time
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from collections import defaultdict

from torch.utils.data import DataLoader, ConcatDataset, Dataset
from src.attribution import get_relevance_scores_dataloader
from src.utils import get_data, parse_args, get_model, get_trainable_modules_main, _configure_logging
# from src.idc import IDC
from src.wisdom import WisdomIDC
from src.deepidc import DeepIDC
from src.nlc_coverage import calculate_coverage_ratio


"""
U_I (Importance-perturbed): Each image has Gaussian white noise (mean 0, std 0.3) added to its most important 2% pixels.
U_R (Random-perturbed): Each image has noise added to a random 2% of its pixels.

U_IO (Importance-perturbed Dataset + Original Dataset): Original images with noise added to the most important 2% pixels.
U_RO (Random-perturbed Dataset + Original Dataset): Original images with noise added to a random 2% of its pixels.

# Step 1: Get the relevance maps for the test set using Attribution methods (e.g., LRP).
# Step 2: For each image, find the top 2% most important pixels based on the relevance map (Baseline: random & LRP) and form the new dataset.
# Step 3: Coverage testing for the perturbed dataset.

"""

# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------

TOPK = 0.02  # Top-k fraction of pixels to perturb (2%)
gaussian_STD = 0.5  # Standard deviation for Gaussian noise
cmap = ['PuBuGn', 'Greens', 'Purples', 'Reds', 'Blues', 'YlGn', 'summer', 'cool', 'bwr']
start_ms = int(time.time() * 1000)
TIMESTAMP = time.strftime("%Y%m%d‑%H%M%S", time.localtime(start_ms / 1000))
acts = defaultdict(list)

cluster_name_all = ["KMeans", "MiniBatchKMeans", "BisectingKMeans", 
                    "AgglomerativeClustering", "SpectralClustering", "DBSCAN", 
                    "OPTICS", "HDBSCAN", "MeanShift", "AffinityPropagation", "Birch"]
clustering_params_all = {
    "KMeans": {"n_clusters": 2, "random_state": 42, "n_init": 10},
    "MiniBatchKMeans": {"n_clusters": 2, "batch_size": 32, "max_iter": 100, "random_state": 42},
    "BisectingKMeans": {"n_clusters": 2, "random_state": 42, "n_init": 10},
    "AgglomerativeClustering": {"n_clusters": 2, "linkage": "ward", "metric": "euclidean"},
    "SpectralClustering": {"n_clusters": 2, "affinity": "rbf", "assign_labels": "kmeans"},
    "DBSCAN": {"eps": 0.1, "min_samples": 10, "metric": "euclidean"},
    "OPTICS": {"min_samples": 2, "xi": 0.05, "min_cluster_size": 2},
    "HDBSCAN": {"min_cluster_size": 2, "min_samples": 2, "cluster_selection_epsilon": 0.01, "cluster_selection_method": "eom"},
    "MeanShift": {"bandwidth": 0.5, "bin_seeding": True, "cluster_all": False, "max_iter": 300, "min_bin_freq": 1},
    "AffinityPropagation": {"damping": 0.9, "preference": -50},
    "Birch": {"threshold": 0.5, "n_clusters": 2},
}

def prapare_data_models(args):
    # Logger settings
    logger = _configure_logging(args.logging, args, 'debug')
    
    # Model settings
    model_path = os.getenv("HOME") + args.saved_model
    
    # Model loading
    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)

    return model, module_name, module, trainable_module, trainable_module_name, logger

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_csv_results(updated_column_dict, csv_path='results.csv', tag='original'):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        for column_name, value in updated_column_dict.items():
            df.loc[tag, column_name] = value
        df.to_csv(csv_path)
    else:
        df = pd.DataFrame(updated_column_dict, index=[tag])
        mode  = 'a' if os.path.exists(csv_path) else 'w'
        header= False if mode == 'a' else True
        df.to_csv(csv_path, mode=mode, header=header)

    print(f"[{tag}] Updated results saved to {csv_path}")

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

# -----------------------------------------------------------
# Dataset Wrapper
# -----------------------------------------------------------
def build_masks_exclusive(saliency: torch.Tensor, 
                          k: float,
                          gen = None,
                          device = 'cuda'):
    """
    Build (important_mask, random_mask) with NON-OVERLAP:
      - important_mask: top-k% |saliency| pixels set to 1
      - random_mask:    k% pixels sampled UNIFORMLY from the COMPLEMENT of important_mask

    saliency: (H, W) or (1, H, W) tensor (CPU or GPU)
    k:        fraction in (0,1]
    gen:      optional torch.Generator for deterministic sampling
    device:   optional device for mask tensors (defaults to saliency.device)

    Returns two tensors of shape (1, H, W) on `device`, dtype=float32.
    """
    if not (0 < k <= 1):
        raise ValueError("k must be in (0, 1].")

    if saliency.dim() == 3 and saliency.size(0) == 1:
        sal = saliency[0]
    elif saliency.dim() == 2:
        sal = saliency
    else:
        raise ValueError("saliency must be (H,W) or (1,H,W).")

    device = device or sal.device
    H, W = sal.shape
    n = H * W
    kpix = max(1, int(round(k * n)))

    # Important = top-k by absolute saliency
    flat = sal.abs().reshape(-1)
    if kpix >= n:
        imp_flat = torch.ones(n, dtype=torch.float32, device=device)
    else:
        # Use topk threshold to include ties consistently
        # (move to CPU if needed for efficiency, then back)
        work = flat if flat.device.type == "cpu" else flat.cpu()
        thresh = torch.topk(work, kpix, largest=True).values.min()
        mask_cpu = (work >= thresh).float()
        imp_flat = mask_cpu.to(device)

    # Random from complement
    comp_idx = torch.nonzero(imp_flat == 0, as_tuple=False).view(-1)
    if comp_idx.numel() == 0:
        # Degenerate (all important). Mirror the imp mask so masks have same cardinality.
        rand_flat = imp_flat.clone()
    else:
        # Deterministic sampling using a per-dataset generator if provided
        # torch.randperm can take generator on CPU; ensure indices on same device later.
        cpu_gen = gen if (gen is not None and gen.device == torch.device('cpu')) else gen
        perm = torch.randperm(comp_idx.numel(), generator=cpu_gen, device='cpu')
        sel = comp_idx.cpu()[perm[:kpix]]
        rand_flat = torch.zeros(n, dtype=torch.float32, device='cpu')
        rand_flat[sel] = 1.0
        rand_flat = rand_flat.to(device)

    imp = imp_flat.view(1, H, W)
    rand = rand_flat.view(1, H, W)
    return imp, rand


class PerturbedDataset(Dataset):
    """
    Lazily generates perturbed samples from a base dataset.

    - mode='important': perturb the top-k fraction of pixels (by |saliency|)
    - mode='random':    perturb k% pixels sampled uniformly from the COMPLEMENT
                        of the important set (non-overlapping)
    - strategy:         a callable that maps (1,C,H,W) -> (H,W) saliency tensor
                        (e.g., LRPStrategy or WisdomStrategy)

    Notes:
      * This dataset does not copy the entire set; it computes one sample at a time.
      * Random selection is deterministic per dataset if you pass a fixed seed.
      * Noise is added in the same normalized space as your model inputs.
    """

    def __init__(self,
                 base_dataset: Dataset,
                 strategy,
                 k: float,
                 std: float,
                 mode: str = "important",
                 seed: int = 42):
        if mode not in ("important", "random"):
            raise ValueError("mode must be 'important' or 'random'")
        if not (0 < k <= 1):
            raise ValueError("k must be in (0, 1].")

        self.base_dataset = base_dataset
        self.strategy = strategy
        self.k = float(k)
        self.std = float(std)
        self.mode = mode

        # Deterministic generator for random masks
        self.gen = torch.Generator(device='cpu')
        self.gen.manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, label = self.base_dataset[idx]  # expects img as (C,H,W) tensor
        if not torch.is_tensor(img):
            raise TypeError("Base dataset must return tensors for images (C,H,W).")

        # 1. Compute saliency (CPU or GPU inside strategy)
        x = img.unsqueeze(0)  # (1,C,H,W)
        sal = self.strategy(x)  # (H,W), typically on CPU

        # 2. Build non-overlapping masks using the same saliency
        imp_mask, rand_mask = build_masks_exclusive(
            saliency=sal,
            k=self.k,
            gen=self.gen,
            device=img.device
        )

        # 3. Choose mask and perturb
        mask = imp_mask if self.mode == "important" else rand_mask
        x_pert = add_gaussian_noise(img, mask, std=self.std)

        return x_pert, int(label)

def make_wisdom_strategy(model, csv_path: str):
    model.eval().to(next(model.parameters()).device)
    df = pd.read_csv(csv_path)
    df = df[df.Score != 0]
    layer2score = defaultdict(dict)
    for _, row in df.iterrows():
        layer2score[row.LayerName][int(row.NeuronIndex)] = float(row.Score)

    named = dict(model.named_modules())

    def strategy(x: torch.Tensor) -> torch.Tensor:
        device = next(model.parameters()).device
        x = x.to(device)
        H, W = x.shape[-2:]
        activations = {}

        handles = []
        try:
            for name, score_dict in layer2score.items():
                if name not in named:
                    continue
                mod = named[name]
                def _make_hook(lname=name, sd=score_dict):
                    def _hook(_, __, out):
                        acts = out.detach()
                        # (B,C,H,W) or (B,C)
                        if acts.dim() == 2:
                            acts = acts.unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
                        # weight channels
                        C = acts.size(1)
                        score_vec = torch.zeros(C, device=acts.device)
                        for idx, s in sd.items():
                            if 0 <= idx < C: score_vec[idx] = s
                        weighted = acts * score_vec.view(1, -1, 1, 1)  # (B,C,h,w)
                        activations[lname] = weighted
                    return _hook
                handles.append(mod.register_forward_hook(_make_hook()))

            with torch.no_grad():
                _ = model(x)

            heat = torch.zeros(1, 1, H, W, device=device)
            for lname, acts in activations.items():
                if acts.size(2) != H or acts.size(3) != W:
                    acts = F.interpolate(acts, size=(H, W), mode="bilinear", align_corners=False)
                layer_heat = acts.sum(dim=1, keepdim=True)  # (1,1,H,W)
                heat += layer_heat
            return heat[0, 0].detach().cpu()  # (H,W)

        finally:
            for h in handles:
                h.remove()

    return strategy


def get_generated_dataset_optimized(args, model, test_dataset, logger):

    strategy = make_wisdom_strategy(model, args.csv_file)
    U_I_dataset = PerturbedDataset(base_dataset=test_dataset, 
                                   strategy=strategy,
                                   k=TOPK, 
                                   std=gaussian_STD,
                                   mode='important',
                                   seed=42)
    U_R_dataset = PerturbedDataset(base_dataset=test_dataset, 
                                   strategy=strategy,
                                   k=TOPK, 
                                   std=gaussian_STD,
                                   mode='random',
                                   seed=42)

    # Build the concatenated datasets without pre‑allocating all images
    U_IO_dataset = ConcatDataset([test_dataset, U_I_dataset])  # original + importance‑perturbed
    U_RO_dataset = ConcatDataset([test_dataset, U_R_dataset])  # original + random‑perturbed

    # Create dataloaders (no additional memory overhead beyond batch size)
    U_IO_loader = DataLoader(U_IO_dataset, batch_size=args.batch_size, shuffle=False)
    U_RO_loader = DataLoader(U_RO_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"[Sanity] Generated datasets: U_I: {len(U_I_dataset)}, U_R: {len(U_R_dataset)}, U_IO: {len(U_IO_dataset)}, U_RO: {len(U_RO_dataset)}")
    
    del U_I_dataset, U_R_dataset, U_IO_dataset, U_RO_dataset
    
    return U_IO_loader, U_RO_loader



# -----------------------------------------------------------
# Coverage Method
# -----------------------------------------------------------
"""
choices = ['NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'CC', 'NLC', 'LSC', 'DSC', 'MDSC']

Hyperparameters:
@NC: Activation threshold \in {0.25, 0.50, 0.75}
@KMNC: Number of segments per neuron \in {100, 1000, 10000}
@SNAC: Parameter-free
@NBC: Parameter-free
@TKNC: Top k neurons \in {1, 10, 50}
@TKNP: Pattern width \in {1, 10, 50}
@CC: Radius T (#neurons per cluster) \in {10, 20, 50}
@NLC: Parameter-free
@LSC: Bucket width/threshold \in {1, 10, 100}
@DSC: Bucket width \in {0.01, 0.1, 1}
@MDSC: Bucket width \in {1, 10, 100}
"""

def calculate_all_coverage_ratios(build_loader, target_loader, model, device, num_class=10):

    results = {}
    
    # Methods with their typical hyperparameters
    methods_config = {
        'NC': {'hyper': 0.5},
        'KMNC': {'hyper': 1000},
        'SNAC': {'hyper': None},
        'NBC': {'hyper': None}, 
        'TKNC': {'hyper': 10},
        'TKNP': {'hyper': 10},
        'LSC': {'hyper': 10, 'min_var': 1e-5, 'num_class': num_class},
        'DSC': {'hyper': 0.1, 'min_var': 1e-5, 'num_class': num_class},
        'MDSC': {'hyper': 10, 'min_var': 1e-5, 'num_class': num_class},
        'NLC': {'hyper': None},
        'CC': {'hyper': 10}
    }
    
    for method, config in methods_config.items():
        ratio = calculate_coverage_ratio(
                build_loader, target_loader, method, model,
                hyper=config['hyper'],
                device=device,
                min_var=config.get('min_var', 1e-5),
                num_class=config.get('num_class', num_class)
            )
        results[method] = ratio
        print(f"{method}: {ratio:.4f}")
    return results

def calculate_single_coverage_ratios(build_loader, target_loader, model, device, method_name, num_class=10):
    results = {}
    
    # Methods with their typical hyperparameters
    methods_config = {
        'NC': {'hyper': 0.5},
        'KMNC': {'hyper': 1000},
        'SNAC': {'hyper': None},
        'NBC': {'hyper': None}, 
        'TKNC': {'hyper': 10},
        'TKNP': {'hyper': 10},
        'LSC': {'hyper': 10, 'min_var': 1e-5, 'num_class': num_class},
        'DSC': {'hyper': 0.1, 'min_var': 1e-5, 'num_class': num_class},
        'MDSC': {'hyper': 10, 'min_var': 1e-5, 'num_class': num_class},
        'NLC': {'hyper': None},
        'CC': {'hyper': 10}
    }
    
    if method_name in methods_config:
        config = methods_config[method_name]
        ratio = calculate_coverage_ratio(
                build_loader, target_loader, method_name, model,
                hyper=config['hyper'],
                device=device,
                min_var=config.get('min_var', 1e-5),
                num_class=config.get('num_class', num_class)
            )
        results[method_name] = ratio
        print(f"{method_name}: {ratio:.4f}")
    else:
        print(f"{method_name} is not a valid method.")

    return results
        
# -----------------------------------------------------------
# Wisdom-based input trace
# -----------------------------------------------------------
def wisdom_coverage(args, model, train_loader, test_loader, logger, cluster_method_name, device, tag='original'):
    df = pd.read_csv(args.csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(args.top_m_neurons)
    top_k_neurons = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        top_k_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)
    
    if args.use_silhouette:
        cluster_info = f"_{cluster_method_name}_silhouette_"
    else:
        cluster_info = f"_{cluster_method_name}_"

    extra = clustering_params_all[cluster_method_name]
    cache_path = "./cluster_pkl/" + args.model + "_" + args.dataset + "_top_" + str(args.top_m_neurons) + cluster_info + "wisdom_clusters.pkl"
    
    wisdom_idc = WisdomIDC(
        model,
        args.top_m_neurons,
        args.n_clusters,
        args.use_silhouette,
        args.all_class,
        cluster_method_name,
        device,
        extra,
        cache_path
    )
    
    train_acts  = wisdom_idc.get_selected_activations(train_loader, top_k_neurons)
    cluster_groups = wisdom_idc.cluster_per_neuron(train_acts)
    test_acts = wisdom_idc.get_selected_activations(test_loader, top_k_neurons)
    coverage_rate, total_comb, max_cov = wisdom_idc.compute_coverage(test_acts, cluster_groups)
    
    logger.info(f"[WisdomIDC] [{tag}] Coverage: {coverage_rate:.6f} | total combinations: {total_comb} | max_coverage: {max_cov:.6f}")
    results = {}
    results['WISDOM'] = coverage_rate  
    df = pd.DataFrame(results, index=[tag])
    save_csv_results(results, "rq2_results_{}_{}_top_{}_{}.csv".format(args.dataset, args.model, args.top_m_neurons, TIMESTAMP), tag=tag)
    logger.info(f"Total Combination: {total_comb}, Max Coverage: {max_cov:.4f}, IDC Coverage: {coverage_rate:.4f}, Attribution: WISDOM")

    del wisdom_idc
    del train_acts
    del cluster_groups
    del test_acts
    torch.cuda.empty_cache()  # Clear GPU memory

    return coverage_rate

# -----------------------------------------------------------
# Run full coverage suite on one dataloader with other methods
# -----------------------------------------------------------
def run_coverage_suite_single(model,
                       build_loader,   # clean data for build phases
                       target_loader,  # UI_loader / UR_loader / etc.
                       num_classes,
                       method_name,
                       device,
                       logger,
                       tag: str = 'dataset', *args, **kwargs):
    """
    Returns a dict {metric_name: coverage_value} for the given target_loader.
    """
    
    results = calculate_single_coverage_ratios(build_loader, target_loader, model, device, method_name, num_class=num_classes)
    model_name = kwargs.get('model_name')
    dataset_name = kwargs.get('dataset_name')
    df = pd.DataFrame(results, index=[tag])
    logger.info(f"=== Coverage results for {tag} ===")
    logger.info(df.to_string(float_format=lambda x: f"{x:.4f}"))
    
    save_csv_results(results, "rq2_results_{}_{}_{}.csv".format(dataset_name, model_name, TIMESTAMP), tag=tag)
    return results, df

# -----------------------------------------------------------
# IDC coverage testing
# -----------------------------------------------------------
def idc_coverage(args, model, train_loader, test_loader, cluster_method_name, device, logger, tag='original'):
    # 0) get layer-wise scores (not indices)
    layer_scores = get_relevance_scores_dataloader(
        model, train_loader, device, attribution_method='lrp'
    )
    
    extra = clustering_params_all[cluster_method_name]
    cache_path = (
        "./cluster_pkl/"
        f"{args.model}_{args.dataset}_top_{args.top_m_neurons}_"
        f"{cluster_method_name}{'_sil' if args.use_silhouette else ''}_deepimportance.pkl"
    )
    if "random_state" not in extra:
        extra = {**extra, "random_state": 42}
    if cluster_method_name == "KMeans" and "n_init" not in extra:
        extra = {**extra, "n_init": 10}

    idc = DeepIDC(
        model=model,
        top_m_neurons=args.top_m_neurons,
        n_clusters=extra.get("n_clusters", args.n_clusters),
        use_silhouette=args.use_silhouette,
        test_all_classes=args.all_class,
        clustering_method_name=cluster_method_name,
        device=device,
        clustering_params=extra,
        cache_path=cache_path,
    )

    selected_indices = idc.select_top_neurons(layer_scores)  # Dict[layer]->LongTensor
    train_acts = idc.get_selected_activations(train_loader, selected_indices)
    cluster_groups = idc.cluster_per_neuron(train_acts)
    test_acts = idc.get_selected_activations(test_loader, selected_indices)
    coverage_rate, total_combination, max_coverage = idc.compute_coverage(test_acts, cluster_groups)
    
    results = {}
    results['IDC'] = coverage_rate

    df = pd.DataFrame(results, index=[tag])
    save_csv_results(results, "rq2_results_{}_{}_top_{}_{}.csv".format(args.dataset, args.model, args.top_m_neurons, TIMESTAMP), tag=tag)
    logger.info(f"Total Combination: {total_combination}, Max Coverage: {max_coverage:.4f}, IDC Coverage: {coverage_rate:.4f}, Attribution: LRP")
    
    del idc
    del train_acts
    del test_acts
    torch.cuda.empty_cache()  # Clear GPU memory

    return coverage_rate
    

# -----------------------------------------------------------
# Main entry point
# -----------------------------------------------------------
def main():
    set_seed()
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    
    # Model settings
    model, module_name, module, trainable_module, trainable_module_name, logger = prapare_data_models(args)

    # Data settings
    train_loader, test_loader, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path)
    U_IO_loader, U_RO_loader = get_generated_dataset_optimized(args, model, test_dataset, logger)
    
    cluster_method_name = cluster_name_all[0]

    # Run the coverage suite
    logger.info("=== Running coverage suite ===")    
    # dpo_results = idc_coverage(args, model, train_loader, test_loader, cluster_method_name, device, logger, tag='original')
    # dpr_results = idc_coverage(args, model, train_loader, U_RO_loader, cluster_method_name, device, logger, tag='U_RO')
    # dpi_results = idc_coverage(args, model, train_loader, U_IO_loader, cluster_method_name, device, logger, tag='U_IO')
    
    # dpo_results_w = wisdom_coverage(args, model, train_loader, test_loader, logger, cluster_method_name, device, tag='original')
    # dpr_results_w = wisdom_coverage(args, model, train_loader, U_RO_loader, logger, cluster_method_name, device, tag='U_RO')
    # dpi_results_w = wisdom_coverage(args, model, train_loader, U_IO_loader, logger, cluster_method_name, device, tag='U_IO')
    
    # methods_name = ['NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'LSC', 'DSC', 'MDSC', 'NLC', 'CC']
    methods_name = ['DSC']

    for method in methods_name:
        uo_results, _ = run_coverage_suite_single(model, train_loader, test_loader, len(classes), method, device, logger, tag='original', skip_train=True, model_name=args.model, dataset_name=args.dataset)
        ur_results, _ = run_coverage_suite_single(model, train_loader, U_RO_loader, len(classes), method, device, logger, tag='U_RO', skip_train=True, model_name=args.model, dataset_name=args.dataset)
        ui_results, _ = run_coverage_suite_single(model, train_loader, U_IO_loader, len(classes), method, device, logger, tag='U_IO', skip_train=True, model_name=args.model, dataset_name=args.dataset)

if __name__ == '__main__':
    main()
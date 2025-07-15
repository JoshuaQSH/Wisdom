import random
import time
import os
import warnings

import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.attribution import get_relevance_scores_dataloader
from src.utils import get_data, parse_args, get_model, get_trainable_modules_main, _configure_logging
from src.idc import IDC
from src.nlc_coverage import (
    NC, KMNC, NBC, SNAC, TKNC, TKNP, CC,
    NLC, LSC, DSC, MDSC
)
from src.nlc_tool import get_layer_output_sizes

import matplotlib.pyplot as plt

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

color_14 = [
    "#F4EFFF", "#E5DAFA", "#D6C5F6", "#C7B0F1",
    "#B89BED", "#A986E8", "#9A71E4", "#8B5CDF",
    "#7C47DB", "#6D32D6", "#5E1DD2", "#4F08CD",
    "#3A00A8", "#260080"
]

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

def _make_synthetic_loader(n_samples, batch_size, channels, size, num_class) -> DataLoader:
    """Return DataLoader of random tensors in [0,1] and random labels."""
    if n_samples <= 0:
        raise ValueError("n_samples must be >0 for synthetic loader")
    h, w = size
    data = torch.rand(n_samples, channels, h, w)
    labels = torch.randint(0, num_class, (n_samples,))
    ds = TensorDataset(data, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

def synth_data(synthetic_build=5000, synthetic_target=1000, batch_size=128, channels=3, input_size=[32, 32], num_class=10):
    train_loader = _make_synthetic_loader(
        synthetic_build,
        batch_size,
        channels,
        tuple(input_size),
        num_class,
    )
    test_loader = _make_synthetic_loader(
        synthetic_target,
        batch_size,
        channels,
        tuple(input_size),
        num_class,
    )
    classes = list(range(num_class))
    
    return train_loader, test_loader, classes

def viz_scale(df, ts, logger):

    # Sort by batch_size for a sensible line ordering
    df_sorted = df.sort_values('batch_size')

    # Create the plot
    plt.figure()
    plt.scatter(df_sorted['batch_size'], df_sorted['throughput'])
    plt.plot(df_sorted['batch_size'], df_sorted['throughput'])
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (img/s)')
    # plt.title('Throughput vs Batch Size')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    out_file = f"scale_{ts}.pdf"
    plt.savefig(out_file, dpi=1200, format='pdf')
    logger.info("Saved plot to %s", out_file)

def viz_bench(df, ts, logger):
    for metric, ylabel in (("elapsed_sec", "Overhead (s)"), ("build_time", "Build time (s)"), ("run_time", "Runtime (s)"), ("throughput", "Throughput (img/s)")):
        fig, ax = plt.subplots()
        methods = df["method"].unique()
        devices = df["device"].unique()
        x = np.arange(len(methods))
        width = 0.35
        for i, dev in enumerate(devices):
            for j, m in enumerate(methods):
                val = df[(df.device == dev) & (df.method == m)][metric].values[0]
                hatch = '//' if j == len(methods) - 1 else None
                ax.bar(
                    x[j] + i * width, val, width,
                    label=dev if j == 0 else "", 
                    color=color_14[j],
                    hatch=hatch
                )
        ax.set_xticks(x + width * (len(devices) - 1) / 2)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" – ")[0])
        # ax.legend()
        fig.tight_layout()
        out_file = f"coverage_{metric}_{ts}.pdf"
        fig.savefig(out_file, dpi=1200, format='pdf')
        logger.info("Saved plot to %s", out_file)

def run_bench(coverage_methods, method, model, layer_size_dict, build_loader, target_loader, classes, hyper=None, device='cpu', **kwargs):
    model.to(device)

    # Surprise coverage methods, min_var and num_class are required
    if method in ['LSC', 'DSC', 'MDSC']:
        min_var = kwargs.get('min_var', 1e-5)
        num_class = kwargs.get('num_class', 10)
        cov = coverage_methods[method](model, device, layer_size_dict, hyper, min_var=min_var, num_class=num_class)
    
    # DeepImportance and Wisdom
    elif method in ['DeepImportance(w/o-vec)', 'DeepImportance(w/-vec)', 'Wisdom']:
        top_m_neurons = kwargs.get('top_m_neurons', 10)
        n_clusters = kwargs.get('n_clusters', 2)
        use_silhouette = kwargs.get('use_silhouette', False)
        all_class = kwargs.get('all_class', False)
        cache_path = kwargs.get('cache_path', None)
        cov = IDC(
            model,
            top_m_neurons,
            n_clusters,
            use_silhouette,
            all_class,
            "KMeans",
            None,
            cache_path
        )

    # Traditional coverage methods
    else:
        cov = coverage_methods[method](model, device, layer_size_dict, hyper)

    start = time.perf_counter()
    if method == "DeepImportance(w/-vec)":
        final_layer = kwargs.get('final_layer', None)
        layer_relevance_scores = get_relevance_scores_dataloader(
            model,
            build_loader,
            device,
            attribution_method='lrp',
        )
        important_neuron_indices, inorderd_indices = cov.select_top_neurons_all(layer_relevance_scores, final_layer)
        activation_values, selected_activations = cov.get_activations_model_dataloader(build_loader, important_neuron_indices)
        selected_activations = {k: v.half().cpu() for k, v in selected_activations.items()}
        cluster_groups = cov.cluster_activation_values_all(selected_activations)
    
    elif method == "DeepImportance(w/o-vec)":
        final_layer = kwargs.get('final_layer', None)
        per_train_loader = kwargs.get('per_train_loader', None)
        layer_relevance_scores = get_relevance_scores_dataloader(
            model,
            per_train_loader,
            device,
            attribution_method='lrp',
        )
        important_neuron_indices, inorderd_indices = cov.select_top_neurons_all(layer_relevance_scores, final_layer)
        activation_values, selected_activations = cov.get_activations_model_dataloader(per_train_loader, important_neuron_indices)
        selected_activations = {k: v.half().cpu() for k, v in selected_activations.items()}
        cluster_groups = cov.cluster_activation_values_all(selected_activations)
        
    elif method == "Wisdom":
        csv_file = kwargs.get('csv_file', None)
        df = pd.read_csv(csv_file)
        df_sorted = df.sort_values(by='Score', ascending=False).head(top_m_neurons)
        important_neuron_indices = {}
        for layer_name, group in df_sorted.groupby('LayerName'):
            important_neuron_indices[layer_name] = torch.tensor(group['NeuronIndex'].values)
        
        activation_values, selected_activations = cov.get_activations_model_dataloader(build_loader, important_neuron_indices)
        selected_activations = {k: v.half().cpu() for k, v in selected_activations.items()}
        cluster_groups = cov.cluster_activation_values_all(selected_activations)
    else:
        # Build baseline if needed
        cov.build(build_loader)
    build_time = time.perf_counter() - start

    if method in ['DeepImportance(w/-vec)', 'Wisdom']:
        coverage, total_combination, max_coverage = cov.compute_idc_test_whole_dataloader(target_loader, 
            important_neuron_indices, 
            cluster_groups)
    elif method == 'DeepImportance(w/o-vec)':
        per_test_loader = kwargs.get('per_test_loader', None)
        coverage, total_combination, max_coverage = cov.compute_idc_test_whole_dataloader(per_test_loader, 
            important_neuron_indices, 
            cluster_groups)
    else:
        cov.assess(target_loader)
        coverage = cov.current
    elapsed = time.perf_counter() - start
    imgs = len(target_loader.dataset)

    return build_time, elapsed, imgs / elapsed, coverage

def run_scale_bench(model, build_loader, target_loader, classes, device='cpu', **kwargs):
    model.to(device)
    top_m_neurons = kwargs.get('top_m_neurons', 10)
    n_clusters = kwargs.get('n_clusters', 2)
    use_silhouette = kwargs.get('use_silhouette', False)
    all_class = kwargs.get('all_class', False)
    cache_path = kwargs.get('cache_path', None)
    cov = IDC(
            model,
            top_m_neurons,
            n_clusters,
            use_silhouette,
            all_class,
            "KMeans",
            None,
            cache_path
    )


    start = time.perf_counter()
    csv_file = kwargs.get('csv_file', None)
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(top_m_neurons)
    important_neuron_indices = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        important_neuron_indices[layer_name] = torch.tensor(group['NeuronIndex'].values)
        
    activation_values, selected_activations = cov.get_activations_model_dataloader(build_loader, important_neuron_indices)
    selected_activations = {k: v.half().cpu() for k, v in selected_activations.items()}
    cluster_groups = cov.cluster_activation_values_all(selected_activations)
    
    build_time = time.perf_counter() - start
    coverage, total_combination, max_coverage = cov.compute_idc_test_whole_dataloader(target_loader, 
            important_neuron_indices, 
            cluster_groups)

    elapsed = time.perf_counter() - start
    imgs = len(target_loader.dataset)

    return build_time, elapsed, imgs / elapsed, coverage

def scaliblity_run_main():
    set_seed()
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    # Model settings
    model, module_name, module, trainable_module, trainable_module_name, logger = prapare_data_models(args)

    methods_config = {
        'Wisdom': {'top_m_neurons': args.top_m_neurons, 
                   'n_clusters': args.n_clusters, 
                   'use_silhouette': args.use_silhouette,
                   'all_class': args.all_class,
                   'cache_path': "./cluster_pkl/" + args.model + "_" + args.dataset + "_" + "_wisdom_clusters.pkl",
                   'csv_file': args.csv_file, 
                },
    }
    records = []
    for batch_size in [16, 32, 64, 128, 256, 512, 1024]:
        synthetic_build = 10000
        synthetic_target = 5000
        logger.warning("Using synthetic dataset: build=%d, target=%d", synthetic_build, synthetic_target)
        train_loader, test_loader, classes = synth_data(synthetic_build=synthetic_build, 
                   synthetic_target=synthetic_target, 
                   batch_size=batch_size, 
                   channels=3, 
                   input_size=[32, 32], 
                   num_class=10)
        per_train_loader = DataLoader(train_loader.dataset, batch_size=1, shuffle=True)
        per_test_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)
        
        num_class = len(classes)

        build_time, elapsed, thrpt, coverage = run_scale_bench(model, 
                        train_loader, 
                        test_loader, 
                        classes, 
                        device=device,
                        top_m_neurons=args.top_m_neurons,
                        n_clusters=args.n_clusters,
                        use_silhouette=args.use_silhouette,
                        all_class=args.all_class,
                        cache_path="./cluster_pkl/" + args.model + "_" + args.dataset + "_" + "_wisdom_clusters.pkl",
                        csv_file=args.csv_file
        )

        records.append(
                {
                    "model": args.model,
                    "device": str(device),
                    "batch_size": batch_size,
                    "build_time": build_time,
                    "elapsed_sec": elapsed,
                    "run_time": (elapsed - build_time),
                    "throughput": thrpt,
                    "coverage": coverage,
                }
            )
              
    df = pd.DataFrame(records)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"benchmark_wisdomscale_{ts}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV to: %s", csv_path)

    viz_scale(df, ts, logger)

def main():
    set_seed()
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    
    # Model settings
    model, module_name, module, trainable_module, trainable_module_name, logger = prapare_data_models(args)

    # Data settings
    if args.dataset == 'synthetic':
        synthetic_build = 5000
        synthetic_target = 1000
        logger.warning("Using synthetic dataset: build=%d, target=%d", synthetic_build, synthetic_target)
        train_loader, test_loader, classes = synth_data(synthetic_build=synthetic_build, 
                   synthetic_target=synthetic_target, 
                   batch_size=args.batch_size, 
                   channels=3, 
                   input_size=[32, 32], 
                   num_class=10)
        per_train_loader = DataLoader(train_loader.dataset, batch_size=1, shuffle=True)
        per_test_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)
    else:
        train_loader, test_loader, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path)
        per_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        per_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    num_class = len(classes)
    records = []
    coverage_methods = {
        'NC': NC,
        'KMNC': KMNC, 
        'SNAC': SNAC,
        'NBC': NBC,
        'TKNC': TKNC,
        'TKNP': TKNP,
        'LSC': LSC,
        'DSC': DSC, 
        'MDSC': MDSC,
        'NLC': NLC,
        'CC': CC,
        'DeepImportance(w/o-vec)': None,
        'DeepImportance(w/-vec)': None,
        'Wisdom': None
    }
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
        'CC': {'hyper': 10},
        'DeepImportance(w/o-vec)': {'top_m_neurons': args.top_m_neurons, 
                           'n_clusters': args.n_clusters, 
                           'use_silhouette': args.use_silhouette,
                           'all_class': args.all_class,
                           'cache_path': "./cluster_pkl/" + args.model + "_" + args.dataset + "_" + "_deepimportance_old_clusters.pkl",
                           'final_layer': trainable_module_name[-1],
                           'per_train_loader': per_train_loader,
                           'per_test_loader': per_test_loader,
                        },
        'DeepImportance(w/-vec)': {'top_m_neurons': args.top_m_neurons, 
                           'n_clusters': args.n_clusters, 
                           'use_silhouette': args.use_silhouette,
                           'all_class': args.all_class,
                           'cache_path': "./cluster_pkl/" + args.model + "_" + args.dataset + "_" + "_deepimportance_clusters.pkl",
                           'final_layer': trainable_module_name[-1], 
                        },
        'Wisdom': {'top_m_neurons': args.top_m_neurons, 
                   'n_clusters': args.n_clusters, 
                   'use_silhouette': args.use_silhouette,
                   'all_class': args.all_class,
                   'cache_path': "./cluster_pkl/" + args.model + "_" + args.dataset + "_" + "_wisdom_clusters.pkl",
                   'csv_file': args.csv_file, 
                },
    }
    
    # Get a sample to determine layer sizes
    sample_batch, *_ = next(iter(train_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    num_neuron = 0
    layer_size_dict = get_layer_output_sizes(model.to(device), sample_batch.to(device))
    for layer_name in layer_size_dict.keys():
        num_neuron += layer_size_dict[layer_name][0]
    
    # run_bench(coverage_methods, method, model, layer_size_dict, build_loader, target_loader, classes, hyper=None, device='cpu', **kwargs)
    model.eval()
    logger.info("=== Device: %s ===", device)
    
    # for method, config in methods_config.items():
    for method, config in tqdm(methods_config.items(), desc=f"Benchmarking for {args.model} with {args.dataset}"):
        # logger.info("-- %s", method)
        build_time, elapsed, thrpt, coverage = run_bench(coverage_methods=coverage_methods, 
                                                                  method=method, 
                                                                  model=model, 
                                                                  layer_size_dict=layer_size_dict, 
                                                                  build_loader=train_loader, 
                                                                  target_loader=test_loader, 
                                                                  classes=classes, 
                                                                  hyper=config.get('hyper', None),
                                                                  device=device, 
                                                                  min_var=config.get('min_var', 1e-5),
                                                                  num_class=config.get('num_class', num_class),
                                                                  top_m_neurons=config.get('top_m_neurons', 10),
                                                                  n_clusters=config.get('n_clusters', 2),
                                                                  use_silhouette=config.get('use_silhouette', False),
                                                                  all_class=config.get('all_class', False),
                                                                  cache_path=config.get('cache_path', None),
                                                                  final_layer=config.get('final_layer', trainable_module_name[-1]),
                                                                  csv_file=config.get('csv_file', args.csv_file),
                                                                  per_train_loader=config.get('per_train_loader', train_loader),
                                                                  per_test_loader=config.get('per_test_loader', test_loader),
                                                        )
        records.append(
                {
                    "method": method,
                    "model": args.model,
                    "total_layers": len(layer_size_dict.keys()),
                    "total_neurons": num_neuron,
                    "device": str(device),
                    "build_time": build_time,
                    "elapsed_sec": elapsed,
                    "run_time": (elapsed - build_time),
                    "throughput": thrpt,
                    "coverage": coverage,
                }
            )
              
    df = pd.DataFrame(records)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"benchmark_results_{ts}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV to: %s", csv_path)
    
    # Visualization
    viz_bench(df, ts, logger)

# python run_rq_5_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '../datasets/' --batch-size 128 --device 'cuda:0' --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/lenet_cifar10.csv' --attr lrp --top-m-neurons 10
# python run_rq_5_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset synthetic --batch-size 128 --device 'cuda:0' --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/lenet_cifar10.csv' --attr lrp --top-m-neurons 10 
# python run_rq_5_demo.py --model vgg16 --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset synthetic --batch-size 128 --device 'cuda:0' --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/vgg16_cifar10.csv' --attr lrp --top-m-neurons 10 
if __name__ == "__main__":
    # main()
    scaliblity_run_main()
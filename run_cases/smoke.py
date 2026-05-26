"""Small smoke entrypoints for score generation, optimized RQ2, and BO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimize.run_rq2_opt import run_rq2_opt
from run_cases.support import get_data, resolve_saved_model_path
from wisdom import ClusteringConfig, run_bo, WisdomConfig, WisdomIDC
from wisdom.utils.common import get_model
from wisdom_yolo_train import COCOImageDataset, train_wisdom_yolo


DEFAULT_DNN_MODEL = './models_info/saved_models/lenet_MNIST_whole.pth'
DEFAULT_DNN_CSV = './saved_files/pre_csv/lenet_mnist.csv'
DEFAULT_DATA_ROOT = '../../datasets'


def _load_top_neurons(csv_file: str, top_m: int = 4, strip_yolo_prefix: bool = False) -> Dict[str, list[int]]:
    df = pd.read_csv(csv_file)
    if 'Score' in df.columns:
        df = df.sort_values(by='Score', ascending=False)
    df = df.head(top_m)
    selected: Dict[str, list[int]] = {}
    for layer_name, group in df.groupby('LayerName'):
        key = str(layer_name)
        if strip_yolo_prefix and key.startswith('yolo_model.'):
            key = key.replace('yolo_model.', '', 1)
        selected[key] = group['NeuronIndex'].astype(int).tolist()
    if not selected:
        raise ValueError(f'No neurons were selected from {csv_file}.')
    return selected


def _subset_loader(dataset, count: int, batch_size: int) -> DataLoader:
    take = min(max(1, count), len(dataset))
    subset = Subset(dataset, list(range(take)))
    return DataLoader(subset, batch_size=min(batch_size, take), shuffle=False)


def _subset_loader_from_indices(dataset, indices: list[int], batch_size: int) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=min(batch_size, len(indices)), shuffle=False)


def _load_legacy_torch_model(model_path: str):
    resolved = resolve_saved_model_path(model_path)
    model_file = Path(resolved).resolve()
    repo_root = model_file.parents[2]
    added = False
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        added = True
    try:
        model, _, _ = get_model(str(model_file))
    finally:
        if added:
            sys.path.pop(0)
    return model


def _cluster_params_from_cfg(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    algo = str(cfg['algo'])
    algo_key = algo.lower()
    n_clusters = int(cfg['n_clusters'])
    params: Dict[str, Any] = {}
    if algo_key in {'kmeans', 'minibatchkmeans', 'agglomerativeclustering', 'birch'}:
        params['n_clusters'] = n_clusters
    if algo_key == 'kmeans':
        params.update({'random_state': seed, 'n_init': 10})
    elif algo_key == 'minibatchkmeans':
        params.update({'random_state': seed, 'batch_size': 16, 'max_iter': 100})
    elif algo_key == 'birch':
        params.update({'threshold': 0.5})
    return params


def _bo_search_space(max_build_samples: int) -> Dict[str, Any]:
    max_clusters = max(2, min(3, int(max_build_samples)))
    cluster_values = [2] if max_clusters <= 2 else [2, 3]
    return {
        'algo': ['KMeans', 'MiniBatchKMeans', 'Birch'],
        'n_clusters': cluster_values,
    }


def _make_bo_objective(
    model,
    build_loader: DataLoader,
    eval_loader: DataLoader,
    selected: Dict[str, list[int]],
    top_m: int,
    device: str,
    seed: int,
):
    def objective(cfg: Dict[str, Any]) -> float:
        params = _cluster_params_from_cfg(cfg, seed=seed)
        engine = WisdomIDC(
            model=model,
            impl='wisdom',
            cfg=WisdomConfig(top_m_neurons=top_m, test_all_classes=True, cache_path=None),
            cluster=ClusteringConfig(method=str(cfg['algo']), params=params, use_silhouette=False),
        )
        engine.fit_selected(build_loader, selected, device=device)
        coverage_rate, _total, max_coverage = engine.coverage(eval_loader, selected=selected, device=device)
        if max_coverage <= 0:
            return 0.0
        return float(coverage_rate / max_coverage)

    return objective


class _YOLOBOClassificationDataset(Dataset):
    def __init__(self, img_dir: str, max_images: int = 6, imgsz: int = 160):
        self.base = COCOImageDataset(img_dir, max_images=max_images, imgsz=imgsz)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx][0], 0


def run_yolo_score_smoke(
    weights: str,
    img_dir: str,
    out_csv: str,
    device: str = 'cuda:0',
    num_images: int = 2,
    batch_size: int = 1,
    imgsz: int = 320,
) -> str:
    return train_wisdom_yolo(
        weights=weights,
        img_dir=img_dir,
        out_csv=out_csv,
        batch_size=batch_size,
        num_images=num_images,
        top_m=4,
        methods=['lgxa'],
        voting_mode='coarse',
        device=device,
        imgsz=imgsz,
    )


def run_yolo_rq2_smoke(
    weights: str,
    img_dir: str,
    csv_file: str,
    out_prefix: str,
    device: str = 'cuda:0',
    num_images: int = 4,
    batch_size: int = 1,
    imgsz: int = 320,
) -> str:
    return run_rq2_opt(
        weights=weights,
        img_dir=img_dir,
        csv_file=csv_file,
        out_prefix=out_prefix,
        device=device,
        num_images=num_images,
        batch_size=batch_size,
        imgsz=imgsz,
        coverage_mode='cluster',
        importance='wisdom',
        neuron_select='per-layer',
        per_layer_k=2,
        n_clusters=3,
        num_iters=1,
    )


def run_dnn_bo_smoke(
    out_path: str,
    model_path: str = DEFAULT_DNN_MODEL,
    dataset: str = 'mnist',
    data_path: str = DEFAULT_DATA_ROOT,
    csv_file: str = DEFAULT_DNN_CSV,
    device: str = 'cpu',
    batch_size: int = 16,
    top_m: int = 4,
    build_samples: int = 64,
    eval_samples: int = 32,
    n_init: int = 2,
    n_iter: int = 2,
    backend: str = 'auto',
) -> str:
    _, _, train_dataset, test_dataset, _ = get_data(dataset, batch_size, data_path)
    build_loader = _subset_loader(train_dataset, build_samples, batch_size)
    eval_loader = _subset_loader(test_dataset, eval_samples, batch_size)
    model = _load_legacy_torch_model(model_path).to(device).eval()
    selected = _load_top_neurons(csv_file, top_m=top_m)
    search_space = _bo_search_space(len(build_loader.dataset))
    _result, out = run_bo(
        search_space,
        _make_bo_objective(model, build_loader, eval_loader, selected, top_m, device, seed=42),
        random_state=42,
        backend=backend,
        n_init=n_init,
        n_iter=n_iter,
        candidate_pool_size=64,
        out_path=out_path,
        payload_extras={
            'smoke_type': 'dnn-bo',
            'search_meta': {
                'dataset': dataset,
                'data_path': data_path,
                'model_path': str(resolve_saved_model_path(model_path)),
                'csv_file': csv_file,
                'build_samples': build_samples,
                'eval_samples': eval_samples,
                'top_m': top_m,
            },
        },
    )
    return out or out_path


def run_yolo_bo_smoke(
    out_path: str,
    weights: str,
    img_dir: str,
    csv_file: str,
    device: str = 'cpu',
    top_m: int = 4,
    build_samples: int = 4,
    eval_samples: int = 2,
    imgsz: int = 160,
    n_init: int = 2,
    n_iter: int = 2,
    backend: str = 'auto',
) -> str:
    from ultralytics import YOLO

    total_images = max(build_samples + eval_samples, 6)
    dataset = _YOLOBOClassificationDataset(img_dir, max_images=total_images, imgsz=imgsz)
    build_indices = list(range(min(build_samples, len(dataset))))
    eval_start = min(build_samples, len(dataset))
    eval_end = min(eval_start + eval_samples, len(dataset))
    eval_indices = list(range(eval_start, eval_end))
    if not eval_indices:
        eval_indices = build_indices

    build_loader = _subset_loader_from_indices(dataset, build_indices, batch_size=max(1, len(build_indices)))
    eval_loader = _subset_loader_from_indices(dataset, eval_indices, batch_size=max(1, len(eval_indices)))

    model = YOLO(weights).model.to(device).eval()
    selected = _load_top_neurons(csv_file, top_m=top_m, strip_yolo_prefix=True)
    search_space = _bo_search_space(len(build_indices))
    _result, out = run_bo(
        search_space,
        _make_bo_objective(model, build_loader, eval_loader, selected, top_m, device, seed=42),
        random_state=42,
        backend=backend,
        n_init=n_init,
        n_iter=n_iter,
        candidate_pool_size=64,
        out_path=out_path,
        payload_extras={
            'smoke_type': 'yolo-bo',
            'search_meta': {
                'weights': weights,
                'img_dir': img_dir,
                'csv_file': csv_file,
                'build_samples': build_samples,
                'eval_samples': eval_samples,
                'imgsz': imgsz,
                'top_m': top_m,
            },
        },
    )
    return out or out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Smoke helpers for the cleaned WISDOM run cases')
    parser.add_argument('--mode', choices=['score', 'rq2', 'dnn-bo', 'yolo-bo'], required=True)
    parser.add_argument('--weights', default='weights/yolo11n.pt')
    parser.add_argument('--img-dir', default='standalone/data/coco/images/val2017')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-images', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--out', required=True, help='Output CSV/path for smoke results.')
    parser.add_argument('--csv-file', default='', help='Existing WISDOM score CSV or DNN pre-score CSV.')
    parser.add_argument('--model-path', default=DEFAULT_DNN_MODEL)
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--data-path', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--top-m', type=int, default=4)
    parser.add_argument('--build-samples', type=int, default=64)
    parser.add_argument('--eval-samples', type=int, default=32)
    parser.add_argument('--bo-backend', choices=['auto', 'sklearn', 'botorch'], default='auto')
    parser.add_argument('--bo-init', type=int, default=2)
    parser.add_argument('--bo-iter', type=int, default=2)
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == 'score':
        return run_yolo_score_smoke(
            weights=args.weights,
            img_dir=args.img_dir,
            out_csv=args.out,
            device=args.device,
            num_images=args.num_images,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
        )
    if args.mode == 'rq2':
        if not args.csv_file:
            raise ValueError('--csv-file is required for --mode rq2')
        return run_yolo_rq2_smoke(
            weights=args.weights,
            img_dir=args.img_dir,
            csv_file=args.csv_file,
            out_prefix=args.out,
            device=args.device,
            num_images=args.num_images,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
        )
    if args.mode == 'dnn-bo':
        csv_file = args.csv_file or DEFAULT_DNN_CSV
        return run_dnn_bo_smoke(
            out_path=args.out,
            model_path=args.model_path,
            dataset=args.dataset,
            data_path=args.data_path,
            csv_file=csv_file,
            device=args.device,
            batch_size=max(1, args.batch_size),
            top_m=args.top_m,
            build_samples=args.build_samples,
            eval_samples=args.eval_samples,
            n_init=args.bo_init,
            n_iter=args.bo_iter,
            backend=args.bo_backend,
        )
    if not args.csv_file:
        raise ValueError('--csv-file is required for --mode yolo-bo')
    return run_yolo_bo_smoke(
        out_path=args.out,
        weights=args.weights,
        img_dir=args.img_dir,
        csv_file=args.csv_file,
        device=args.device,
        top_m=args.top_m,
        build_samples=max(2, min(args.build_samples, args.num_images or args.build_samples)),
        eval_samples=max(1, args.eval_samples),
        imgsz=args.imgsz,
        n_init=args.bo_init,
        n_iter=args.bo_iter,
        backend=args.bo_backend,
    )


if __name__ == '__main__':
    main()

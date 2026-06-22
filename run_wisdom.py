"""Unified entrypoint for WISDOM and vanilla IDC coverage runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from run_cases.run_rq1 import COCOLabeledDataset, _collate_labeled, _nms_predictions
from run_cases.run_rq2 import (
    perturb_important_pixels,
    pixel_importance_gradient,
    pixel_importance_wisdom,
)
from run_cases.support import get_data, resolve_saved_model_path
from wisdom import (
    ConsensusWisdom,
    COCOImageDataset,
    ClusteringConfig,
    run_bo as run_search_bo,
    train_wisdom_yolo,
    WisdomConfig,
    WisdomIDC,
    WisdomTrainConfig,
    get_group_names,
    load_groupwise_top_neurons,
    load_layerwise_top_neurons,
)
from wisdom.attribution.captum_backend import ATTRS
from wisdom.clustering.assign import assign_clusters
from wisdom.core.activation import collect_per_neuron_once
from wisdom.utils.common import eval_model_dataloder, get_model, get_trainable_modules_main
from wisdom.utils.detection_loader import detect_head_prefixes, load_detection_model
from wisdom.utils.io_cache import read_layer_scores_csv
from wisdom.utils.visulization import viz_attr, viz_attr_diff, viz_topk_neurons_score
from wisdom.utils.yolo_wrapper import YOLOWrapper


ATTRIBUTION_METHODS = sorted(ATTRS)
DEFAULT_WISDOM_METHODS = list(ATTRIBUTION_METHODS)


class DetectionClassificationDataset(Dataset):
    """Wrap raw detection images as classification-style ``(image, label)`` pairs."""

    def __init__(self, img_dir: str, max_images: int | None, imgsz: int):
        self.base = COCOImageDataset(img_dir, max_images=max_images, imgsz=imgsz)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        return self.base[idx][0], 0


def _default_device() -> str:
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def _parse_csv_list(raw: str | None, cast=str) -> list[Any]:
    if not raw:
        return []
    return [cast(item.strip()) for item in raw.split(',') if item.strip()]


def _subset_loader(dataset, count: int | None, batch_size: int, collate_fn=None) -> DataLoader:
    total = len(dataset)
    if total <= 0:
        raise ValueError('Dataset must contain at least one sample.')
    take = total if count is None else min(max(1, count), total)
    subset = dataset if take == total else Subset(dataset, list(range(take)))
    return DataLoader(subset, batch_size=min(batch_size, take), shuffle=False, collate_fn=collate_fn)


def _subset_from_loader(loader: DataLoader, count: int) -> DataLoader:
    batch_size = loader.batch_size or count
    return _subset_loader(loader.dataset, count, batch_size, collate_fn=getattr(loader, 'collate_fn', None))


def _loader_sample_count(loader: DataLoader, requested: int | None = None) -> int | None:
    dataset = getattr(loader, 'dataset', None)
    if dataset is not None:
        try:
            return len(dataset)
        except TypeError:
            pass
    return requested


def _load_legacy_torch_model(model_path: str):
    resolved = resolve_saved_model_path(model_path)
    model_file = Path(resolved).resolve()
    repo_root = model_file.parents[2] if len(model_file.parents) > 2 else Path.cwd()
    added = False
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        added = True
    try:
        model, _, _ = get_model(str(model_file))
    finally:
        if added:
            sys.path.pop(0)
    return model.eval(), str(model_file)


def _normalize_layer_scores(layer_scores: dict[str, torch.Tensor], strip_prefix: str | None = None):
    normalized: dict[str, torch.Tensor] = {}
    for layer_name, scores in layer_scores.items():
        key = layer_name
        if strip_prefix and key.startswith(strip_prefix):
            key = key[len(strip_prefix):]
        normalized[key] = scores
    return normalized


def _filter_layer_scores(layer_scores: dict[str, torch.Tensor], excluded_prefixes: list[str]):
    return {
        layer_name: scores
        for layer_name, scores in layer_scores.items()
        if not any(layer_name.startswith(prefix) for prefix in excluded_prefixes)
    }


def _cluster_params(cluster_method: str, n_clusters: int, seed: int) -> dict[str, Any]:
    key = cluster_method.lower()
    params: dict[str, Any] = {}
    if key in {'kmeans', 'minibatchkmeans', 'birch', 'agglomerativeclustering'}:
        params['n_clusters'] = int(n_clusters)
    if key == 'kmeans':
        params.update({'random_state': seed, 'n_init': 10})
    elif key == 'minibatchkmeans':
        params.update({'random_state': seed, 'batch_size': 32, 'max_iter': 100})
    elif key == 'birch':
        params.setdefault('threshold', 0.5)
    return params


def _infer_task(args) -> str:
    if args.task != 'auto':
        return args.task
    if args.img_dir or str(args.model_path).endswith('.pt') or 'yolo' in Path(args.model_path).name.lower():
        return 'detection'
    return 'classification'


def _default_single_method(task: str) -> str:
    return 'lgxa' if task == 'detection' else 'lrp'


def _resolve_methods(args, task: str) -> list[str]:
    if args.methods:
        methods = list(args.methods)
    elif args.attribution_method:
        methods = [args.attribution_method]
    elif args.impl == 'idc':
        methods = [_default_single_method(task)]
    else:
        methods = list(DEFAULT_WISDOM_METHODS)

    if args.impl == 'idc' and len(methods) != 1:
        raise ValueError('IDC mode expects exactly one attribution method. Use --attribution-method or one --methods value.')
    return methods


def _resolve_testing_mode(args) -> dict[str, bool]:
    class_iters = bool(getattr(args, 'class_iters', False))
    all_class = bool(getattr(args, 'all_class', True))
    if class_iters:
        all_class = False
    return {
        'all_class': all_class,
        'class_iters': class_iters,
    }


def _resolve_selection_mode(args) -> dict[str, Any]:
    single_layer = getattr(args, 'single_layer', None)
    if single_layer:
        return {
            'mode': 'single-layer',
            'n_groups': None,
            'label': f'Single-Layer ({single_layer})',
            'aggregation_label': None,
            'layer_name': str(single_layer),
        }
    if getattr(args, 'per_layer', False):
        return {
            'mode': 'per-layer',
            'n_groups': None,
            'label': 'Per-Layer',
            'aggregation_label': 'Average across layers',
        }
    per_group = getattr(args, 'per_group', None)
    if per_group is not None:
        if int(per_group) <= 0:
            raise ValueError('--per-group must be a positive integer.')
        return {
            'mode': 'per-group',
            'n_groups': int(per_group),
            'label': f'Per-Group ({int(per_group)})',
            'aggregation_label': f'Average across {int(per_group)} groups',
        }
    return {
        'mode': 'global',
        'n_groups': None,
        'label': 'Global',
        'aggregation_label': None,
    }


def _selection_strip_prefix(task: str) -> str:
    return 'yolo_model.' if task == 'detection' else ''


def _filter_selected_layers(prepared, selected: dict[str, list[int]]) -> dict[str, list[int]]:
    allowed_layers = set(prepared['layer_scores'].keys())
    if prepared.get('exclude_last'):
        allowed_layers.discard(prepared['exclude_last'])
    return {
        layer_name: sorted(dict.fromkeys(int(index) for index in indices))
        for layer_name, indices in selected.items()
        if layer_name in allowed_layers and indices
    }


def _resolve_requested_layer_name(prepared, requested: str) -> str:
    raw = str(requested)
    normalized = raw
    if prepared.get('task') == 'detection' and raw.startswith('yolo_model.'):
        normalized = raw[len('yolo_model.'):]
    available_layers = set(prepared['layer_scores'].keys())
    if prepared.get('exclude_last'):
        available_layers.discard(prepared['exclude_last'])
    if normalized in available_layers:
        return normalized
    if raw in available_layers:
        return raw
    available_preview = ', '.join(sorted(available_layers)[:10])
    raise ValueError(
        f'Layer {requested!r} was not found in the selected layer scores. '
        f'Available layers include: {available_preview}'
    )


def _select_top_neurons_from_layer(scores: torch.Tensor, top_k: int) -> list[int]:
    if scores.dim() == 1:
        flattened = scores.detach().cpu()
    else:
        flattened = scores.mean(dim=tuple(range(1, scores.dim()))).detach().cpu()
    if flattened.numel() == 0:
        return []
    take = flattened.numel() if top_k <= 0 else min(int(top_k), int(flattened.numel()))
    indices = torch.topk(flattened, k=take).indices.tolist()
    return sorted(int(index) for index in indices)


def _resolve_selected_neurons(args, prepared, selector_engine: WisdomIDC) -> dict[str, list[int]]:
    selection_mode = _resolve_selection_mode(args)
    if selection_mode['mode'] == 'global':
        return selector_engine.select_top_neurons_all(prepared['layer_scores'], exclude_last=prepared['exclude_last'])
    if selection_mode['mode'] == 'single-layer':
        layer_name = _resolve_requested_layer_name(prepared, selection_mode['layer_name'])
        selected = {
            layer_name: _select_top_neurons_from_layer(
                prepared['layer_scores'][layer_name],
                args.top_m_neurons,
            )
        }
        filtered = _filter_selected_layers(prepared, selected)
        if not filtered:
            raise ValueError(f'No neurons were selected for {selection_mode["label"]} coverage.')
        return filtered

    if not prepared.get('csv_path'):
        raise ValueError('Per-group and per-layer coverage require a resolved neuron-score CSV.')

    loader_kwargs = {
        'csv_path': prepared['csv_path'],
        'strip_prefix': _selection_strip_prefix(prepared['task']),
    }
    if selection_mode['mode'] == 'per-group':
        selected = load_groupwise_top_neurons(
            per_group_k=args.top_m_neurons,
            n_groups=selection_mode['n_groups'] or 3,
            **loader_kwargs,
        )
    else:
        selected = load_layerwise_top_neurons(
            per_layer_k=args.top_m_neurons,
            **loader_kwargs,
        )

    filtered = _filter_selected_layers(prepared, selected)
    if not filtered:
        raise ValueError(f'No neurons were selected for {selection_mode["label"]} coverage.')
    return filtered


def _build_coverage_breakdown_log(
    suite_rows: list[dict[str, Any]],
    selection_label: str,
    aggregation_label: str | None,
) -> str:
    scope_rows = [row for row in suite_rows if row.get('row_type') == 'scope']
    if not scope_rows:
        return ''

    lines = ['=' * 88, f'Coverage Breakdown ({selection_label})', '=' * 88]
    if aggregation_label:
        lines.append(f'Aggregation: {aggregation_label}')
        lines.append('-' * 88)

    current_suite = None
    for row in scope_rows:
        suite_name = f'{row["suite_label"]} [{row["suite_name"]}]'
        if suite_name != current_suite:
            if current_suite is not None:
                lines.append('-' * 88)
            current_suite = suite_name
            lines.append(f'Suite: {suite_name}')
        lines.append(
            f'  {row["scope_name"]:<24} '
            f'coverage={row["coverage_rate"]:.6f} '
            f'combos={int(row["total_combinations"])} '
            f'max={row["max_coverage"]:.6f}'
        )
    lines.append('=' * 88)
    return '\n'.join(lines)


def _build_suite_aggregate_log(suite_rows: list[dict[str, Any]], testing_mode_label: str) -> str:
    aggregate_rows = [row for row in suite_rows if row.get('row_type', 'aggregate') == 'aggregate']
    if not aggregate_rows:
        return ''

    title = 'Class Coverage Scores' if testing_mode_label == 'Class-Iters' else 'Suite Coverage Scores'
    lines = ['=' * 88, title, '=' * 88]
    for row in aggregate_rows:
        suite_display = row.get('suite_name') if testing_mode_label == 'Class-Iters' else row.get('suite_label')
        if not suite_display:
            suite_display = row.get('suite_label') or row.get('suite_name') or f"size:{row.get('suite_size', 'unknown')}"
        lines.append(
            f'{suite_display:<24} '
            f'coverage={row["coverage_rate"]:.6f} '
            f'combos={int(row["total_combinations"])} '
            f'max={row["max_coverage"]:.6f} '
            f'f1={row["f1_score"]:.6f}'
        )
    lines.append('=' * 88)
    return '\n'.join(lines)


def _build_cluster_config(args, cluster_method: str | None = None, n_clusters: int | None = None) -> ClusteringConfig:
    effective_clusters = int(n_clusters or args.n_clusters)
    if args.impl == 'idc':
        return ClusteringConfig(
            method='KMeans',
            params={'n_clusters': effective_clusters, 'random_state': args.seed, 'n_init': 10},
            use_silhouette=True,
        )
    method = cluster_method or args.cluster_method
    return ClusteringConfig(
        method=method,
        params=_cluster_params(method, effective_clusters, args.seed),
        use_silhouette=False,
    )


def _build_engine(args, model, cluster_method: str | None = None, n_clusters: int | None = None) -> WisdomIDC:
    testing_mode = _resolve_testing_mode(args)
    selection_mode = _resolve_selection_mode(args)
    return WisdomIDC(
        model=model,
        impl=args.impl,
        cfg=WisdomConfig(
            top_m_neurons=args.top_m_neurons,
            test_all_classes=testing_mode['all_class'],
            cache_path=args.cache_path,
            selection_mode=selection_mode['mode'],
            n_groups=selection_mode['n_groups'] or 3,
        ),
        cluster=_build_cluster_config(args, cluster_method=cluster_method, n_clusters=n_clusters),
    )


def _loader_from_indices(dataset, indices: list[int], batch_size: int, collate_fn=None) -> DataLoader:
    if not indices:
        raise ValueError('Class-wise evaluation requires at least one sample.')
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=min(batch_size, len(indices)), shuffle=False, collate_fn=collate_fn)


def _normalize_class_names(raw_names) -> list[str]:
    if raw_names is None:
        return []
    if isinstance(raw_names, dict):
        return [str(raw_names[idx]) for idx in sorted(raw_names)]
    return [str(name) for name in raw_names]


def _extract_class_ids(label) -> list[int]:
    if isinstance(label, torch.Tensor):
        if label.numel() == 0:
            return []
        if label.dim() == 0:
            return [int(label.item())]
        return sorted({int(item) for item in label.reshape(-1).tolist()})
    if isinstance(label, (int, np.integer)):
        return [int(label)]
    if isinstance(label, list):
        if not label:
            return []
        if isinstance(label[0], tuple):
            return sorted({int(item[0]) for item in label if item})
        return sorted({int(item) for item in label})
    if isinstance(label, tuple):
        return _extract_class_ids(list(label))
    return []


def _class_iter_loaders(prepared, batch_size: int) -> list[tuple[int, str, DataLoader]]:
    eval_loader = prepared.get('eval_loader')
    dataset = getattr(eval_loader, 'dataset', None) or prepared['eval_dataset']
    class_names = prepared.get('classes', [])
    collate_fn = prepared.get('eval_collate_fn')
    index_map: dict[int, list[int]] = {}

    for idx in range(len(dataset)):
        _sample, label = dataset[idx]
        for class_id in _extract_class_ids(label):
            if class_id < 0:
                continue
            index_map.setdefault(class_id, []).append(idx)

    loaders: list[tuple[int, str, DataLoader]] = []
    for class_id in sorted(index_map):
        name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        loaders.append((class_id, name, _loader_from_indices(dataset, index_map[class_id], batch_size, collate_fn=collate_fn)))
    return loaders


def _prepare_classification(args, csv_path: str | None):
    if not args.dataset:
        raise ValueError('--dataset is required for classification runs.')
    if not args.data_path:
        raise ValueError('--data-path is required for classification runs.')

    model, resolved_model_path = _load_legacy_torch_model(args.model_path)
    _, _, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path)
    build_loader = _subset_loader(train_dataset, args.build_samples, args.batch_size)
    eval_loader = _subset_loader(test_dataset, args.eval_samples, args.batch_size)
    _, trainable_names = get_trainable_modules_main(model)
    final_layer = trainable_names[-1] if trainable_names else None

    if csv_path is None:
        if not args.pretrain:
            raise ValueError('Provide --csv-file or enable --pretrain for classification runs.')
        out_csv = args.csv_file or str(Path(args.out_dir) / f'{args.run_name}_scores.csv')
        trainer = ConsensusWisdom(model, device=args.device)
        cfg = WisdomTrainConfig(
            methods=_resolve_methods(args, 'classification'),
            device=args.device,
            voting_mode=args.voting_mode,
            out_csv=out_csv,
        )
        layer_scores, csv_path = trainer.fit(
            build_loader,
            cfg,
            top_m_neurons=args.top_m_neurons,
            final_layer=final_layer,
            prune_mode='mask',
        )
    else:
        layer_scores = read_layer_scores_csv(csv_path)

    return {
        'task': 'classification',
        'model': model.eval().to(args.device),
        'model_path': resolved_model_path,
        'dataset_name': args.dataset,
        'classes': _normalize_class_names(classes),
        'build_dataset': train_dataset,
        'eval_dataset': test_dataset,
        'build_collate_fn': None,
        'eval_collate_fn': None,
        'build_loader': build_loader,
        'eval_loader': eval_loader,
        'layer_scores': layer_scores,
        'exclude_last': final_layer,
        'csv_path': csv_path,
    }


def _prepare_detection(args, csv_path: str | None):
    if not args.img_dir:
        raise ValueError('--img-dir is required for detection runs.')

    bundle = load_detection_model(args.model_path, device=args.device)
    dataset = COCOLabeledDataset(
        args.img_dir,
        max_images=None,
        imgsz=args.imgsz,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f'No images found in {args.img_dir}')

    build_loader = _subset_loader(dataset, args.build_samples, args.batch_size, collate_fn=_collate_labeled)
    eval_loader = _subset_loader(dataset, args.eval_samples, args.batch_size, collate_fn=_collate_labeled)

    if csv_path is None:
        if not args.pretrain:
            raise ValueError('Provide --csv-file or enable --pretrain for detection runs.')
        out_csv = args.csv_file or str(Path(args.out_dir) / f'{args.run_name}_scores.csv')
        csv_path = train_wisdom_yolo(
            weights=args.model_path,
            img_dir=args.img_dir,
            out_csv=out_csv,
            batch_size=args.batch_size,
            num_images=_loader_sample_count(build_loader, args.build_samples) or len(dataset),
            top_m=args.top_m_neurons,
            methods=_resolve_methods(args, 'detection'),
            voting_mode=args.voting_mode,
            selection_mode='global',
            device=args.device,
            imgsz=args.imgsz,
        )

    layer_scores = _normalize_layer_scores(read_layer_scores_csv(csv_path), strip_prefix='yolo_model.')
    excluded_prefixes = detect_head_prefixes(bundle.model, wrapper_prefix='')
    layer_scores = _filter_layer_scores(layer_scores, excluded_prefixes)

    return {
        'task': 'detection',
        'model': bundle.model.eval().to(args.device),
        'model_path': args.model_path,
        'dataset_name': args.img_dir,
        'classes': _normalize_class_names(bundle.names),
        'build_dataset': dataset,
        'eval_dataset': dataset,
        'build_collate_fn': _collate_labeled,
        'eval_collate_fn': _collate_labeled,
        'build_loader': build_loader,
        'eval_loader': eval_loader,
        'layer_scores': layer_scores,
        'exclude_last': None,
        'csv_path': csv_path,
        'num_classes': bundle.num_classes,
    }


def _candidate_n_clusters(args, build_size: int) -> list[int]:
    values = _parse_csv_list(args.bo_n_clusters, int) or [2, 3, 4]
    candidates = sorted({value for value in values if 2 <= value <= build_size})
    if not candidates:
        raise ValueError(f'No valid BO n_clusters candidates for build size {build_size}.')
    return candidates


def _suite_sizes(total_size: int, points: int) -> list[int]:
    if total_size <= 0:
        raise ValueError('Evaluation dataset must contain at least one sample.')
    if total_size == 1:
        return [1]
    steps = max(2, min(points, total_size))
    sizes = {max(1, int(round(v))) for v in np.linspace(1, total_size, num=steps)}
    sizes.add(total_size)
    return sorted(sizes)


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def _eval_detection_f1(model, loader: DataLoader, device: str, imgsz: int, conf_thresh: float = 0.25, iou_thresh: float = 0.1) -> float:
    model.eval().to(device)
    true_positive = 0
    total_pred = 0
    total_gt = 0

    with torch.no_grad():
        for images, gt_labels_batch in loader:
            raw_out = model(images.to(device))
            raw_preds = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
            preds_batch = _nms_predictions(raw_preds, conf_thresh=conf_thresh, imgsz=imgsz)

            for preds, gt_boxes in zip(preds_batch, gt_labels_batch):
                total_pred += len(preds)
                total_gt += len(gt_boxes)
                matched_preds = [False] * len(preds)

                for gt_cls, gt_cx, gt_cy, gt_w, gt_h in gt_boxes:
                    best_iou = 0.0
                    best_idx = -1
                    gt_x1 = gt_cx - gt_w / 2
                    gt_y1 = gt_cy - gt_h / 2
                    gt_x2 = gt_cx + gt_w / 2
                    gt_y2 = gt_cy + gt_h / 2

                    for pred_idx, (pred_cls, pred_cx, pred_cy, pred_w, pred_h, _pred_conf) in enumerate(preds):
                        if matched_preds[pred_idx] or pred_cls != gt_cls:
                            continue
                        pred_x1 = pred_cx - pred_w / 2
                        pred_y1 = pred_cy - pred_h / 2
                        pred_x2 = pred_cx + pred_w / 2
                        pred_y2 = pred_cy + pred_h / 2
                        inter_x1 = max(gt_x1, pred_x1)
                        inter_y1 = max(gt_y1, pred_y1)
                        inter_x2 = min(gt_x2, pred_x2)
                        inter_y2 = min(gt_y2, pred_y2)
                        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
                        gt_area = max(0.0, gt_x2 - gt_x1) * max(0.0, gt_y2 - gt_y1)
                        pred_area = max(0.0, pred_x2 - pred_x1) * max(0.0, pred_y2 - pred_y1)
                        union = gt_area + pred_area - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = pred_idx

                    if best_idx >= 0 and best_iou >= iou_thresh:
                        matched_preds[best_idx] = True
                        true_positive += 1

    precision = true_positive / total_pred if total_pred > 0 else 0.0
    recall = true_positive / total_gt if total_gt > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _compute_correlation_workflow(args, prepared, engine: WisdomIDC, selected: dict[str, list[int]]):
    rows = []
    coverage_scores: list[float] = []
    f1_scores: list[float] = []
    testing_mode = _resolve_testing_mode(args)

    if testing_mode['class_iters']:
        suites = [
            {
                'suite_label': f'class:{class_id}',
                'suite_name': class_name,
                'suite_size': len(loader.dataset),
                'loader': loader,
            }
            for class_id, class_name, loader in _class_iter_loaders(prepared, args.batch_size)
        ]
        if not suites:
            raise ValueError('No class-specific samples found for --class-iters evaluation.')
    else:
        suites = [
            {
                'suite_label': f'size:{suite_size}',
                'suite_name': 'all-class',
                'suite_size': suite_size,
                'loader': _subset_from_loader(prepared['eval_loader'], suite_size),
            }
            for suite_size in _suite_sizes(len(prepared['eval_loader'].dataset), args.corr_points)
        ]

    for suite in suites:
        suite_loader = suite['loader']
        coverage_result = engine.coverage_details(
            suite_loader,
            selected=selected,
            device=args.device,
        )
        coverage_rate = float(coverage_result['coverage_rate'])
        total_combinations = int(coverage_result['total_combinations'])
        max_coverage = float(coverage_result['max_coverage'])
        if prepared['task'] == 'classification':
            _accuracy, _loss, f1_score = eval_model_dataloder(prepared['model'], suite_loader, device=args.device)
        else:
            f1_score = _eval_detection_f1(prepared['model'], suite_loader, device=args.device, imgsz=args.imgsz)

        rows.append(
            {
                'suite_label': suite['suite_label'],
                'suite_name': suite['suite_name'],
                'suite_size': suite['suite_size'],
                'testing_mode': 'class-iters' if testing_mode['class_iters'] else 'all-class',
                'row_type': 'aggregate',
                'scope_name': 'overall',
                'coverage_rate': coverage_rate,
                'max_coverage': max_coverage,
                'total_combinations': total_combinations,
                'f1_score': f1_score,
            }
        )
        coverage_scores.append(float(coverage_rate))
        f1_scores.append(float(f1_score))
        scope_details = coverage_result.get('scope_details', {})
        if _resolve_selection_mode(args)['mode'] != 'global':
            for scope_name, detail in scope_details.items():
                rows.append(
                    {
                        'suite_label': suite['suite_label'],
                        'suite_name': suite['suite_name'],
                        'suite_size': suite['suite_size'],
                        'testing_mode': 'class-iters' if testing_mode['class_iters'] else 'all-class',
                        'row_type': 'scope',
                        'scope_name': scope_name,
                        'coverage_rate': float(detail['coverage_rate']),
                        'max_coverage': float(detail['max_coverage']),
                        'total_combinations': int(detail['total_combinations']),
                        'f1_score': f1_score,
                    }
                )

    correlation = _pearson_correlation(coverage_scores, f1_scores)
    return rows, correlation


def _coverage_objective(args, *, model, build_loader, eval_loader, selected):
    def objective(cfg: dict[str, Any]) -> float:
        engine = _build_engine(
            args,
            model,
            cluster_method=str(cfg['cluster_method']),
            n_clusters=int(cfg['n_clusters']),
        )
        engine.fit_selected(build_loader, selected, device=args.device)
        prepared = {
            'task': args.task if args.task != 'auto' else ('detection' if args.img_dir else 'classification'),
            'eval_loader': eval_loader,
            'model': model,
        }
        _rows, correlation = _compute_correlation_workflow(args, prepared, engine, selected)
        return correlation

    return objective


def _summarize_aggregate_rows(aggregate_rows: list[dict[str, Any]], *, class_iters: bool) -> dict[str, float | int]:
    if not aggregate_rows:
        raise ValueError('No aggregate suite rows were produced.')
    if not class_iters:
        final_row = aggregate_rows[-1]
        return {
            'coverage_rate': float(final_row['coverage_rate']),
            'total_combinations': int(final_row['total_combinations']),
            'max_coverage': float(final_row['max_coverage']),
            'f1_score': float(final_row['f1_score']),
        }

    return {
        'coverage_rate': float(np.mean([row['coverage_rate'] for row in aggregate_rows])),
        'total_combinations': int(round(float(np.mean([row['total_combinations'] for row in aggregate_rows])))),
        'max_coverage': float(np.mean([row['max_coverage'] for row in aggregate_rows])),
        'f1_score': float(np.mean([row['f1_score'] for row in aggregate_rows])),
    }


def _write_combo_log(engine: WisdomIDC, model, eval_loader: DataLoader, selected: dict[str, list[int]], device: str, out_path: Path):
    seen = Counter()
    rows = []
    sample_index = 0
    for images, _ in eval_loader:
        for offset in range(images.size(0)):
            acts = collect_per_neuron_once(model, images[offset:offset + 1], selected, device=device)
            assigned = assign_clusters(engine.groups, acts)
            flat = {f'{layer}:{idx}': cluster for layer, idx_map in assigned.items() for idx, cluster in idx_map.items()}
            combo_key = '|'.join(f'{key}={flat[key]}' for key in sorted(flat))
            seen[combo_key] += 1
            rows.append({'sample_index': sample_index, 'combo_key': combo_key, 'assignments': flat})
            sample_index += 1

    with out_path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')

    counts_path = out_path.with_name(out_path.stem + '_counts.json')
    counts_path.write_text(json.dumps(dict(seen), indent=2))
    return str(out_path), str(counts_path), len(seen)


def _move_generated_plot(src_name: str, dest_path: Path):
    src = Path(src_name)
    if src.exists():
        src.replace(dest_path)
        return str(dest_path)
    return None


def _write_neuron_plot(csv_path: str, top_k: int, out_dir: Path, run_name: str):
    viz_topk_neurons_score(csv_path, top_k=top_k)
    return _move_generated_plot(f'top_{top_k}_neuron_scores.pdf', out_dir / f'{run_name}_top_{top_k}_neurons.pdf')


def _write_pixel_plots(args, prepared, out_dir: Path):
    if prepared['task'] != 'detection':
        raise ValueError('--plot-pixels is only supported for detection runs.')

    wrapper = YOLOWrapper(prepared['model'], num_classes=prepared['num_classes']).eval().to(args.device)
    images, _ = next(iter(prepared['eval_loader']))
    sample = images[:1]
    if args.impl == 'wisdom':
        importance = pixel_importance_wisdom(prepared['csv_path'], wrapper, sample, args.device)
        tag = 'wisdom'
    else:
        importance = pixel_importance_gradient(wrapper, sample, args.device)
        tag = _default_single_method('detection')

    attr = importance[0].unsqueeze(0).repeat(sample.size(1), 1, 1)
    viz_attr(sample[0], attr, prepared['task'], args.run_name, with_original=True)
    heatmap_path = _move_generated_plot(
        f'{prepared["task"]}_{args.run_name}.png',
        out_dir / f'{args.run_name}_{tag}_pixels.png',
    )

    perturbed = perturb_important_pixels(sample, importance, frac=args.pixel_frac, std=args.noise_std)
    viz_attr_diff(sample[0], perturbed[0], tag=tag)
    diff_path = _move_generated_plot(
        f'feature_importance_{tag}_diff.pdf',
        out_dir / f'{args.run_name}_{tag}_pixels_diff.pdf',
    )
    return {'pixel_heatmap': heatmap_path, 'pixel_diff': diff_path}


def _format_terminal_summary(summary: dict[str, Any], *, summary_csv: Path, summary_json: Path) -> str:
    attribution = summary['attribution']
    rows = [
        ('Model Name', summary['model_name']),
        ('Dataset', summary['dataset']),
        ('Testing Mode', summary['testing_mode_label']),
        ('Selection', summary['selection_mode_label']),
        ('Top-k', summary['top_m_neurons']),
        ('Attribution', attribution),
        ('Total Combination', summary['total_combinations']),
        ('Max Coverage', f"{summary['max_coverage']:.6f}"),
        ('Coverage Rate/Score', f"{summary['coverage_rate']:.6f}"),
        ('Summary CSV', str(summary_csv)),
        ('Summary JSON', str(summary_json)),
    ]
    if summary.get('suite_metrics_csv'):
        rows.append(('Suite Metrics CSV', summary['suite_metrics_csv']))
    if summary.get('consensus_methods'):
        rows.append(('Consensus Methods', ', '.join(summary['consensus_methods'])))
    if summary.get('pretrain_voting_mode'):
        rows.append(('Voting Mode', summary['pretrain_voting_mode']))
    if summary.get('coverage_aggregation'):
        rows.append(('Coverage Aggregation', summary['coverage_aggregation']))
    if summary.get('suite_aggregation'):
        rows.append(('Suite Aggregation', summary['suite_aggregation']))
    if summary.get('coverage_breakdown_log'):
        rows.append(('Coverage Breakdown Log', summary['coverage_breakdown_log']))
    if summary.get('suite_coverage_log'):
        rows.append(('Suite Coverage Log', summary['suite_coverage_log']))
    if summary.get('bo_result'):
        rows.append(('BO Result', summary['bo_result']))

    width = max(len(label) for label, _ in rows)
    border = '=' * 88
    lines = [border, f"{summary['impl'].upper()} Testing Summary", border]
    lines.extend(f"{label:<{width}} : {value}" for label, value in rows)
    lines.append(border)
    return '\n'.join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run WISDOM or vanilla IDC coverage from one entrypoint.')
    parser.add_argument('--impl', choices=['idc', 'wisdom'], default='wisdom')
    parser.add_argument('--mode', dest='impl', choices=['idc', 'wisdom'], help=argparse.SUPPRESS)
    parser.add_argument('--task', choices=['auto', 'classification', 'detection'], default='auto')
    parser.add_argument('--model-path', required=True, help='Pretrained classification model or detection weights.')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'], default=None)
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--img-dir', default=None)
    parser.add_argument('--csv-file', default=None, help='Pretrained neuron-score CSV to reuse.')
    parser.add_argument('--pretrain', action='store_true', help='Generate a neuron-score CSV before running coverage.')
    parser.add_argument('--methods', nargs='+', choices=ATTRIBUTION_METHODS, default=None, help='Attribution methods used during pretraining.')
    parser.add_argument('--attribution-method', choices=ATTRIBUTION_METHODS, default=None, help='Single attribution method for IDC or single-method pretraining.')
    parser.add_argument('--voting-mode', choices=['fine-grained', 'coarse'], default='fine-grained', help='Consensus voting mode used when pretraining WISDOM scores.')
    parser.add_argument('--device', default=_default_device())
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--build-samples', type=int, default=None, help='Optional build-data subset size. Omit to use the full build dataset.')
    parser.add_argument('--eval-samples', type=int, default=None, help='Optional evaluation-data subset size. Omit to use the full evaluation dataset.')
    parser.add_argument('--top-m-neurons', type=int, default=10)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument('--single-layer', default=None, help='Compute coverage for one chosen layer name using the top-k neurons within that layer.')
    selection_group.add_argument('--per-group', nargs='?', const=3, type=int, default=None, help='Average coverage across N contiguous layer groups (default: 3 when provided without a value).')
    selection_group.add_argument('--per-layer', action='store_true', help='Average coverage across layers, with top-k neurons selected separately per layer.')
    parser.add_argument('--cluster-method', default='KMeans')
    parser.add_argument('--n-clusters', type=int, default=2)
    parser.add_argument('--cache-path', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--noise-std', type=float, default=0.3)
    parser.add_argument('--pixel-frac', type=float, default=0.02)
    parser.add_argument('--bo', action='store_true', help='Tune clustering hyperparameters before the final run.')
    parser.add_argument('--bo-backend', choices=['auto', 'sklearn', 'botorch'], default='auto')
    parser.add_argument('--bo-init', type=int, default=3)
    parser.add_argument('--bo-iter', type=int, default=3)
    parser.add_argument('--bo-cluster-methods', default='KMeans,MiniBatchKMeans,Birch')
    parser.add_argument('--bo-n-clusters', default='2,3,4')
    parser.add_argument('--corr-points', type=int, default=5, help='Number of evaluation suite sizes used for F1/coverage correlation.')
    parser.add_argument('--all-class', action='store_true', default=True, help='Evaluate the full evaluation set across all classes.')
    parser.add_argument('--class-iters', action='store_true', help='Iterate over each class separately and compute coverage/F1 correlation across classes.')
    parser.add_argument('--combo-log', action='store_true', help='Write per-sample activated combinations.')
    parser.add_argument('--plot-neurons', action='store_true')
    parser.add_argument('--plot-pixels', action='store_true')
    parser.add_argument('--out-dir', default='results/run_wisdom')
    parser.add_argument('--run-name', default='run_wisdom')
    return parser


def run(args) -> dict[str, Any]:
    task = _infer_task(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_mode = _resolve_selection_mode(args)

    csv_path = args.csv_file if args.csv_file and Path(args.csv_file).is_file() else None
    prepared = _prepare_detection(args, csv_path) if task == 'detection' else _prepare_classification(args, csv_path)

    engine_for_selection = _build_engine(args, prepared['model'])
    selected = _resolve_selected_neurons(args, prepared, engine_for_selection)

    best_cluster_method = 'KMeans' if args.impl == 'idc' else args.cluster_method
    best_n_clusters = args.n_clusters
    bo_path = None
    if args.bo:
        search_space = {
            'cluster_method': ['KMeans'] if args.impl == 'idc' else (_parse_csv_list(args.bo_cluster_methods, str) or ['KMeans', 'MiniBatchKMeans', 'Birch']),
            'n_clusters': _candidate_n_clusters(args, len(prepared['build_loader'].dataset)),
        }
        result, bo_path = run_search_bo(
            search_space,
            _coverage_objective(
                args,
                model=prepared['model'],
                build_loader=prepared['build_loader'],
                eval_loader=prepared['eval_loader'],
                selected=selected,
            ),
            random_state=args.seed,
            backend=args.bo_backend,
            n_init=args.bo_init,
            n_iter=args.bo_iter,
            candidate_pool_size=32,
            out_path=out_dir / f'{args.run_name}_bo.json',
        )
        best_config = result.best_config
        best_cluster_method = str(best_config['cluster_method'])
        best_n_clusters = int(best_config['n_clusters'])

    engine = _build_engine(args, prepared['model'], cluster_method=best_cluster_method, n_clusters=best_n_clusters)
    selected = engine.fit_selected(prepared['build_loader'], selected, device=args.device)
    suite_rows, pearson_correlation = _compute_correlation_workflow(args, prepared, engine, selected)
    aggregate_rows = [row for row in suite_rows if row.get('row_type', 'aggregate') == 'aggregate']
    build_samples_used = _loader_sample_count(prepared['build_loader'], args.build_samples)
    eval_samples_used = _loader_sample_count(prepared['eval_loader'], args.eval_samples)
    methods = _resolve_methods(args, task)
    testing_mode = _resolve_testing_mode(args)
    testing_mode_label = 'Class-Iters' if testing_mode['class_iters'] else 'All-Class'
    aggregated_metrics = _summarize_aggregate_rows(aggregate_rows, class_iters=testing_mode['class_iters'])
    coverage_rate = float(aggregated_metrics['coverage_rate'])
    total_combinations = int(aggregated_metrics['total_combinations'])
    max_coverage = float(aggregated_metrics['max_coverage'])
    final_f1 = float(aggregated_metrics['f1_score'])
    attribution_label = 'WISDOM' if args.impl == 'wisdom' else methods[0]

    summary = {
        'impl': args.impl,
        'task': task,
        'methods': methods,
        'attribution': attribution_label,
        'consensus_methods': methods if args.impl == 'wisdom' else None,
        'pretrain_voting_mode': args.voting_mode if args.impl == 'wisdom' else None,
        'testing_mode': testing_mode,
        'testing_mode_label': testing_mode_label,
        'suite_aggregation': 'Average across classes' if testing_mode['class_iters'] else 'Largest evaluation suite',
        'selection_mode': selection_mode['mode'],
        'selection_mode_label': selection_mode['label'],
        'coverage_aggregation': selection_mode['aggregation_label'],
        'model_path': prepared['model_path'],
        'model_name': Path(str(prepared['model_path'])).name,
        'dataset': prepared['dataset_name'],
        'csv_file': prepared['csv_path'],
        'coverage_rate': coverage_rate,
        'coverage_score': coverage_rate,
        'max_coverage': max_coverage,
        'f1_score': final_f1,
        'pearson_correlation': pearson_correlation,
        'total_combinations': total_combinations,
        'cluster_method': best_cluster_method,
        'n_clusters': best_n_clusters,
        'top_m_neurons': args.top_m_neurons,
        'selected_layers': len(selected),
        'selected_neurons': sum(len(indices) for indices in selected.values()),
        'build_samples': build_samples_used,
        'eval_samples': eval_samples_used,
        'build_subset_requested': args.build_samples,
        'eval_subset_requested': args.eval_samples,
        'bo_enabled': args.bo,
        'bo_result': bo_path,
    }

    breakdown_log = _build_coverage_breakdown_log(suite_rows, selection_mode['label'], selection_mode['aggregation_label'])
    if breakdown_log:
        breakdown_path = out_dir / f'{args.run_name}_coverage_breakdown.log'
        breakdown_path.write_text(breakdown_log + '\n')
        summary['coverage_breakdown_log'] = str(breakdown_path)

    suite_aggregate_log = _build_suite_aggregate_log(suite_rows, testing_mode_label)
    if suite_aggregate_log:
        suite_aggregate_path = out_dir / f'{args.run_name}_suite_coverage.log'
        suite_aggregate_path.write_text(suite_aggregate_log + '\n')
        summary['suite_coverage_log'] = str(suite_aggregate_path)

    if args.combo_log:
        combo_log, combo_counts, unique_combos = _write_combo_log(
            engine,
            prepared['model'],
            prepared['eval_loader'],
            selected,
            args.device,
            out_dir / f'{args.run_name}_combinations.jsonl',
        )
        summary['unique_triggered_combinations'] = unique_combos
        summary['combination_log'] = combo_log
        summary['combination_counts'] = combo_counts

    if args.plot_neurons and prepared['csv_path']:
        summary['neuron_plot'] = _write_neuron_plot(prepared['csv_path'], args.top_m_neurons, out_dir, args.run_name)

    if args.plot_pixels:
        summary.update(_write_pixel_plots(args, prepared, out_dir))

    suite_metrics_csv = out_dir / f'{args.run_name}_suite_metrics.csv'
    summary_csv = out_dir / f'{args.run_name}_coverage.csv'
    summary_json = out_dir / f'{args.run_name}_coverage.json'
    summary['suite_metrics_csv'] = str(suite_metrics_csv)
    pd.DataFrame(suite_rows).to_csv(suite_metrics_csv, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2))

    print(_format_terminal_summary(summary, summary_csv=summary_csv, summary_json=summary_json))
    if testing_mode['class_iters'] and suite_aggregate_log:
        print(suite_aggregate_log)
    if breakdown_log:
        print(breakdown_log)
    return {
        'summary': summary,
        'summary_csv': str(summary_csv),
        'summary_json': str(summary_json),
    }


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == '__main__':
    main()

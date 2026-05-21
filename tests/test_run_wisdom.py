import argparse
import json
import pathlib
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from wisdom.search import SearchResult

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import run_wisdom


class DummyEngine:
    def __init__(self):
        self.groups = {}

    def select_top_neurons_all(self, layer_scores, exclude_last=None):
        return {'layer': [0]}

    def fit_selected(self, build_loader, selected, device=None):
        return selected

    def coverage(self, eval_loader, selected=None, device=None, layer_scores=None, exclude_last=None):
        return 0.5, 4, 0.75


class DummyLoader:
    def __init__(self, size):
        self.dataset = [None] * size


def _base_args(tmp_path, *, impl='wisdom'):
    return argparse.Namespace(
        impl=impl,
        task='classification',
        model_path='dummy_model.pth',
        dataset='mnist',
        data_path='dummy-data',
        img_dir=None,
        csv_file=None,
        pretrain=False,
        methods=None,
        attribution_method='lrp',
        voting_mode='fine-grained',
        device='cpu',
        batch_size=4,
        build_samples=None,
        eval_samples=None,
        top_m_neurons=3,
        cluster_method='Birch',
        n_clusters=2,
        cache_path=None,
        seed=42,
        imgsz=320,
        noise_std=0.3,
        pixel_frac=0.02,
        bo=False,
        bo_backend='auto',
        bo_init=2,
        bo_iter=2,
        bo_cluster_methods='KMeans,Birch',
        bo_n_clusters='2,3',
        corr_points=4,
        end2end=True,
        all_class=True,
        class_iters=False,
        per_group=None,
        per_layer=False,
        combo_log=False,
        plot_neurons=False,
        plot_pixels=False,
        out_dir=str(tmp_path),
        run_name='demo',
    )


def test_run_wisdom_writes_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_wisdom, '_build_engine', lambda *args, **kwargs: DummyEngine())
    monkeypatch.setattr(
        run_wisdom,
        '_compute_correlation_workflow',
        lambda args, prepared, engine, selected: (
            [
                {
                    'suite_size': 2,
                    'coverage_rate': 0.25,
                    'max_coverage': 0.75,
                    'total_combinations': 4,
                    'f1_score': 0.6,
                },
                {
                    'suite_size': 4,
                    'coverage_rate': 0.5,
                    'max_coverage': 0.75,
                    'total_combinations': 4,
                    'f1_score': 0.8,
                },
            ],
            1.0,
        ),
    )
    monkeypatch.setattr(
        run_wisdom,
        '_prepare_classification',
        lambda args, csv_path: {
            'csv_path': str(tmp_path / 'scores.csv'),
            'eval_loader': DummyLoader(4),
            'build_loader': DummyLoader(8),
            'exclude_last': 'fc',
            'layer_scores': {'layer': torch.tensor([1.0])},
            'model': object(),
            'model_path': 'dummy_model.pth',
            'dataset_name': 'mnist',
            'task': 'classification',
        },
    )

    args = _base_args(tmp_path, impl='wisdom')
    args.attribution_method = None
    result = run_wisdom.run(args)

    summary_csv = tmp_path / 'demo_coverage.csv'
    summary_json = tmp_path / 'demo_coverage.json'
    assert summary_csv.is_file()
    assert summary_json.is_file()
    assert result['summary']['cluster_method'] == 'Birch'
    assert result['summary']['build_samples'] == 8
    assert result['summary']['eval_samples'] == 4
    assert result['summary']['attribution'] == 'WISDOM'
    assert result['summary']['consensus_methods'] == run_wisdom.ATTRIBUTION_METHODS
    assert result['summary']['pretrain_voting_mode'] == 'fine-grained'
    assert result['summary']['testing_mode_label'] == 'All-Class'
    assert result['summary']['selection_mode_label'] == 'Global'

    payload = json.loads(summary_json.read_text())
    assert payload['coverage_rate'] == 0.5
    assert payload['coverage_score'] == 0.5
    assert payload['f1_score'] == 0.8
    assert payload['pearson_correlation'] == 1.0
    assert payload['total_combinations'] == 4
    assert payload['suite_metrics_csv'].endswith('demo_suite_metrics.csv')
    assert payload['dataset'] == 'mnist'
    assert payload['model_name'] == 'dummy_model.pth'
    assert payload['attribution'] == 'WISDOM'

    stdout = capsys.readouterr().out
    assert 'Model Name' in stdout
    assert 'Dataset' in stdout
    assert 'Testing Mode' in stdout
    assert 'Selection' in stdout
    assert 'Top-k' in stdout
    assert 'Attribution' in stdout
    assert 'Total Combination' in stdout
    assert 'Max Coverage' in stdout
    assert 'Coverage Rate/Score' in stdout


def test_run_wisdom_idc_forces_kmeans(monkeypatch, tmp_path):
    monkeypatch.setattr(run_wisdom, '_build_engine', lambda *args, **kwargs: DummyEngine())
    monkeypatch.setattr(
        run_wisdom,
        '_compute_correlation_workflow',
        lambda args, prepared, engine, selected: (
            [
                {
                    'suite_size': 4,
                    'coverage_rate': 0.5,
                    'max_coverage': 0.75,
                    'total_combinations': 4,
                    'f1_score': 0.8,
                }
            ],
            0.0,
        ),
    )
    monkeypatch.setattr(
        run_wisdom,
        '_prepare_classification',
        lambda args, csv_path: {
            'csv_path': str(tmp_path / 'scores.csv'),
            'eval_loader': DummyLoader(4),
            'build_loader': DummyLoader(8),
            'exclude_last': None,
            'layer_scores': {'layer': torch.tensor([1.0])},
            'model': object(),
            'model_path': 'dummy_model.pth',
            'dataset_name': 'mnist',
            'task': 'classification',
        },
    )

    args = _base_args(tmp_path, impl='idc')
    result = run_wisdom.run(args)
    assert result['summary']['cluster_method'] == 'KMeans'
    assert result['summary']['attribution'] == 'lrp'


def test_run_wisdom_bo_uses_packaged_search(monkeypatch, tmp_path):
    monkeypatch.setattr(run_wisdom, '_build_engine', lambda *args, **kwargs: DummyEngine())
    monkeypatch.setattr(
        run_wisdom,
        '_compute_correlation_workflow',
        lambda args, prepared, engine, selected: (
            [
                {
                    'suite_size': 4,
                    'coverage_rate': 0.5,
                    'max_coverage': 0.75,
                    'total_combinations': 4,
                    'f1_score': 0.8,
                }
            ],
            0.5,
        ),
    )
    monkeypatch.setattr(
        run_wisdom,
        '_prepare_classification',
        lambda args, csv_path: {
            'csv_path': str(tmp_path / 'scores.csv'),
            'eval_loader': DummyLoader(4),
            'build_loader': DummyLoader(8),
            'exclude_last': None,
            'layer_scores': {'layer': torch.tensor([1.0])},
            'model': object(),
            'model_path': 'dummy_model.pth',
            'dataset_name': 'mnist',
            'task': 'classification',
        },
    )

    seen = {}

    def fake_run_bo(search_space, objective, **kwargs):
        seen['search_space'] = search_space
        seen['kwargs'] = kwargs
        assert callable(objective)
        return (
            SearchResult(
                best_config={'cluster_method': 'Birch', 'n_clusters': 3},
                best_score=0.9,
                history=[({'cluster_method': 'Birch', 'n_clusters': 3}, 0.9)],
                backend='sklearn',
            ),
            str(tmp_path / 'demo_bo.json'),
        )

    monkeypatch.setattr(run_wisdom, 'run_search_bo', fake_run_bo)

    args = _base_args(tmp_path, impl='wisdom')
    args.attribution_method = None
    args.bo = True
    result = run_wisdom.run(args)

    assert seen['search_space'] == {
        'cluster_method': ['KMeans', 'Birch'],
        'n_clusters': [2, 3],
    }
    assert seen['kwargs']['out_path'] == tmp_path / 'demo_bo.json'
    assert result['summary']['cluster_method'] == 'Birch'
    assert result['summary']['n_clusters'] == 3
    assert result['summary']['bo_result'] == str(tmp_path / 'demo_bo.json')


def test_pearson_correlation_handles_constant_series():
    assert run_wisdom._pearson_correlation([0.1, 0.1, 0.1], [0.2, 0.3, 0.4]) == 0.0


def test_suite_sizes_cover_endpoints():
    assert run_wisdom._suite_sizes(5, 3)[0] == 1
    assert run_wisdom._suite_sizes(5, 3)[-1] == 5


def test_subset_loader_uses_full_dataset_when_count_omitted():
    loader = run_wisdom._subset_loader([1, 2, 3], None, batch_size=8)
    assert len(loader.dataset) == 3
    assert loader.batch_size == 3


def test_resolve_methods_supports_multi_method_dnn_defaults(tmp_path):
    args = _base_args(tmp_path, impl='wisdom')
    args.methods = None
    args.attribution_method = None
    assert run_wisdom._resolve_methods(args, 'classification') == run_wisdom.ATTRIBUTION_METHODS


def test_resolve_methods_rejects_multi_method_idc(tmp_path):
    args = _base_args(tmp_path, impl='idc')
    args.methods = ['lrp', 'lig']
    args.attribution_method = None
    try:
        run_wisdom._resolve_methods(args, 'classification')
    except ValueError as exc:
        assert 'exactly one attribution method' in str(exc)
    else:
        raise AssertionError('Expected IDC multi-method validation to fail.')


def test_resolve_testing_mode_class_iters_disables_all_class(tmp_path):
    args = _base_args(tmp_path, impl='wisdom')
    args.class_iters = True
    mode = run_wisdom._resolve_testing_mode(args)
    assert mode == {'end2end': True, 'all_class': False, 'class_iters': True}


def test_resolve_selection_mode_defaults_to_global(tmp_path):
    args = _base_args(tmp_path, impl='wisdom')
    mode = run_wisdom._resolve_selection_mode(args)
    assert mode == {
        'mode': 'global',
        'n_groups': None,
        'label': 'Global',
        'aggregation_label': None,
    }


def test_resolve_selection_mode_per_group(tmp_path):
    args = _base_args(tmp_path, impl='wisdom')
    args.per_group = 4
    mode = run_wisdom._resolve_selection_mode(args)
    assert mode == {
        'mode': 'per-group',
        'n_groups': 4,
        'label': 'Per-Group (4)',
        'aggregation_label': 'Average across 4 groups',
    }


def test_resolve_selection_mode_per_layer(tmp_path):
    args = _base_args(tmp_path, impl='wisdom')
    args.per_layer = True
    mode = run_wisdom._resolve_selection_mode(args)
    assert mode == {
        'mode': 'per-layer',
        'n_groups': None,
        'label': 'Per-Layer',
        'aggregation_label': 'Average across layers',
    }


class _LabelDataset(Dataset):
    def __init__(self):
        self.samples = [
            ('a', 0),
            ('b', 1),
            ('c', 2),
            ('d', 3),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_class_iter_loaders_use_eval_loader_subset():
    dataset = _LabelDataset()
    subset = Subset(dataset, [1, 3])
    prepared = {
        'eval_dataset': dataset,
        'eval_loader': DataLoader(subset, batch_size=2),
        'classes': ['zero', 'one', 'two', 'three'],
        'eval_collate_fn': None,
    }

    loaders = run_wisdom._class_iter_loaders(prepared, batch_size=2)
    assert [(class_id, class_name, len(loader.dataset)) for class_id, class_name, loader in loaders] == [
        (1, 'one', 1),
        (3, 'three', 1),
    ]


def test_resolve_selected_neurons_per_layer_filters_layers(tmp_path):
    csv_path = tmp_path / 'scores.csv'
    pd.DataFrame(
        [
            {'LayerName': 'yolo_model.model.0.conv', 'NeuronIndex': 0, 'Score': 0.9},
            {'LayerName': 'yolo_model.model.1.conv', 'NeuronIndex': 1, 'Score': 0.8},
            {'LayerName': 'yolo_model.head.0', 'NeuronIndex': 0, 'Score': 1.1},
        ]
    ).to_csv(csv_path, index=False)

    args = _base_args(tmp_path, impl='wisdom')
    args.task = 'detection'
    args.per_layer = True
    prepared = {
        'task': 'detection',
        'csv_path': str(csv_path),
        'layer_scores': {
            'model.0.conv': torch.tensor([1.0]),
            'model.1.conv': torch.tensor([1.0]),
        },
        'exclude_last': None,
    }

    selected = run_wisdom._resolve_selected_neurons(args, prepared, DummyEngine())
    assert selected == {
        'model.0.conv': [0],
        'model.1.conv': [1],
    }

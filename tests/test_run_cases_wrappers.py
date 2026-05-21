import sys
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

import run_cases.run_rq1 as rq1
import run_cases.run_rq2 as rq2
import run_cases.run_rq3 as rq3
import run_cases.run_rq4 as rq4
import run_cases.run_rq5 as rq5
import run_cases.smoke as smoke
from wisdom.search import SearchResult


def test_rq2_wrapper_delegates(monkeypatch):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return 'ok'

    monkeypatch.setattr(rq2, 'run_rq2', fake_run)
    result = rq2.run_rq2(weights='w', img_dir='i', csv_file='c')
    assert result == 'ok'
    assert seen['weights'] == 'w'


def test_rq3_wrapper_delegates(monkeypatch):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return 'ok'

    monkeypatch.setattr(rq3, 'run_rq3', fake_run)
    result = rq3.run_rq3(weights='w', img_dir='i', csv_file='c')
    assert result == 'ok'
    assert seen['csv_file'] == 'c'


def test_rq4_wrapper_delegates(monkeypatch):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return 'ok'

    monkeypatch.setattr(rq4, 'run_rq4', fake_run)
    result = rq4.run_rq4(weights='w', img_dir='i', csv_file='c')
    assert result == 'ok'
    assert seen['img_dir'] == 'i'


def test_rq5_wrapper_delegates(monkeypatch):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return 'ok'

    monkeypatch.setattr(rq5, 'run_rq5', fake_run)
    result = rq5.run_rq5(weights='w', img_dir='i', csv_file='c')
    assert result == 'ok'
    assert seen['weights'] == 'w'


def test_rq1_wrapper_exports_runner():
    assert callable(rq1.run_rq1)


def test_smoke_score_wrapper_delegates(monkeypatch):
    seen = {}

    def fake_train(**kwargs):
        seen.update(kwargs)
        return 'scores.csv'

    monkeypatch.setattr(smoke, 'train_wisdom_yolo', fake_train)
    result = smoke.run_yolo_score_smoke('w', 'i', 'o.csv', device='cpu')
    assert result == 'scores.csv'
    assert seen['out_csv'] == 'o.csv'


def test_smoke_rq2_wrapper_delegates(monkeypatch):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return 'rq2.csv'

    monkeypatch.setattr(smoke, 'run_rq2_opt', fake_run)
    result = smoke.run_yolo_rq2_smoke('w', 'i', 'scores.csv', 'out', device='cpu')
    assert result == 'rq2.csv'
    assert seen['csv_file'] == 'scores.csv'


def test_smoke_bo_search_space_caps_clusters_for_tiny_build_sets():
    assert smoke._bo_search_space(2) == {
        'algo': ['KMeans', 'MiniBatchKMeans', 'Birch'],
        'n_clusters': [2],
    }
    assert smoke._bo_search_space(4) == {
        'algo': ['KMeans', 'MiniBatchKMeans', 'Birch'],
        'n_clusters': [2, 3],
    }


def test_smoke_dnn_bo_writes_summary(monkeypatch, tmp_path):
    sample = [(torch.zeros(1, 32, 32), 0) for _ in range(4)]

    class DummyModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_get_data(dataset, batch_size, data_path):
        return None, None, sample, sample, ['0']

    def fake_run_bo(search_space, objective, **kwargs):
        out_path = pathlib.Path(kwargs['out_path'])
        out_path.write_text(json.dumps({
            'smoke_type': 'dnn-bo',
            'backend_used': 'sklearn',
            'best_config': {'algo': 'KMeans', 'n_clusters': 2},
            'best_score': 1.0,
        }))
        return (
            SearchResult(
                best_config={'algo': 'KMeans', 'n_clusters': 2},
                best_score=1.0,
                history=[({'algo': 'KMeans', 'n_clusters': 2}, 1.0)],
                backend='sklearn',
            ),
            str(out_path),
        )

    monkeypatch.setattr(smoke, 'get_data', fake_get_data)
    monkeypatch.setattr(smoke, '_load_legacy_torch_model', lambda path: DummyModel())
    monkeypatch.setattr(smoke, '_load_top_neurons', lambda *args, **kwargs: {'layer': [0]})
    monkeypatch.setattr(smoke, 'run_bo', fake_run_bo)

    out = smoke.run_dnn_bo_smoke(
        str(tmp_path / 'dnn_bo.json'),
        model_path='dummy_model.pth',
        csv_file='dummy.csv',
        data_path='dummy-data',
        device='cpu',
        build_samples=2,
        eval_samples=2,
    )

    payload = json.loads(pathlib.Path(out).read_text())
    assert payload['smoke_type'] == 'dnn-bo'
    assert payload['backend_used'] == 'sklearn'
    assert payload['best_config']['algo'] == 'KMeans'

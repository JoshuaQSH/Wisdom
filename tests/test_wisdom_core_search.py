import sys
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wisdom.search import BOSearch, run_bo


def test_bo_search_improves_simple_objective():
    search = BOSearch(
        {
            'x': (-2.0, 2.0),
            'y': (-3.0, 1.0),
            'algo': ['kmeans', 'meanshift'],
        },
        random_state=7,
    )

    def objective(cfg):
        algo_bonus = 0.2 if cfg['algo'] == 'kmeans' else 0.0
        return -((cfg['x'] - 0.5) ** 2) - ((cfg['y'] + 1.0) ** 2) + algo_bonus

    result = search.optimize(objective, n_init=5, n_iter=8, candidate_pool_size=64, backend='sklearn')

    assert result.best_score > -1.0
    assert result.best_config['algo'] in {'kmeans', 'meanshift'}
    assert len(result.history) == 13
    assert result.backend == 'sklearn'


def test_bo_search_auto_backend_reports_backend():
    search = BOSearch({'x': (-1.0, 1.0), 'algo': ['a', 'b']}, random_state=3, backend='auto')

    def objective(cfg):
        return 1.0 if cfg['algo'] == 'a' else 0.5

    result = search.optimize(objective, n_init=2, n_iter=1, candidate_pool_size=16)

    assert result.backend in {'sklearn', 'botorch'}
    assert len(result.history) == 3


def test_bo_search_sklearn_handles_small_finite_space_without_hanging():
    search = BOSearch(
        {
            'cluster_method': ['KMeans', 'MiniBatchKMeans', 'Birch'],
            'n_clusters': (2, 3, 'int'),
        },
        random_state=5,
        backend='sklearn',
    )

    def objective(cfg):
        return 1.0 if (cfg['cluster_method'], cfg['n_clusters']) == ('KMeans', 3) else 0.0

    result = search.optimize(objective, n_init=3, n_iter=3, candidate_pool_size=16, backend='sklearn')

    assert len(result.history) == 6
    assert len({tuple(sorted(cfg.items())) for cfg, _ in result.history}) == 6
    assert result.best_config == {'cluster_method': 'KMeans', 'n_clusters': 3}


def test_run_bo_writes_packaged_summary(tmp_path):
    out_path = tmp_path / 'bo.json'

    def objective(cfg):
        return 1.0 if cfg['algo'] == 'a' else 0.5

    result, written = run_bo(
        {'algo': ['a', 'b']},
        objective,
        random_state=1,
        backend='sklearn',
        n_init=1,
        n_iter=1,
        candidate_pool_size=8,
        out_path=out_path,
        payload_extras={'tag': 'smoke'},
    )

    payload = json.loads(out_path.read_text())
    assert result.best_config['algo'] == 'a'
    assert written == str(out_path)
    assert payload['tag'] == 'smoke'
    assert payload['best_config']['algo'] == 'a'
    assert payload['num_evaluations'] == 2

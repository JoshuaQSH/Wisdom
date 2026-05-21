import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from optimize.run_rq2_opt import (
    _bo_objective_value,
    _build_bo_search_space,
    summarize_rq2_df,
)


def test_summarize_rq2_df_computes_expected_ratios():
    df = pd.DataFrame(
        [
            {"variant": "I", "coverage_mode": "cluster", "scope": "dataset", "C_O_overall": 0.10, "delta_overall": 0.03, "delta_early": 0.02, "mag_overall": 1.5, "mag_early": 1.2},
            {"variant": "R", "coverage_mode": "cluster", "scope": "dataset", "C_O_overall": 0.10, "delta_overall": 0.02, "delta_early": 0.01, "mag_overall": 1.0, "mag_early": 0.8},
            {"variant": "I_obj", "coverage_mode": "cluster", "scope": "dataset", "C_O_overall": 0.10, "delta_overall": 0.04, "delta_early": 0.03, "mag_overall": 1.8, "mag_early": 1.4},
            {"variant": "R_obj", "coverage_mode": "cluster", "scope": "dataset", "C_O_overall": 0.10, "delta_overall": 0.02, "delta_early": 0.01, "mag_overall": 1.1, "mag_early": 0.9},
            {"variant": "I_bg", "coverage_mode": "cluster", "scope": "dataset", "C_O_overall": 0.10, "delta_overall": 0.01, "delta_early": 0.005, "mag_overall": 0.6, "mag_early": 0.4},
            {"variant": "R_bg", "coverage_mode": "cluster", "scope": "dataset", "C_O_overall": 0.10, "delta_overall": 0.02, "delta_early": 0.01, "mag_overall": 0.8, "mag_early": 0.5},
        ]
    )

    summary = summarize_rq2_df(df)

    assert summary["coverage_mode"] == "cluster"
    assert summary["scope"] == "dataset"
    assert summary["baseline_mean"] == pytest.approx(0.10)
    assert summary["sections"]["classic"]["ratio"] == pytest.approx(1.5)
    assert summary["sections"]["object"]["ratio"] == pytest.approx(2.0)
    assert summary["sections"]["background"]["ratio"] == pytest.approx(0.5)
    assert summary["sections"]["classic"]["by_group"]["early"]["ratio"] == pytest.approx(2.0)
    assert summary["importance_object_vs_background_ratio"] == pytest.approx(4.0)
    assert summary["passes"] == 2
    assert summary["magnitude"]["classic"]["ratio"] == pytest.approx(1.5)


def test_build_bo_search_space_parses_cli_strings():
    search_space = _build_bo_search_space(
        bo_cluster_method_values="KMeans,MiniBatchKMeans,Birch",
        bo_n_clusters_values="2,3",
    )

    assert search_space["cluster_method"] == ["KMeans", "MiniBatchKMeans", "Birch"]
    assert search_space["n_clusters"] == [2, 3]


def test_bo_objective_value_selects_requested_metric():
    summary = {
        "sections": {
            "classic": {"ratio": 1.1, "delta_i": 0.03, "delta_r": 0.02},
            "object": {"ratio": 1.3, "delta_i": 0.04, "delta_r": 0.03},
            "background": {"ratio": 0.9, "delta_i": 0.01, "delta_r": 0.02},
        }
    }

    assert _bo_objective_value(summary, "classic_ratio") == pytest.approx(1.5)
    assert _bo_objective_value(summary, "object_ratio") == pytest.approx(4.0 / 3.0)
    assert _bo_objective_value(summary, "background_ratio") == pytest.approx(0.5)
    assert _bo_objective_value(summary, "mean_ratio") == pytest.approx((1.5 + (4.0 / 3.0) + 0.5) / 3)
    assert _bo_objective_value(summary, "classic_gap") == pytest.approx(0.01)


def test_bo_ratio_objectives_do_not_explode_when_random_gain_is_zero():
    summary = {
        "sections": {
            "classic": {"ratio": 999999.0, "delta_i": 0.03, "delta_r": 0.0},
            "object": {"ratio": 888888.0, "delta_i": 0.02, "delta_r": 0.0},
            "background": {"ratio": 0.0, "delta_i": 0.0, "delta_r": 0.0},
        }
    }

    assert _bo_objective_value(summary, "classic_ratio") == pytest.approx(0.03)
    assert _bo_objective_value(summary, "object_ratio") == pytest.approx(0.02)
    assert _bo_objective_value(summary, "background_ratio") == pytest.approx(0.0)
    assert _bo_objective_value(summary, "mean_ratio") == pytest.approx((0.03 + 0.02 + 0.0) / 3)

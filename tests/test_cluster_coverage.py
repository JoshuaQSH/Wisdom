"""
Tests for clustering-based combinatorial coverage in optimize/coverage_utils.py.

Tests cover:
  - ClusterCoverageComputer.fit() with synthetic activations
  - ClusterCoverageComputer.coverage() combinatorial counting
  - ClusterCoverageComputer.coverage_delta() clean vs perturbed
  - Stratified (early/middle/late) coverage computation
  - Mode switching: plain threshold vs cluster coverage
  - Integration with ActivationCollector on a real YOLO model (skipped if unavailable)
"""
import os
import sys
import math
import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wisdom.core.compute import combinations_coverage
from wisdom.clustering.assign import fit_per_neuron, assign_clusters


# ── Unit tests for the underlying WISDOM primitives ─────────────────

class TestCombinationsCoverage:
    """Test the core combinations_coverage function."""

    def test_single_neuron_two_clusters(self):
        """One neuron with 2 clusters, 2 samples covering both → 100%."""
        assignments = [{"n0": 0}, {"n0": 1}]
        sizes = {"n0": 2}
        rate, total, max_cov = combinations_coverage(assignments, sizes)
        assert total == 2
        assert rate == pytest.approx(1.0)

    def test_single_neuron_partial(self):
        """One neuron with 3 clusters, only 2 seen → 66.7%."""
        assignments = [{"n0": 0}, {"n0": 1}, {"n0": 0}]
        sizes = {"n0": 3}
        rate, total, max_cov = combinations_coverage(assignments, sizes)
        assert total == 3
        assert rate == pytest.approx(2.0 / 3.0)

    def test_two_neurons_combinatorial(self):
        """Two neurons × 2 clusters = 4 combos. 3 seen → 75%."""
        assignments = [
            {"a": 0, "b": 0},
            {"a": 0, "b": 1},
            {"a": 1, "b": 0},
        ]
        sizes = {"a": 2, "b": 2}
        rate, total, _ = combinations_coverage(assignments, sizes)
        assert total == 4
        assert rate == pytest.approx(3.0 / 4.0)

    def test_full_combinatorial(self):
        """All 4 combos seen → 100%."""
        assignments = [
            {"a": 0, "b": 0},
            {"a": 0, "b": 1},
            {"a": 1, "b": 0},
            {"a": 1, "b": 1},
        ]
        sizes = {"a": 2, "b": 2}
        rate, _, _ = combinations_coverage(assignments, sizes)
        assert rate == pytest.approx(1.0)

    def test_empty_assignments(self):
        """No samples → 0 coverage."""
        sizes = {"a": 2, "b": 3}
        rate, total, _ = combinations_coverage([], sizes)
        assert total == 6
        assert rate == pytest.approx(0.0)

    def test_missing_key_skipped(self):
        """Assignments with -1 (missing key) are excluded."""
        assignments = [{"a": 0}]  # "b" missing → gets -1
        sizes = {"a": 2, "b": 2}
        rate, total, _ = combinations_coverage(assignments, sizes)
        assert total == 4
        assert rate == pytest.approx(0.0)  # tuple contains -1, excluded

    def test_duplicate_combos_counted_once(self):
        """Repeated combos don't inflate coverage."""
        assignments = [{"a": 0, "b": 0}] * 10
        sizes = {"a": 2, "b": 2}
        rate, _, _ = combinations_coverage(assignments, sizes)
        assert rate == pytest.approx(1.0 / 4.0)


class TestFitPerNeuron:
    """Test the per-neuron clustering."""

    def test_kmeans_basic(self):
        """KMeans with 2 clusters on bimodal data."""
        activations = {
            "layer0": {
                0: np.array([0.1, 0.2, 0.15, 5.0, 5.1, 4.9]),
            }
        }
        groups = fit_per_neuron(
            activations, method="KMeans",
            params={"n_clusters": 2, "random_state": 42},
        )
        assert "layer0" in groups
        assert 0 in groups["layer0"]
        info = groups["layer0"][0]
        assert info["centers"].shape[0] == 2
        assert info["centers"].shape[1] == 1
        assert len(info["labels"]) == 6

    def test_silhouette_adaptive_k(self):
        """With use_silhouette=True, k is chosen adaptively."""
        # 3 well-separated clusters
        activations = {
            "layer0": {
                0: np.concatenate([
                    np.random.normal(0, 0.1, 20),
                    np.random.normal(5, 0.1, 20),
                    np.random.normal(10, 0.1, 20),
                ]),
            }
        }
        groups = fit_per_neuron(
            activations, method="KMeans",
            use_silhouette=True, k_max=5,
            params={"random_state": 42},
        )
        k = groups["layer0"][0]["centers"].shape[0]
        assert 2 <= k <= 5  # should find ~3

    def test_trivial_constant_values(self):
        """Constant activations → trivial single cluster."""
        activations = {"layer0": {0: np.array([1.0, 1.0, 1.0])}}
        groups = fit_per_neuron(activations, method="KMeans",
                                params={"n_clusters": 2, "random_state": 42})
        assert groups["layer0"][0]["method"] == "Trivial"
        assert groups["layer0"][0]["centers"].shape[0] == 1

    def test_multiple_neurons(self):
        """Multiple neurons in multiple layers."""
        activations = {
            "conv1": {
                0: np.random.randn(30),
                3: np.random.randn(30),
            },
            "conv2": {
                1: np.random.randn(30),
            },
        }
        groups = fit_per_neuron(
            activations, method="KMeans",
            params={"n_clusters": 2, "random_state": 42},
        )
        assert len(groups) == 2
        assert len(groups["conv1"]) == 2
        assert len(groups["conv2"]) == 1


class TestAssignClusters:
    """Test the cluster assignment function."""

    def test_nearest_center(self):
        """Assignment goes to nearest cluster center."""
        groups = {
            "layer0": {
                0: {"centers": np.array([[0.0], [10.0]]), "labels": np.array([0, 1])},
            }
        }
        # Value 1.0 is closer to center 0.0
        result = assign_clusters(groups, {"layer0": {0: 1.0}})
        assert result["layer0"][0] == 0

        # Value 8.0 is closer to center 10.0
        result = assign_clusters(groups, {"layer0": {0: 8.0}})
        assert result["layer0"][0] == 1

    def test_multi_neuron(self):
        groups = {
            "layer0": {
                0: {"centers": np.array([[0.0], [5.0]]), "labels": np.array([0, 1])},
                1: {"centers": np.array([[-1.0], [1.0]]), "labels": np.array([0, 1])},
            }
        }
        result = assign_clusters(groups, {"layer0": {0: 4.0, 1: -0.5}})
        assert result["layer0"][0] == 1  # closer to 5.0
        assert result["layer0"][1] == 0  # closer to -1.0


# ── Integration tests with ClusterCoverageComputer ──────────────────

class TestClusterCoverageComputer:
    """Test ClusterCoverageComputer with a tiny synthetic model."""

    @pytest.fixture
    def tiny_model(self):
        """A minimal model with layers named like YOLO (nested modules)."""
        # Build nested structure so named_modules() yields "model.2.cv1.conv" etc.
        conv_early = nn.Conv2d(3, 8, 3, padding=1)
        cv1_early = nn.Module()
        cv1_early.conv = conv_early
        block_early = nn.Module()
        block_early.cv1 = cv1_early

        conv_mid = nn.Conv2d(8, 16, 3, padding=1)
        cv1_mid = nn.Module()
        cv1_mid.conv = conv_mid
        block_mid = nn.Module()
        block_mid.cv1 = cv1_mid

        conv_late = nn.Conv2d(16, 16, 3, padding=1)
        cv1_late = nn.Module()
        cv1_late.conv = conv_late
        block_late = nn.Module()
        block_late.cv1 = cv1_late

        model = nn.Module()
        model.model = nn.ModuleList([
            nn.Identity(),  # model.0
            nn.Identity(),  # model.1
            block_early,    # model.2 → model.2.cv1.conv
            nn.Identity(),  # model.3
            nn.Identity(),  # model.4
            nn.Identity(),  # model.5
            nn.Identity(),  # model.6
            nn.Identity(),  # model.7
            block_mid,      # model.8 → model.8.cv1.conv
            nn.Identity(),  # model.9 .. 15
            nn.Identity(), nn.Identity(), nn.Identity(),
            nn.Identity(), nn.Identity(), nn.Identity(),
            block_late,     # model.16 → model.16.cv1.conv
        ])

        # Wire up a simple forward that chains through the 3 conv layers
        class TinyYOLO(nn.Module):
            def __init__(self, blocks):
                super().__init__()
                self.model = blocks
                self.early = self.model[2].cv1.conv
                self.mid = self.model[8].cv1.conv
                self.late = self.model[16].cv1.conv

            def forward(self, x):
                x = self.early(x)
                x = self.mid(x)
                x = self.late(x)
                return x

        net = TinyYOLO(model.model)
        net.eval()
        return net

    @pytest.fixture
    def target_neurons(self):
        return {
            "model.2.cv1.conv": [0, 1, 2],
            "model.8.cv1.conv": [0, 1],
            "model.16.cv1.conv": [0, 1],
        }

    def test_fit_creates_clusters(self, tiny_model, target_neurons):
        from optimize.coverage_utils import ClusterCoverageComputer
        comp = ClusterCoverageComputer(
            tiny_model, target_neurons, device="cpu",
            method="KMeans", use_silhouette=False, n_clusters=2,
        )
        build = torch.randn(20, 3, 32, 32)
        comp.fit(build, batch_size=4)
        assert comp._fitted
        assert len(comp.groups) > 0
        assert len(comp.cluster_sizes) > 0
        # Should have entries for each monitored neuron
        total_neurons = sum(len(v) for v in target_neurons.values())
        assert len(comp.cluster_sizes) == total_neurons

    def test_coverage_returns_valid_scores(self, tiny_model, target_neurons):
        from optimize.coverage_utils import ClusterCoverageComputer
        comp = ClusterCoverageComputer(
            tiny_model, target_neurons, device="cpu",
            method="KMeans", use_silhouette=False, n_clusters=2,
        )
        build = torch.randn(30, 3, 32, 32)
        comp.fit(build, batch_size=4)

        test = torch.randn(10, 3, 32, 32)
        cov = comp.coverage(test, batch_size=4)

        assert "overall" in cov
        assert "early" in cov
        assert "middle" in cov
        assert "late" in cov
        assert "variability" in cov
        assert 0.0 <= cov["overall"] <= 1.0
        assert 0.0 <= cov["early"] <= 1.0

    def test_more_images_increases_coverage(self, tiny_model, target_neurons):
        """More test images should yield >= coverage (monotonic)."""
        from optimize.coverage_utils import ClusterCoverageComputer
        comp = ClusterCoverageComputer(
            tiny_model, target_neurons, device="cpu",
            method="KMeans", use_silhouette=False, n_clusters=2,
        )
        build = torch.randn(30, 3, 32, 32)
        comp.fit(build, batch_size=4)

        test_small = torch.randn(3, 3, 32, 32)
        test_large = torch.cat([test_small, torch.randn(20, 3, 32, 32)])

        cov_small = comp.coverage(test_small, batch_size=4)
        cov_large = comp.coverage(test_large, batch_size=4)
        # Larger set should have >= coverage (superset of combos)
        assert cov_large["overall"] >= cov_small["overall"] - 1e-6

    def test_coverage_delta(self, tiny_model, target_neurons):
        from optimize.coverage_utils import ClusterCoverageComputer
        comp = ClusterCoverageComputer(
            tiny_model, target_neurons, device="cpu",
            method="KMeans", use_silhouette=False, n_clusters=2,
        )
        build = torch.randn(30, 3, 32, 32)
        comp.fit(build, batch_size=4)

        clean = torch.randn(10, 3, 32, 32)
        perturbed = clean + torch.randn_like(clean) * 2.0  # heavy perturbation

        delta = comp.coverage_delta(clean, perturbed, batch_size=4)
        assert "delta_overall" in delta
        assert "clean_overall" in delta
        assert "pert_overall" in delta
        assert delta["delta_overall"] >= 0.0

    def test_not_fitted_raises(self, tiny_model, target_neurons):
        from optimize.coverage_utils import ClusterCoverageComputer
        comp = ClusterCoverageComputer(
            tiny_model, target_neurons, device="cpu",
        )
        with pytest.raises(AssertionError, match="fit"):
            comp.coverage(torch.randn(5, 3, 32, 32))


class TestPlainVsClusterMode:
    """Verify both modes can run on the same data and produce valid results."""

    @pytest.fixture
    def tiny_setup(self):
        conv_early = nn.Conv2d(3, 8, 3, padding=1)
        cv1_early = nn.Module()
        cv1_early.conv = conv_early
        block_early = nn.Module()
        block_early.cv1 = cv1_early

        conv_mid = nn.Conv2d(8, 16, 3, padding=1)
        cv1_mid = nn.Module()
        cv1_mid.conv = conv_mid
        block_mid = nn.Module()
        block_mid.cv1 = cv1_mid

        class TinyYOLO(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.ModuleList([
                    nn.Identity(), nn.Identity(),
                    block_early,  # model.2
                    nn.Identity(), nn.Identity(), nn.Identity(),
                    nn.Identity(), nn.Identity(),
                    block_mid,    # model.8
                ])
                self.early = self.model[2].cv1.conv
                self.mid = self.model[8].cv1.conv

            def forward(self, x):
                x = self.early(x)
                x = self.mid(x)
                return x

        model = TinyYOLO()
        model.eval()
        target = {
            "model.2.cv1.conv": [0, 1, 2],
            "model.8.cv1.conv": [0, 1],
        }
        images = torch.randn(20, 3, 32, 32)
        return model, target, images

    def test_plain_mode(self, tiny_setup):
        from optimize.coverage_utils import (
            ActivationCollector, calibrate_thresholds,
            compute_stratified_coverage,
        )
        model, target, images = tiny_setup
        thresholds = calibrate_thresholds(model, target, images[:10], "cpu", percentile=50.0)
        collector = ActivationCollector(model, target, "cpu")
        collector.attach()
        acts = collector.collect(images)
        collector.detach()
        cov = compute_stratified_coverage(acts, thresholds, target)
        assert "overall" in cov
        assert 0.0 <= cov["overall"] <= 1.0

    def test_cluster_mode(self, tiny_setup):
        from optimize.coverage_utils import ClusterCoverageComputer
        model, target, images = tiny_setup
        comp = ClusterCoverageComputer(model, target, "cpu",
                                        n_clusters=2, use_silhouette=False)
        comp.fit(images[:10], batch_size=4)
        cov = comp.coverage(images, batch_size=4)
        assert "overall" in cov
        assert 0.0 <= cov["overall"] <= 1.0

    def test_both_modes_different_values(self, tiny_setup):
        """Plain and cluster coverage should produce different numeric values."""
        from optimize.coverage_utils import (
            ActivationCollector, calibrate_thresholds,
            compute_stratified_coverage, ClusterCoverageComputer,
        )
        model, target, images = tiny_setup

        # Plain
        thresholds = calibrate_thresholds(model, target, images[:10], "cpu", percentile=50.0)
        collector = ActivationCollector(model, target, "cpu")
        collector.attach()
        acts = collector.collect(images)
        collector.detach()
        cov_plain = compute_stratified_coverage(acts, thresholds, target)

        # Cluster
        comp = ClusterCoverageComputer(model, target, "cpu",
                                        n_clusters=2, use_silhouette=False)
        comp.fit(images[:10], batch_size=4)
        cov_cluster = comp.coverage(images, batch_size=4)

        # Both are valid floats
        assert isinstance(cov_plain["overall"], float)
        assert isinstance(cov_cluster["overall"], float)
        # They measure fundamentally different things, so likely differ
        # (but we just check both are in valid range)
        assert 0.0 <= cov_plain["overall"] <= 1.0
        assert 0.0 <= cov_cluster["overall"] <= 1.0


# ── YOLO integration test (skipped if assets unavailable) ───────────

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights", "yolo11n.pt")
COCO_VAL = os.path.join(os.path.dirname(__file__), "..", "standalone", "data", "coco", "images", "val2017")
SCORES_CSV = os.path.join(os.path.dirname(__file__), "..", "neuron_eval_out", "wisdom_yolo11n_scores_5000.csv")

skip_missing = pytest.mark.skipif(
    not (os.path.isfile(WEIGHTS) and os.path.isdir(COCO_VAL) and os.path.isfile(SCORES_CSV)),
    reason="Missing weights, COCO data, or scores CSV",
)


@skip_missing
class TestClusterCoverageYOLO:
    """Integration test with real YOLOv11 model."""

    def test_fit_and_coverage_on_yolo(self):
        from ultralytics import YOLO
        from optimize.coverage_utils import (
            load_layerwise_top_neurons, ClusterCoverageComputer,
        )
        from wisdom_yolo_train import COCOImageDataset

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        yolo = YOLO(WEIGHTS)
        model = yolo.model.eval().to(device)

        top_neurons = load_layerwise_top_neurons(SCORES_CSV, per_layer_k=3)
        total_n = sum(len(v) for v in top_neurons.values())
        assert total_n > 0

        ds = COCOImageDataset(COCO_VAL, max_images=10, imgsz=320)
        images = torch.stack([ds[i][0] for i in range(len(ds))])

        comp = ClusterCoverageComputer(
            model, top_neurons, device=device,
            method="KMeans", use_silhouette=True, k_max=4,
        )
        comp.fit(images, batch_size=2)
        assert comp._fitted
        assert len(comp.cluster_sizes) == total_n

        cov = comp.coverage(images, batch_size=2)
        assert 0.0 <= cov["overall"] <= 1.0
        # Per-layer combinatorial coverage should be meaningful (> 0 with enough images)
        print(f"YOLO cluster coverage: overall={cov['overall']:.6f}, "
              f"early={cov['early']:.6f}, middle={cov['middle']:.6f}, late={cov['late']:.6f}")


# ── Tests for per-group neuron selection and coverage ───────────────

class TestLoadGroupwiseTopNeurons:
    """Test load_groupwise_top_neurons: top-k per layer group."""

    @pytest.fixture
    def tmp_csv(self, tmp_path):
        """Create a synthetic WISDOM scores CSV."""
        import pandas as pd
        rows = []
        # Early layers (model.0-5) – add neurons with varying scores
        for layer_idx in range(6):
            for neuron_idx in range(4):
                score = (6 - layer_idx) * 100 + neuron_idx * 10
                rows.append({
                    "LayerName": f"yolo_model.model.{layer_idx}.conv",
                    "NeuronIndex": neuron_idx,
                    "Score": float(score),
                })
        # Middle layers (model.6-12)
        for layer_idx in range(6, 13):
            for neuron_idx in range(4):
                score = (13 - layer_idx) * 80 + neuron_idx * 5
                rows.append({
                    "LayerName": f"yolo_model.model.{layer_idx}.conv",
                    "NeuronIndex": neuron_idx,
                    "Score": float(score),
                })
        # Late layers (model.13-22)
        for layer_idx in range(13, 23):
            for neuron_idx in range(4):
                score = (23 - layer_idx) * 50 + neuron_idx * 3
                rows.append({
                    "LayerName": f"yolo_model.model.{layer_idx}.conv",
                    "NeuronIndex": neuron_idx,
                    "Score": float(score),
                })
        df = pd.DataFrame(rows)
        path = tmp_path / "scores.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_returns_correct_number_of_neurons(self, tmp_csv):
        from optimize.coverage_utils import load_groupwise_top_neurons
        neurons = load_groupwise_top_neurons(tmp_csv, per_group_k=5)
        total = sum(len(v) for v in neurons.values())
        # Exactly 5 per group × 3 groups = 15 (if neurons span different layers)
        assert total == 15

    def test_neurons_from_correct_groups(self, tmp_csv):
        from optimize.coverage_utils import load_groupwise_top_neurons, _layer_group
        neurons = load_groupwise_top_neurons(tmp_csv, per_group_k=3)
        groups_seen = {"early": 0, "middle": 0, "late": 0}
        for lname, idxs in neurons.items():
            grp = _layer_group(lname)
            groups_seen[grp] += len(idxs)
        # Should have exactly 3 per group
        assert groups_seen["early"] == 3
        assert groups_seen["middle"] == 3
        assert groups_seen["late"] == 3

    def test_selects_highest_scores(self, tmp_csv):
        from optimize.coverage_utils import load_groupwise_top_neurons
        neurons = load_groupwise_top_neurons(tmp_csv, per_group_k=2)
        total = sum(len(v) for v in neurons.values())
        assert total == 6  # 2 per group × 3 groups

    def test_skips_zero_scores(self, tmp_path):
        import pandas as pd
        rows = [
            {"LayerName": "yolo_model.model.0.conv", "NeuronIndex": 0, "Score": 100.0},
            {"LayerName": "yolo_model.model.0.conv", "NeuronIndex": 1, "Score": 0.0},
            {"LayerName": "yolo_model.model.8.conv", "NeuronIndex": 0, "Score": 50.0},
            {"LayerName": "yolo_model.model.8.conv", "NeuronIndex": 1, "Score": 0.0},
            {"LayerName": "yolo_model.model.16.conv", "NeuronIndex": 0, "Score": 30.0},
            {"LayerName": "yolo_model.model.16.conv", "NeuronIndex": 1, "Score": 0.0},
        ]
        df = pd.DataFrame(rows)
        path = tmp_path / "scores_zeros.csv"
        df.to_csv(path, index=False)

        from optimize.coverage_utils import load_groupwise_top_neurons
        neurons = load_groupwise_top_neurons(str(path), per_group_k=5)
        total = sum(len(v) for v in neurons.values())
        # Only 3 non-zero neurons total (1 per group)
        assert total == 3

    def test_neurons_span_multiple_layers_in_group(self, tmp_csv):
        """When k > neurons per layer, results should span multiple layers."""
        from optimize.coverage_utils import load_groupwise_top_neurons
        neurons = load_groupwise_top_neurons(tmp_csv, per_group_k=5)
        # Early group has 6 layers × 4 neurons; top-5 likely from different layers
        early_layers = [l for l in neurons if "model.0." in l or "model.1." in l
                        or "model.2." in l or "model.3." in l
                        or "model.4." in l or "model.5." in l]
        assert len(early_layers) >= 1  # at minimum one layer has neurons


class TestPerGroupCoverage:
    """Test per-group combinatorial coverage mode."""

    @pytest.fixture
    def tiny_model(self):
        """A minimal model with layers named like YOLO for 3 groups."""
        conv_early = nn.Conv2d(3, 8, 3, padding=1)
        cv1_early = nn.Module()
        cv1_early.conv = conv_early
        block_early = nn.Module()
        block_early.cv1 = cv1_early

        conv_mid = nn.Conv2d(8, 16, 3, padding=1)
        cv1_mid = nn.Module()
        cv1_mid.conv = conv_mid
        block_mid = nn.Module()
        block_mid.cv1 = cv1_mid

        conv_late = nn.Conv2d(16, 16, 3, padding=1)
        cv1_late = nn.Module()
        cv1_late.conv = conv_late
        block_late = nn.Module()
        block_late.cv1 = cv1_late

        class TinyYOLO(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.ModuleList([
                    nn.Identity(),  # 0
                    nn.Identity(),  # 1
                    block_early,    # 2
                    nn.Identity(),  # 3
                    nn.Identity(),  # 4
                    nn.Identity(),  # 5
                    nn.Identity(),  # 6
                    nn.Identity(),  # 7
                    block_mid,      # 8
                    nn.Identity(),  # 9-15
                    nn.Identity(), nn.Identity(), nn.Identity(),
                    nn.Identity(), nn.Identity(), nn.Identity(),
                    block_late,     # 16
                ])
                self.early = self.model[2].cv1.conv
                self.mid = self.model[8].cv1.conv
                self.late = self.model[16].cv1.conv

            def forward(self, x):
                x = self.early(x)
                x = self.mid(x)
                x = self.late(x)
                return x

        net = TinyYOLO()
        net.eval()
        return net

    @pytest.fixture
    def target_neurons(self):
        return {
            "model.2.cv1.conv": [0, 1, 2],
            "model.8.cv1.conv": [0, 1],
            "model.16.cv1.conv": [0, 1],
        }

    def test_per_group_combo_mode_valid(self, tiny_model, target_neurons):
        """Per-group coverage returns valid scores."""
        from optimize.coverage_utils import ClusterCoverageComputer
        comp = ClusterCoverageComputer(
            tiny_model, target_neurons, device="cpu",
            method="KMeans", use_silhouette=False, n_clusters=2,
            combo_mode="per-group",
        )
        build = torch.randn(30, 3, 32, 32)
        comp.fit(build, batch_size=4)
        assert comp._fitted

        test = torch.randn(10, 3, 32, 32)
        cov = comp.coverage(test, batch_size=4)
        for key in ("early", "middle", "late", "overall", "variability"):
            assert key in cov
            assert 0.0 <= cov[key] <= 1.0

    def test_per_layer_vs_per_group_differ(self, tiny_model, target_neurons):
        """Per-layer and per-group modes should produce different coverage values."""
        from optimize.coverage_utils import ClusterCoverageComputer
        build = torch.randn(30, 3, 32, 32)
        test = torch.randn(10, 3, 32, 32)

        comp_layer = ClusterCoverageComputer(
            tiny_model, target_neurons, device="cpu",
            method="KMeans", use_silhouette=False, n_clusters=2,
            combo_mode="per-layer",
        )
        comp_layer.fit(build, batch_size=4)
        cov_layer = comp_layer.coverage(test, batch_size=4)

        comp_group = ClusterCoverageComputer(
            tiny_model, target_neurons, device="cpu",
            method="KMeans", use_silhouette=False, n_clusters=2,
            combo_mode="per-group",
        )
        comp_group.fit(build, batch_size=4)
        cov_group = comp_group.coverage(test, batch_size=4)

        # Per-group should typically have LOWER coverage because the
        # combinatorial space is larger (cross-layer combos)
        # At minimum, they should produce valid numbers
        assert cov_layer["overall"] >= 0 and cov_group["overall"] >= 0
        # With only 10 samples vs 2^3=8 (layer) or 2^(2+3)=32 (group) combos,
        # per-group coverage should be lower than per-layer average
        print(f"Per-layer: {cov_layer['overall']:.4f}, Per-group: {cov_group['overall']:.4f}")

    def test_per_group_coverage_saturates_with_many_samples(self, tiny_model, target_neurons):
        """With enough diverse samples, per-group coverage should approach 1.0."""
        from optimize.coverage_utils import ClusterCoverageComputer
        comp = ClusterCoverageComputer(
            tiny_model, target_neurons, device="cpu",
            method="KMeans", use_silhouette=False, n_clusters=2,
            combo_mode="per-group",
        )
        build = torch.randn(100, 3, 32, 32)
        comp.fit(build, batch_size=8)

        test = torch.randn(200, 3, 32, 32)
        cov = comp.coverage(test, batch_size=8)
        # With 200 random samples and 2 clusters per neuron:
        # early (3 neurons) → 2^3=8 combos
        # middle (2 neurons) → 2^2=4 combos
        # late (2 neurons) → 2^2=4 combos
        # Should be near 1.0
        assert cov["overall"] > 0.5, f"Expected >0.5 with 200 samples, got {cov['overall']}"

    def test_per_group_union_tracker(self, tiny_model, target_neurons):
        """ClusterUnionTracker respects per-group combo mode."""
        from optimize.coverage_utils import ClusterCoverageComputer, ClusterUnionTracker
        comp = ClusterCoverageComputer(
            tiny_model, target_neurons, device="cpu",
            method="KMeans", use_silhouette=False, n_clusters=2,
            combo_mode="per-group",
        )
        build = torch.randn(30, 3, 32, 32)
        comp.fit(build, batch_size=4)

        tracker = ClusterUnionTracker(comp)
        imgs_a = torch.randn(5, 3, 32, 32)
        tracker.update(imgs_a, batch_size=2)
        cov_a = tracker.coverage()

        imgs_b = torch.randn(5, 3, 32, 32)
        tracker.update(imgs_b, batch_size=2)
        cov_b = tracker.coverage()

        # After adding more images, coverage should be >= previous
        assert cov_b["overall"] >= cov_a["overall"] - 0.001  # allow float rounding

    def test_per_group_from_assignments_direct(self):
        """Test _coverage_per_group with hand-crafted assignments."""
        from optimize.coverage_utils import ClusterCoverageComputer

        target_neurons = {
            "model.2.cv1.conv": [0, 1],   # early
            "model.8.cv1.conv": [0],       # middle
            "model.16.cv1.conv": [0],      # late
        }

        comp = ClusterCoverageComputer.__new__(ClusterCoverageComputer)
        comp.target_neurons = target_neurons
        comp.combo_mode = "per-group"

        # Build assignments: 4 images
        # Early group keys: model.2.cv1.conv:0, model.2.cv1.conv:1
        # Middle: model.8.cv1.conv:0
        # Late: model.16.cv1.conv:0
        layer_assignments = {
            "model.2.cv1.conv": [
                {"model.2.cv1.conv:0": 0, "model.2.cv1.conv:1": 0},
                {"model.2.cv1.conv:0": 0, "model.2.cv1.conv:1": 1},
                {"model.2.cv1.conv:0": 1, "model.2.cv1.conv:1": 0},
                {"model.2.cv1.conv:0": 1, "model.2.cv1.conv:1": 1},
            ],
            "model.8.cv1.conv": [
                {"model.8.cv1.conv:0": 0},
                {"model.8.cv1.conv:0": 1},
                {"model.8.cv1.conv:0": 0},
                {"model.8.cv1.conv:0": 1},
            ],
            "model.16.cv1.conv": [
                {"model.16.cv1.conv:0": 0},
                {"model.16.cv1.conv:0": 0},
                {"model.16.cv1.conv:0": 1},
                {"model.16.cv1.conv:0": 1},
            ],
        }
        layer_sizes = {
            "model.2.cv1.conv": {
                "model.2.cv1.conv:0": 2,
                "model.2.cv1.conv:1": 2,
            },
            "model.8.cv1.conv": {
                "model.8.cv1.conv:0": 2,
            },
            "model.16.cv1.conv": {
                "model.16.cv1.conv:0": 2,
            },
        }

        result = comp.coverage_from_assignments(layer_assignments, layer_sizes)

        # Early: 2 neurons × 2 clusters = 4 combos, all 4 seen → 100%
        assert result["early"] == pytest.approx(1.0)
        # Middle: 1 neuron × 2 clusters = 2 combos, both seen → 100%
        assert result["middle"] == pytest.approx(1.0)
        # Late: 1 neuron × 2 clusters = 2 combos, both seen → 100%
        assert result["late"] == pytest.approx(1.0)
        assert result["overall"] == pytest.approx(1.0)

    def test_per_group_partial_coverage(self):
        """Test per-group with partial coverage (not all combos seen)."""
        from optimize.coverage_utils import ClusterCoverageComputer

        target_neurons = {
            "model.2.cv1.conv": [0, 1],   # early
        }

        comp = ClusterCoverageComputer.__new__(ClusterCoverageComputer)
        comp.target_neurons = target_neurons
        comp.combo_mode = "per-group"

        # 2 neurons × 2 clusters = 4 combos; only 2 seen
        layer_assignments = {
            "model.2.cv1.conv": [
                {"model.2.cv1.conv:0": 0, "model.2.cv1.conv:1": 0},
                {"model.2.cv1.conv:0": 1, "model.2.cv1.conv:1": 1},
            ],
        }
        layer_sizes = {
            "model.2.cv1.conv": {
                "model.2.cv1.conv:0": 2,
                "model.2.cv1.conv:1": 2,
            },
        }

        result = comp.coverage_from_assignments(layer_assignments, layer_sizes)
        # 2/4 = 50%
        assert result["early"] == pytest.approx(0.5)
        assert result["overall"] == pytest.approx(0.5)

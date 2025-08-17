# src/wisdom.py
"""
WisdomIDC: memory-efficient IDC with pluggable clustering.

- Collects only selected neurons' activations (spatially averaged for conv)
- Clusters per neuron
- Silhouette search is pending, BO was used to replace it
- Models that lack `predict()` (e.g., Agglomerative, DBSCAN) are made usable at test time by computing and storing per-cluster centers.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import silhouette_score
from sklearn.cluster import estimate_bandwidth  # for MeanShift

from src.clustering import make
from src.utils import save_cluster_groups, load_cluster_groups


class DeepIDC:
    def __init__(
        self,
        model: nn.Module,
        top_m_neurons: int,
        n_clusters: int,
        use_silhouette: bool,
        test_all_classes: bool,
        clustering_method_name: str,
        device: str = "cpu",
        clustering_params: Dict[str, Any] | None = None,
        cache_path: str | None = None,
    ):
        self.model = model
        self.top_m_neurons = top_m_neurons
        self.use_silhouette = use_silhouette
        self.test_all_classes = test_all_classes
        self.total_combination = 1
        self.cache_path = cache_path
        self.n_clusters = n_clusters
        self.device = torch.device(device)

        self.clustering_method_name = clustering_method_name
        self.clustering_params = dict(clustering_params or {})

    def save_to_json(
        self,
        coverage_rate: float,
        max_coverage: float,
        model_name: str,
        testing_layer: str,
        file_path: str = "coverage_rate.json",
    ):
        data = {
            "Total Combination": self.total_combination,
            "Max Coverage": max_coverage,
            "Coverage Rate": coverage_rate,
            "Model Name": model_name,
            "Testing Layer": testing_layer,
        }
        with open(file_path, "w") as f:
            json.dump(data, f)

    def select_top_neurons(
        self,
        layer_scores: Dict[str, torch.Tensor],
        exclude_last: str | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Select top-k neurons across all layers based on (already aggregated) scores.
        Returns {layer_name: tensor(indices)}. If top_m_neurons == -1, selects all.
        """
        flattened: List[Tuple[str, float, int]] = []
        for name, scores in layer_scores.items():
            if exclude_last and name == exclude_last:
                continue
            if scores.dim() == 1:
                # linear: (C,)
                for i, s in enumerate(scores):
                    flattened.append((name, float(s), i))
            else:
                # conv: expect per-channel scores already (C,)
                # if given HxW maps, caller should have averaged first
                _scores = scores
                if _scores.dim() > 1:
                    # fallback: mean over spatial dims if needed
                    dims = tuple(range(1, _scores.dim()))
                    _scores = _scores.mean(dim=dims)
                for i, s in enumerate(_scores):
                    flattened.append((name, float(s), i))

        flattened.sort(key=lambda x: x[1], reverse=True)
        if self.top_m_neurons > 0:
            flattened = flattened[: self.top_m_neurons]

        selected: Dict[str, List[int]] = {}
        for lname, _, idx in flattened:
            selected.setdefault(lname, []).append(idx)
        return {lname: torch.tensor(idx_list) for lname, idx_list in selected.items()}

    def get_selected_activations(
        self,
        dataloader: DataLoader,
        selected_neurons: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self.model.to(self.device).eval()
        activation_lists: Dict[str, List[torch.Tensor]] = {n: [] for n in selected_neurons}

        def make_hook(layer_name: str):
            def hook_fn(module, inp, out):
                act = out.detach()
                if act.dim() > 2:
                    act = act.mean(dim=[2, 3])  # (B, C)
                idx = selected_neurons.get(layer_name)
                if idx is not None and idx.numel() > 0:
                    act = act[:, idx]
                activation_lists[layer_name].append(act.to("cpu", dtype=torch.float16))

            return hook_fn

        handles = []
        for name, module in self.model.named_modules():
            if name in selected_neurons:
                handles.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                _ = self.model(x)
                torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        activations: Dict[str, torch.Tensor] = {}
        for lname, chunks in activation_lists.items():
            activations[lname] = torch.cat(chunks, dim=0) if len(chunks) else torch.empty(0, 0)

        return activations

    # ---------------------- Clustering (per neuron) ----------------------

    def _fit_one_model(self, values: np.ndarray) -> Any:
        name = (self.clustering_method_name or "").lower()
        params = dict(self.clustering_params or {})

        if name == "kmeans":
            if self.use_silhouette:
                # sample to limit cost
                sample = values
                if len(sample) > 10000:
                    idx = np.random.choice(len(sample), 10000, replace=False)
                    sample = sample[idx]

                best_k = None
                best_score = -1.0
                # candidate ks
                uniq = np.unique(sample)
                max_k = min(10, max(2, uniq.shape[0]))
                for k in range(2, max_k + 1):
                    km_try = make("KMeans", **{**params, "n_clusters": k, "random_state": 42})
                    labels = km_try.fit_predict(sample)
                    # edge case: single cluster returned -> skip
                    if np.unique(labels).shape[0] < 2:
                        continue
                    score = silhouette_score(sample, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                if best_k is None:
                    best_k = self.n_clusters if self.n_clusters and self.n_clusters > 1 else 2
                params["n_clusters"] = best_k
            else:
                params.setdefault("n_clusters", max(2, int(self.n_clusters or 2)))

        elif name == "meanshift":
            # Auto bandwidth if not provided
            if "bandwidth" not in params or params["bandwidth"] in (None, 0):
                q = params.pop("quantile", 0.2)
                n_samp = params.pop("n_samples", 2000)
                bw = estimate_bandwidth(values, quantile=q, n_samples=min(n_samp, len(values)))
                # guard for degenerate cases
                if not np.isfinite(bw) or bw <= 0:
                    bw = None  # MeanShift will handle default
                params["bandwidth"] = bw

        # Create and fit
        model = make(self.clustering_method_name, **params)
        model.fit(values)

        # Ensure we can assign clusters at test time even if estimator lacks predict()
        if not hasattr(model, "predict"):
            # compute centers from training assignments
            if hasattr(model, "labels_"):
                labels = np.asarray(model.labels_)
                centers = []
                for c in np.unique(labels):
                    centers.append(values[labels == c].mean(axis=0))
                centers = np.stack(centers, axis=0) if len(centers) else np.zeros((0, 1), dtype=values.dtype)
                model._centers_ = centers  # (C, 1)
                model._n_clusters_fallback_ = centers.shape[0]
            else:
                # fallback: treat unique values as centers (very rare)
                uniq = np.unique(values)
                model._centers_ = uniq.reshape(-1, 1)
                model._n_clusters_fallback_ = uniq.shape[0]

        # Standardise n_clusters attribute for downstream total combination
        if hasattr(model, "n_clusters"):
            pass
        elif hasattr(model, "n_clusters_"):
            # sklearn set after fit
            model.n_clusters = int(model.n_clusters_)
        elif hasattr(model, "cluster_centers_"):
            model.n_clusters = int(model.cluster_centers_.shape[0])
        elif hasattr(model, "_n_clusters_fallback_"):
            model.n_clusters = int(model._n_clusters_fallback_)
        else:
            # last resort
            model.n_clusters = int(len(np.unique(getattr(model, "labels_", [0]))))

        return model

    def cluster_per_neuron(self, activations: Dict[str, torch.Tensor]) -> Dict[str, List[Any]]:
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"[INFO] Loading cached clusters from {self.cache_path}")
            return load_cluster_groups(self.cache_path)

        cluster_groups: Dict[str, List[Any]] = {}

        for layer_name, acts in activations.items():
            # acts: (N, K_l), float16 -> float32 for sklearn
            acts_np = acts.numpy().astype("float32")
            if acts_np.ndim != 2:
                # safety
                acts_np = acts_np.reshape(acts_np.shape[0], -1)

            per_neuron: List[Any] = []
            num_cols = acts_np.shape[1] if acts_np.size > 0 else 0
            for j in range(num_cols):
                values = acts_np[:, j:j + 1]  # (N,1)
                model = self._fit_one_model(values)
                # ensure `predict` available or usable
                per_neuron.append(model)
            cluster_groups[layer_name] = per_neuron

        if self.cache_path:
            save_cluster_groups(cluster_groups, self.cache_path)
            print(f"[INFO] Saved clusters to {self.cache_path}")

        return cluster_groups

    # ------------------------- Coverage computation -------------------------

    @staticmethod
    def _predict_one(model: Any, x_1d: np.ndarray) -> int:
        """Return cluster id for a single 1-D sample (shape (1, 1))."""
        if hasattr(model, "predict"):
            return int(model.predict(x_1d)[0])
        # fallback: nearest recorded center
        centers = getattr(model, "_centers_", None)
        if centers is None or len(centers) == 0:
            # degenerate: single cluster
            return 0
        # L2 distance since data is 1-D
        idx = int(np.argmin(np.abs(centers.reshape(-1) - x_1d.reshape(-1)[0])))
        return idx

    def compute_coverage(
        self,
        test_activations: Dict[str, torch.Tensor],
        cluster_groups: Dict[str, List[Any]],
        save_json: bool = True,
    ) -> Tuple[float, int, float]:
        """
        Build tuples of per-neuron cluster IDs for each sample and compute coverage:
            coverage = #unique tuples / total_possible_combinations
        """
        # total possible combinations
        total_comb = 1
        for _, models in cluster_groups.items():
            for m in models:
                total_comb *= int(getattr(m, "n_clusters", 1))

        # num samples from any layer tensor
        try:
            num_samples = next(iter(test_activations.values())).shape[0]
        except StopIteration:
            num_samples = 0

        seen: set[Tuple[int, ...]] = set()
        for i in range(num_samples):
            tup: List[int] = []
            for layer_name, models in cluster_groups.items():
                # (K_l,) activations for sample i
                a_row = test_activations[layer_name][i].numpy().astype("float32")
                for j, m in enumerate(models):
                    cid = self._predict_one(m, a_row[j:j + 1].reshape(1, -1))
                    tup.append(cid)
            seen.add(tuple(tup))

        unique = len(seen)
        max_coverage = 1.0 if num_samples > total_comb else (num_samples / total_comb if total_comb > 0 else 0.0)
        coverage_rate = (unique / total_comb) if total_comb > 0 else 0.0

        if save_json:
            model_name = self.model.__class__.__name__
            self.save_to_json(coverage_rate, max_coverage, model_name, "Whole model")

        self.total_combination = total_comb
        return coverage_rate, total_comb, max_coverage
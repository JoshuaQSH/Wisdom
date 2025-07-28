from typing import Dict, Any, List

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    DBSCAN,
    AgglomerativeClustering,
    MeanShift,
    SpectralClustering,
    OPTICS,
    Birch,
    AffinityPropagation,
    BisectingKMeans,
    HDBSCAN,
)
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import estimate_bandwidth


PREDICTLESS = {
    "AgglomerativeClustering",
    "SpectralClustering",
    "DBSCAN",
    "OPTICS",
    "HDBSCAN",
}

class PredictlessClusterWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, base_estimator, radius=None, nn_k=1):
        self.base = base_estimator
        self.radius = radius
        self.nn_k = nn_k

    def fit(self, X, y=None):
        X = np.asarray(X)

        # silence harmless OPTICS warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    category=RuntimeWarning,
                                    message="divide by zero encountered in divide",
                                    module="sklearn.cluster._optics")
            self.base.fit(X, y)

        self.labels_ = self.base.labels_

        # ─ representative points for NN search
        if hasattr(self.base, "core_sample_indices_"):            # DBSCAN / OPTICS
            rep_idx = self.base.core_sample_indices_
            reps = X[rep_idx]
            self.radius = self.radius or getattr(self.base, "eps", np.inf)
            rep_labels  = self.labels_[rep_idx]
        else:                                                     # agglom / spectral …
            reps = X
            rep_labels = self.labels_
            self.radius = np.inf

        self._nn = NearestNeighbors(n_neighbors=self.nn_k).fit(reps)
        self._rep_labels = rep_labels

        # ─ cluster centres & count
        self.cluster_centers_ = self._compute_cluster_centers(X)
        self.n_clusters = self.cluster_centers_.shape[0]

        return self

    def predict(self, X):
        dist, idx = self._nn.kneighbors(X, n_neighbors=1)
        labels = np.where(dist[:, 0] <= self.radius,
                          self._rep_labels[idx[:, 0]],
                          -1)
        return labels

    def _compute_cluster_centers(self, X):
        # Use native centroids if available
        if hasattr(self.base, "cluster_centers_"):
            return np.asarray(self.base.cluster_centers_)

        # Otherwise, mean of each label (excluding noise ‑1)
        unique = [lab for lab in np.unique(self.labels_) if lab != -1]
        if not unique:                                            # all noise
            return np.empty((0, X.shape[1]))
        centers = [X[self.labels_ == lab].mean(axis=0) for lab in unique]
        return np.vstack(centers)

    # delegate everything else
    def __getattr__(self, attr):
        return getattr(self.base, attr)

class MeanShiftAuto(MeanShift):
    def __init__(self, bandwidth=None, quantile=None, n_samples=None,
                 random_state=None, **kwargs):
        super().__init__(bandwidth=bandwidth, **kwargs)
        self.quantile = quantile
        self.n_samples = n_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        if (self.bandwidth is None) and (self.quantile is not None):
            self.bandwidth = estimate_bandwidth(
                X, quantile=self.quantile,
                n_samples=self.n_samples,
                random_state=self.random_state
            )
        return super().fit(X, y)

CLUSTERS: Dict[str, Dict[str, Any]] = {
    # ------------- 1. Partition / centroid methods ---------------------- #
    "KMeans": {
        "cls": KMeans,
        "space": [
            {"name": "n_clusters", "type": "range",  "bounds": [2, 30], "value_type": "int"},
            {"name": "init",       "type": "choice", "values": ["k-means++", "random"]},
            {"name": "max_iter",   "type": "range",  "bounds": [100, 800], "value_type": "int"},
        ],
    },
    "MiniBatchKMeans": {
        "cls": MiniBatchKMeans,
        "space": [
            {"name": "n_clusters", "type": "range",  "bounds": [2, 30], "value_type": "int"},
            {"name": "batch_size", "type": "range",  "bounds": [32, 2048], "value_type": "int"},
            {"name": "max_iter",   "type": "range",  "bounds": [100, 800], "value_type": "int"},
        ],
    },
    "BisectingKMeans": {
        "cls": BisectingKMeans,
        "space": [
            {"name": "n_clusters", "type": "range",  "bounds": [2, 30], "value_type": "int"},
            {"name": "init",       "type": "choice", "values": ["k-means++", "random"]},
            {"name": "max_iter",   "type": "range",  "bounds": [100, 800], "value_type": "int"},
        ],
    },

    # ---------------- 2. Hierarchical (sample-level) -------------------- #
    "AgglomerativeClustering": {
        "cls": AgglomerativeClustering,
        "space": [
            {"name": "n_clusters", "type": "range",  "bounds": [2, 30], "value_type": "int"},
            {"name": "linkage",    "type": "choice", "values": ["ward", "average", "complete", "single"]},
            {"name": "metric",     "type": "choice",
             "values": ["euclidean", "manhattan", "cosine", "l1", "l2"]},
        ],
    },

    # ---------------- 3. Spectral graph clustering --------------------- #
    "SpectralClustering": {
        "cls": SpectralClustering,
        "space": [
            {"name": "n_clusters",    "type": "range",  "bounds": [2, 30], "value_type": "int"},
            {"name": "affinity",      "type": "choice", "values": ["rbf", "nearest_neighbors"]},
            {"name": "assign_labels", "type": "choice", "values": ["kmeans", "discretize"]},
        ],
    },

    # ---------------- 4. Density-based --------------------------------- #
    "DBSCAN": {
        "cls": DBSCAN,
        "space": [
            {"name": "eps",         "type": "range",  "bounds": [0.05, 5.0]},
            {"name": "min_samples", "type": "range",  "bounds": [3, 50], "value_type": "int"},
            {"name": "metric",      "type": "choice", "values": ["euclidean", "manhattan", "chebyshev"]},
        ],
    },
    "OPTICS": {
        "cls": OPTICS,
        "space": [
            {"name": "min_samples",      "type": "range", "bounds": [3, 50], "value_type": "int"},
            {"name": "xi",               "type": "range", "bounds": [0.01, 0.3]},
            {"name": "min_cluster_size", "type": "range", "bounds": [2, 50], "value_type": "int"},
        ],
    },
     "HDBSCAN": {
        "cls": HDBSCAN,
        "space": [
            {"name": "min_cluster_size",          "type": "range", "bounds": [2, 50], "value_type": "int"},
            {"name": "min_samples",               "type": "range", "bounds": [1, 30], "value_type": "int"},
            {"name": "cluster_selection_epsilon", "type": "range", "bounds": [0.0, 1.0]},
        ],
    },

    # "MeanShift": {
    #     "cls": MeanShift,
    #     "space": [
    #         {"name": "bandwidth", "type": "range", "bounds": [0.1, 5.0]},
    #     ],
    # },

    "MeanShift": {
        "cls": MeanShiftAuto,
        "space": [
            {"name": "use_quantile", "type": "choice", "values": [0, 1]},
            {"name": "quantile",     "type": "range",  "bounds": [0.01, 0.9]},
            {"name": "n_samples",    "type": "range",  "bounds": [50, 2000], "value_type": "int"},
            {"name": "bandwidth",    "type": "range",  "bounds": [0.1, 5.0]},  # used when use_quantile=0
        ],
    },

    # ---------------- 5. Exemplar-based -------------------------------- #
    "AffinityPropagation": {
        "cls": AffinityPropagation,
        "space": [
            {"name": "damping",    "type": "range", "bounds": [0.5, 0.99]},
            {"name": "preference", "type": "range", "bounds": [-300, 0]},
        ],
    },

    # ---------------- 6. Constraint-based ------------------------------ #
    "Birch": {
        "cls": Birch,
        "space": [
            {"name": "threshold",   "type": "range", "bounds": [0.1, 2.0]},
            {"name": "n_clusters",  "type": "range", "bounds": [2, 30], "value_type": "int"},
        ],
    },
}


def names() -> List[str]:
    return list(CLUSTERS.keys())

def make(name: str, **kwargs):
    """
    Instantiate clusterer *only* with kwargs that are valid for that algorithm.
    """
    if name not in CLUSTERS:
        raise ValueError(f"Unknown clusterer '{name}'. Try: {names()}")

    if name == "MeanShift":
        use_quantile = kwargs.pop("use_quantile", 0)
        if use_quantile:
            kwargs["bandwidth"] = None

    cls = CLUSTERS[name]["cls"]
    valid = {k: v for k, v in kwargs.items() if k != "algo" and k in cls().__dict__ or k in cls.__init__.__code__.co_varnames}

    if name in PREDICTLESS:
        base_estimator = cls(**valid)
        cluster_model = PredictlessClusterWrapper(base_estimator=base_estimator)
    else:
        cluster_model = cls(**valid)
    return cluster_model
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, BisectingKMeans,
    AgglomerativeClustering, SpectralClustering,
    MeanShift, AffinityPropagation, Birch,
    DBSCAN, OPTICS,
)
from typing import Dict, Any
import numpy as np
from sklearn.datasets import make_blobs

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
    "MeanShift": {
        "cls": MeanShift,
        "space": [
            {"name": "bandwidth", "type": "range", "bounds": [0.1, 5.0]},
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


def names():
    return list(CLUSTERS.keys())


def make(name: str, **kwargs):
    """
    Instantiate clusterer *only* with kwargs that are valid for that algorithm.
    """
    if name not in CLUSTERS:
        raise ValueError(f"Unknown clusterer '{name}'. Try: {names()}")
    cls = CLUSTERS[name]["cls"]
    valid = {k: v for k, v in kwargs.items() if k != "algo" and k in cls().__dict__ or k in cls.__init__.__code__.co_varnames}
    return cls(**valid)

def midpoint(param):
    if param["type"] == "choice":
        return param["values"][0]
    lo, hi = param["bounds"]
    if param.get("value_type") == "int":
        return int((lo + hi) // 2)
    return 0.5 * (lo + hi)

def test_cluster_calling():
    # small toy dataset
    X, _ = make_blobs(n_samples=100, centers=5, n_features=2, random_state=0)
    breakpoint()
    for name, spec in CLUSTERS.items():
        # choose a deterministic config inside allowed bounds
        kwargs = {p["name"]: midpoint(p) for p in spec["space"]}
        kwargs["random_state"] = 0 if "random_state" in spec["cls"].__init__.__code__.co_varnames else None
        clusterer = make(name, **kwargs)
        labels = clusterer.fit_predict(X) if hasattr(clusterer, "fit_predict") else clusterer.fit(X).labels_
        print(f"{name:22s}  |  clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")

if __name__ == "__main__":
    test_cluster_calling()
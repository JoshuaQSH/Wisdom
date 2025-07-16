from typing import Dict, Any, List

import torch
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
    # "AgglomerativeClustering": {
    #     "cls": AgglomerativeClustering,
    #     "space": [
    #         {"name": "n_clusters", "type": "range",  "bounds": [2, 30], "value_type": "int"},
    #         {"name": "linkage",    "type": "choice", "values": ["ward", "average", "complete", "single"]},
    #         {"name": "metric",     "type": "choice",
    #          "values": ["euclidean", "manhattan", "cosine", "l1", "l2"]},
    #     ],
    # },

    # ---------------- 3. Spectral graph clustering --------------------- #
    # "SpectralClustering": {
    #     "cls": SpectralClustering,
    #     "space": [
    #         {"name": "n_clusters",    "type": "range",  "bounds": [2, 30], "value_type": "int"},
    #         {"name": "affinity",      "type": "choice", "values": ["rbf", "nearest_neighbors"]},
    #         {"name": "assign_labels", "type": "choice", "values": ["kmeans", "discretize"]},
    #     ],
    # },

    # ---------------- 4. Density-based --------------------------------- #
    # "DBSCAN": {
    #     "cls": DBSCAN,
    #     "space": [
    #         {"name": "eps",         "type": "range",  "bounds": [0.05, 5.0]},
    #         {"name": "min_samples", "type": "range",  "bounds": [3, 50], "value_type": "int"},
    #         {"name": "metric",      "type": "choice", "values": ["euclidean", "manhattan", "chebyshev"]},
    #     ],
    # },
    # "OPTICS": {
    #     "cls": OPTICS,
    #     "space": [
    #         {"name": "min_samples",      "type": "range", "bounds": [3, 50], "value_type": "int"},
    #         {"name": "xi",               "type": "range", "bounds": [0.01, 0.3]},
    #         {"name": "min_cluster_size", "type": "range", "bounds": [2, 50], "value_type": "int"},
    #     ],
    # },
    #  "HDBSCAN": {
    #     "cls": HDBSCAN,
    #     "space": [
    #         {"name": "min_cluster_size",          "type": "range", "bounds": [2, 50], "value_type": "int"},
    #         {"name": "min_samples",               "type": "range", "bounds": [1, 30], "value_type": "int"},
    #         {"name": "cluster_selection_epsilon", "type": "range", "bounds": [0.0, 1.0]},
    #     ],
    # },
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
    # "Birch": {
    #     "cls": Birch,
    #     "space": [
    #         {"name": "threshold",   "type": "range", "bounds": [0.1, 2.0]},
    #         {"name": "n_clusters",  "type": "range", "bounds": [2, 30], "value_type": "int"},
    #     ],
    # },
}


def names() -> List[str]:
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
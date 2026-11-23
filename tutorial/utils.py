"""Data splits, preprocessing, and artifact paths for the tabular WISDOM tutorial."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wisdom import ClusteringConfig, WisdomConfig, WisdomIDC
from wisdom.core.layers import discover_eligible_layers
from wisdom.core.wisdom import load_layerwise_top_neurons


DATASETS = ("education", "healthcare")
FEATURE_NAMES = ("minHr", "maxHr", "avgHr", "medianHr")
TARGET_NAME = "stress_index"
PARTICIPANT_COLUMN = "userId"
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 128
DEFAULT_TOP_K = 3
METHODS = ("ldl", "lig", "lgxa")
CLUSTER_METHOD = "KMeans"
CLUSTERS_PER_NEURON = 2
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetPaths:
    data: Path
    weights: Path
    scores: Path
    plot: Path
    coverage_csv: Path
    coverage_json: Path
    suite_csv: Path


def dataset_paths(dataset: str) -> DatasetPaths:
    if dataset not in DATASETS:
        raise ValueError(f"Choose one of {', '.join(DATASETS)}")
    return DatasetPaths(
        data=ROOT / "healthcare_datasets" / f"ai4work_{dataset}_completed.csv",
        weights=ROOT / "weights" / f"{dataset}_dnn.pt",
        scores=ROOT / "importances" / f"{dataset}_wisdom.csv",
        plot=ROOT / "importances" / f"{dataset}_neuron_importance.pdf",
        coverage_csv=ROOT / "results" / f"{dataset}_coverage.csv",
        coverage_json=ROOT / "results" / f"{dataset}_coverage.json",
        suite_csv=ROOT / "results" / f"{dataset}_suite_metrics.csv",
    )


@dataclass(frozen=True)
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    data_digest: str
    preprocessing: dict


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(
    contents: bytes, *, require_participants: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    required = [TARGET_NAME, *FEATURE_NAMES]
    frame = pd.read_csv(
        io.BytesIO(contents),
        usecols=lambda name: name in required or name == PARTICIPANT_COLUMN,
        dtype={PARTICIPANT_COLUMN: "string"},
    )
    if require_participants:
        required.append(PARTICIPANT_COLUMN)
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
    if require_participants:
        frame = frame.dropna(subset=[PARTICIPANT_COLUMN])
    features = frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float64, copy=True)
    targets = frame[TARGET_NAME].to_numpy(dtype=np.float64)
    features[~np.isfinite(features)] = np.nan
    usable = np.isfinite(targets) & ~np.isnan(features).all(axis=1)
    if not usable.any():
        raise ValueError("CSV has no rows with a finite target and heart-rate measurement")
    groups = frame[PARTICIPANT_COLUMN].to_numpy()[usable] if PARTICIPANT_COLUMN in frame else None
    return features[usable], targets[usable], groups


def prepare_data(dataset: str, seed: int, path: Path | None = None) -> PreparedData:
    """Hold out participants and fit the feature transforms on training rows."""
    contents = (path if path is not None else dataset_paths(dataset).data).read_bytes()
    features, targets, groups = _read_rows(contents, require_participants=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, test = next(splitter.split(features, groups=groups))

    if np.isnan(features[train]).all(axis=0).any():
        raise ValueError(f"Training split for {dataset} has an entirely missing feature")
    imputer = SimpleImputer(strategy="median")
    train_features = imputer.fit_transform(features[train])
    scaler = StandardScaler().fit(train_features)
    return PreparedData(
        x_train=scaler.transform(train_features).astype(np.float32),
        y_train=targets[train],
        x_test=scaler.transform(imputer.transform(features[test])).astype(np.float32),
        y_test=targets[test],
        feature_names=FEATURE_NAMES,
        data_digest=hashlib.sha256(contents).hexdigest(),
        preprocessing={
            "target": TARGET_NAME,
            "holdout_fraction": 0.2,
            "train_participants": int(len(np.unique(groups[train]))),
            "test_participants": int(len(np.unique(groups[test]))),
            "feature_medians": imputer.statistics_.tolist(),
            "feature_means": scaler.mean_.tolist(),
            "feature_scales": scaler.scale_.tolist(),
        },
    )


def load_test_data(path: Path, preprocessing: dict) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Apply the checkpoint's training statistics to a separate test CSV."""
    features, targets, groups = _read_rows(path.read_bytes(), require_participants=False)
    imputed = np.where(np.isnan(features), preprocessing["feature_medians"], features)
    scaled = (imputed - preprocessing["feature_means"]) / preprocessing["feature_scales"]
    participants = int(pd.Series(groups).nunique()) if groups is not None else None
    return scaled.astype(np.float32), targets, participants


def select_neurons(scores: Path, model: nn.Module, top_k: int = DEFAULT_TOP_K) -> dict[str, list[int]]:
    final_layer = discover_eligible_layers(model)[-1]
    selected = load_layerwise_top_neurons(str(scores), top_k, exclude_layers=(final_layer,))
    if len(selected) < 2:
        raise ValueError("Cross-layer coverage needs scored neurons in at least two hidden layers")
    return selected


def coverage_engine(model: nn.Module, seed: int) -> WisdomIDC:
    # Use one joint tuple across the selected layers.
    return WisdomIDC(
        model,
        cfg=WisdomConfig(selection_mode="global"),
        cluster=ClusteringConfig(
            method=CLUSTER_METHOD,
            params={"n_clusters": CLUSTERS_PER_NEURON, "random_state": seed},
        ),
    )


def restore_clusters(engine: WisdomIDC, selected: dict[str, list[int]], centers: dict) -> None:
    """Restore the selected neurons' training cluster centers for coverage."""
    groups = {}
    sizes = {}
    for layer, neurons in selected.items():
        groups[layer] = {}
        for index in neurons:
            if index not in centers.get(layer, {}):
                raise ValueError(f"The checkpoint has no clusters for neuron {layer}:{index}")
            values = np.asarray(centers[layer][index], dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 1 or not len(values) or not np.isfinite(values).all():
                raise ValueError(f"Invalid cluster centers for neuron {layer}:{index}")
            groups[layer][index] = {"centers": values}
            sizes[f"{layer}:{index}"] = len(values)
    engine.groups = groups
    engine.cluster_sizes = sizes


def tensor_loader(
    inputs: np.ndarray,
    targets: np.ndarray | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DataLoader:
    tensors = [torch.as_tensor(inputs, dtype=torch.float32)]
    if targets is not None:
        tensors.append(torch.as_tensor(targets, dtype=torch.float32))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)

"""Measure joint WISDOM activation coverage on the held-out tutorial rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from tutorial.dnn import QuantileDNN
from tutorial.utils import (
    CLUSTER_METHOD,
    CLUSTERS_PER_NEURON,
    DATASETS,
    DEFAULT_TOP_K,
    TARGET_NAME,
    coverage_engine,
    dataset_paths,
    file_digest,
    load_test_data,
    prepare_data,
    restore_clusters,
    select_neurons,
    tensor_loader,
)


def prediction_metrics(
    model: QuantileDNN, features: np.ndarray, targets: np.ndarray, alpha: float
) -> dict[str, float]:
    """Score midpoint errors and quantile loss in the original stress-index units."""
    with torch.inference_mode():
        predictions = model(torch.as_tensor(features, dtype=torch.float32)).numpy() * model.scale + model.center
    midpoint_errors = np.abs(targets - predictions.mean(axis=1))
    errors = targets[:, None] - predictions
    quantiles = np.array([alpha / 2, 1 - alpha / 2])
    return {
        "acc_at_5": float(np.mean(midpoint_errors <= 5)),
        "acc_at_10": float(np.mean(midpoint_errors <= 10)),
        "nominal_interval_coverage": 1 - alpha,
        "mean_interval_width": float(np.mean(np.abs(predictions[:, 1] - predictions[:, 0]))),
        "midpoint_mae": float(np.mean(midpoint_errors)),
        "mean_pinball_loss": float(np.maximum(quantiles * errors, (quantiles - 1) * errors).mean()),
    }


def write_csv(path: Path, row: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def print_table(rows: list[tuple[str, object]]) -> None:
    label_width = max(len(label) for label, _ in rows)
    lines = [f"{label:<{label_width}} : {value}" for label, value in rows]
    divider = "=" * max(72, max(map(len, lines)))
    print("\n".join(["", divider, "WISDOM Testing Summary", divider, *lines, divider]))


def preprocessing_matches(recreated: dict, checkpoint: dict) -> bool:
    # Read/scaling implementations can differ slightly in floating-point rounding.
    feature_stats = ("feature_medians", "feature_means", "feature_scales")
    return recreated.keys() == checkpoint.keys() and all(
        np.allclose(value, checkpoint[key], rtol=1e-8, atol=1e-10)
        if key in feature_stats else value == checkpoint[key]
        for key, value in recreated.items()
    )


def test_dataset(
    dataset: str | None,
    top_k: int,
    *,
    weights_path: Path | None = None,
    test_data_path: Path | None = None,
    wisdom_csv: Path | None = None,
) -> None:
    if weights_path is None:
        if dataset is None:
            raise ValueError("Choose a dataset or supply --weights-path.")
        weights_path = dataset_paths(dataset).weights
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights missing at {weights_path}. Run the pretraining script first.")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    model_dataset = checkpoint.get("dataset") if isinstance(checkpoint, dict) else None
    if model_dataset not in DATASETS:
        raise ValueError(f"Expected a tutorial checkpoint at {weights_path}.")
    if dataset is not None and dataset != model_dataset:
        raise ValueError(f"The checkpoint is for {model_dataset}, but the requested dataset is {dataset}.")
    dataset = model_dataset
    paths = dataset_paths(dataset)
    pretrain = f"uv run tutorial/run_wisdom_healthcare_pretrain.py --dataset {dataset}"
    scores_path = wisdom_csv or paths.scores
    if not scores_path.is_file():
        raise FileNotFoundError(f"Neuron scores missing at {scores_path}. Run {pretrain} first.")
    if checkpoint["scores_digest"] != file_digest(scores_path):
        raise ValueError(f"The neuron scores at {scores_path} do not match the checkpoint. Run {pretrain} to regenerate them.")

    test_path = test_data_path or paths.data
    use_original_split = file_digest(test_path) == checkpoint["data_digest"]
    if not use_original_split and test_path.resolve() == paths.data.resolve():
        raise ValueError(f"The original CSV at {test_path} has changed. Supply a held-out CSV or rerun {pretrain}.")

    # Recreate the participant holdout when given the full original CSV. Older
    # checkpoints also need that CSV to fit activation clusters on training rows.
    centers = checkpoint.get("cluster_centers")
    split = None
    if use_original_split or centers is None:
        source = test_path if use_original_split else paths.data
        split = prepare_data(dataset, checkpoint["seed"], source)
        if (
            checkpoint["data_digest"] != split.data_digest
            or checkpoint["feature_names"] != list(split.feature_names)
            or not preprocessing_matches(split.preprocessing, checkpoint["preprocessing"])
        ):
            raise ValueError(f"The original training split does not match the checkpoint. Rerun {pretrain}.")

    if use_original_split:
        assert split is not None
        test_features, test_targets = split.x_test, split.y_test
        test_participants = split.preprocessing["test_participants"]
        suite_name = "held-out participant split"
    else:
        test_features, test_targets, test_participants = load_test_data(test_path, checkpoint["preprocessing"])
        suite_name = "provided test CSV"

    model = QuantileDNN(
        len(checkpoint["feature_names"]), checkpoint["target_center"], checkpoint["target_scale"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Select within each layer, then evaluate one joint tuple across all selected layers.
    selected = select_neurons(scores_path, model, top_k)
    engine = coverage_engine(model, checkpoint["seed"])
    if centers is None:
        assert split is not None
        engine.fit_selected(tensor_loader(split.x_train), selected, device="cpu")
        train_rows = len(split.x_train)
    else:
        restore_clusters(engine, selected, centers)
        train_rows = checkpoint["train_rows"]
    coverage = engine.coverage_details(tensor_loader(test_features), selected, device="cpu")
    metrics = prediction_metrics(model, test_features, test_targets, checkpoint["alpha"])

    overall = coverage["scope_details"]["overall"]
    summary = {
        "dataset": dataset,
        "model": weights_path.name,
        "target": TARGET_NAME,
        "testing_mode": "cross-layer",
        "attribution": "WISDOM",
        "selection": "top-k per hidden layer",
        "top_k_per_layer": top_k,
        "selected_layers": len(selected),
        "train_rows": train_rows,
        "test_rows": len(test_features),
        "train_participants": checkpoint["preprocessing"]["train_participants"],
        "test_participants": test_participants,
        "cluster_method": CLUSTER_METHOD,
        "requested_clusters_per_neuron": CLUSTERS_PER_NEURON,
        **overall,
        "acc_at_5": metrics["acc_at_5"],
        "acc_at_10": metrics["acc_at_10"],
    }
    report = {
        **summary,
        "suite": suite_name,
        "weights_path": str(weights_path),
        "test_data_path": str(test_path),
        "wisdom_csv": str(scores_path),
        "seed": checkpoint["seed"],
        "feature_names": checkpoint["feature_names"],
        "consensus_methods": checkpoint["methods"],
        "voting_mode": checkpoint["voting_mode"],
        "selected_neurons": selected,
        "scope_details": coverage["scope_details"],
        "prediction_metrics": metrics,
    }
    paths.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=paths.coverage_csv.parent) as directory:
        staging = Path(directory)
        write_csv(staging / paths.coverage_csv.name, summary)
        write_csv(staging / paths.suite_csv.name, {"suite": suite_name, **summary, **metrics})
        (staging / paths.coverage_json.name).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for destination in (paths.coverage_csv, paths.suite_csv, paths.coverage_json):
            (staging / destination.name).replace(destination)

    selection = f"Top {top_k} per hidden layer, {overall['monitored_neurons']} across {len(selected)} layers"
    participant_count = test_participants if test_participants is not None else "n/a"
    print_table([
        ("Model Name", weights_path.name),
        ("Dataset", dataset),
        ("Target", TARGET_NAME),
        ("Testing Mode", "Cross-layer"),
        ("Test suite", suite_name),
        ("Neuron selection", selection),
        ("Attribution", summary["attribution"]),
        ("Train / test participants", f"{summary['train_participants']} / {participant_count}"),
        ("Train / test rows", f"{summary['train_rows']:,} / {summary['test_rows']:,}"),
        ("Total combinations", overall["total_combinations"]),
        ("Max coverage for suite", f"{overall['max_coverage']:.6f}"),
        ("WISDOM coverage", f"{overall['coverage_rate']:.6f}"),
        ("Acc@5", f"{metrics['acc_at_5']:.2%}"),
        ("Acc@10", f"{metrics['acc_at_10']:.2%}"),
        ("Mean interval width", f"{metrics['mean_interval_width']:.4f}"),
        ("Midpoint MAE", f"{metrics['midpoint_mae']:.4f}"),
        ("Consensus methods", ", ".join(checkpoint["methods"])),
        ("Voting mode", checkpoint["voting_mode"]),
        ("Coverage CSV", paths.coverage_csv),
        ("Coverage JSON", paths.coverage_json),
        ("Test suite CSV", paths.suite_csv),
    ])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("use_case", nargs="?", choices=(*DATASETS, "all"), help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), help="dataset to evaluate (default all)")
    parser.add_argument("--weights-path", type=Path, help="DNN checkpoint from the pretraining script")
    parser.add_argument("--test-data-path", type=Path, help="original dataset CSV or a separate held-out CSV")
    parser.add_argument("--wisdom-csv", type=Path, help="WISDOM neuron scores from pretraining")
    parser.add_argument("--top-k", type=int, choices=range(1, DEFAULT_TOP_K + 1), default=DEFAULT_TOP_K,
                        help="maximum number of scored neurons to keep per hidden layer")
    args = parser.parse_args(argv)

    if args.dataset and args.use_case and args.dataset != args.use_case:
        parser.error("The positional dataset and --dataset must agree.")
    dataset = args.dataset or args.use_case
    explicit_paths = args.weights_path or args.test_data_path or args.wisdom_csv
    if explicit_paths and (dataset == "all" or (dataset is None and args.weights_path is None)):
        parser.error("Choose one dataset for explicit paths, or let --weights-path identify the dataset.")
    datasets = DATASETS if dataset == "all" or (dataset is None and not explicit_paths) else (dataset,)
    torch.set_num_threads(1)
    for dataset in datasets:
        try:
            test_dataset(
                dataset,
                args.top_k,
                weights_path=args.weights_path,
                test_data_path=args.test_data_path,
                wisdom_csv=args.wisdom_csv,
            )
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

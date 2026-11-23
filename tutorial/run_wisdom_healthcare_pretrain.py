"""Train the tutorial quantile DNN and score its hidden neurons with WISDOM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisdom import ConsensusWisdom, WisdomTrainConfig
from wisdom.core.layers import discover_eligible_layers
from wisdom.core.task import TaskBatch
from wisdom.utils.visualization import viz_topk_neurons_score

from tutorial.dnn import QuantileDNN, fit_standard
from tutorial.utils import (
    DATASETS,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    METHODS,
    coverage_engine,
    dataset_paths,
    file_digest,
    prepare_data,
    select_neurons,
    tensor_loader,
)


class QuantileAdapter:
    """Connect the tabular DNN's quantile outputs to WISDOM attribution."""

    def __init__(self, model: QuantileDNN, alpha: float) -> None:
        self.analysis_model = model
        self.quantiles = torch.tensor([alpha / 2, 1 - alpha / 2])

    def prepare_batch(self, batch, device: str) -> TaskBatch:
        inputs, targets = batch
        return TaskBatch(inputs.to(device), targets.to(device))

    @torch.no_grad()
    def captum_target(self, batch: TaskBatch) -> torch.Tensor:
        # Attribute the lower or upper output according to the target's tail.
        targets = cast(torch.Tensor, batch.targets)
        midpoint = self.analysis_model(batch.inputs).mean(dim=1)
        return (targets >= midpoint).long()

    def excluded_layer_names(self) -> tuple[str, ...]:
        return discover_eligible_layers(self.analysis_model)[-1:]

    def excluded_layer_prefixes(self) -> tuple[str, ...]:
        return ()

    def capture_reference(self, batch: TaskBatch) -> None:
        return None

    def loss(self, batch: TaskBatch, reference: object | None = None) -> torch.Tensor:
        targets = cast(torch.Tensor, batch.targets)
        predictions = self.analysis_model(batch.inputs)
        quantiles = self.quantiles.to(predictions)
        error = targets[:, None] - predictions
        return torch.maximum(quantiles * error, (quantiles - 1) * error).mean()


def pretrain(
    dataset: str, *, epochs: int, alpha: float, seed: int, data_path: Path | None = None
) -> None:
    paths = dataset_paths(dataset)
    data = prepare_data(dataset, seed, path=data_path)
    print(f"Training {dataset} DNN on {len(data.y_train):,} rows for {epochs} epochs.", flush=True)
    model, training = fit_standard(
        data.x_train, data.y_train, seed=seed, epochs=epochs, alpha=alpha
    )
    center, scale = training["target_center"], training["target_scale"]
    normalized_targets = (data.y_train - center) / scale
    train_loader = tensor_loader(data.x_train, normalized_targets)
    trainer = ConsensusWisdom(model, device="cpu", task_adapter=QuantileAdapter(model, alpha))

    paths.weights.parent.mkdir(parents=True, exist_ok=True)
    paths.scores.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{dataset}-", dir=paths.weights.parent) as staging:
        scores = Path(staging) / paths.scores.name
        plot = Path(staging) / paths.plot.name
        weights = Path(staging) / paths.weights.name
        config = WisdomTrainConfig(
            methods=list(METHODS),
            device="cpu",
            selection_mode="per-layer",
            out_csv=str(scores),
        )
        print(f"Scoring hidden neurons with WISDOM ({', '.join(METHODS)}).", flush=True)
        trainer.fit(train_loader, config, top_m_neurons=DEFAULT_TOP_K)
        selected = select_neurons(scores, model)
        engine = coverage_engine(model, seed)
        engine.fit_selected(train_loader, selected, device="cpu")
        cluster_centers = {
            layer: {index: info["centers"].tolist() for index, info in neurons.items()}
            for layer, neurons in engine.groups.items()
        }
        viz_topk_neurons_score(scores, top_k=10, output_path=plot)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "dataset": dataset,
                "seed": seed,
                "epochs": epochs,
                "alpha": alpha,
                "target_center": center,
                "target_scale": scale,
                "feature_names": list(data.feature_names),
                "data_digest": data.data_digest,
                "train_rows": len(data.y_train),
                "preprocessing": data.preprocessing,
                "cluster_centers": cluster_centers,
                "methods": list(METHODS),
                "voting_mode": config.voting_mode,
                "scores_digest": file_digest(scores),
            },
            weights,
        )
        scores.replace(paths.scores)
        plot.replace(paths.plot)
        weights.replace(paths.weights)

    print(f"Training pinball loss  {training['final_pinball']:.6f}")
    print(f"DNN weights            {paths.weights}")
    print(f"WISDOM scores          {paths.scores}")
    print(f"Importance plot        {paths.plot}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = (*DATASETS, "all")
    parser.add_argument("dataset_name", metavar="dataset", choices=choices, nargs="?")
    parser.add_argument("--dataset", choices=choices, help="dataset to train (default: all)")
    parser.add_argument("--data-path", type=Path, help="source CSV override for a single dataset")
    parser.add_argument("--epochs", type=int, default=200, help="DNN epochs (default 200)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Quantile interval miscoverage (default 0.1)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Split and DNN seed (default 42)")
    args = parser.parse_args(argv)
    if args.dataset_name and args.dataset and args.dataset_name != args.dataset:
        parser.error("the positional dataset and --dataset must agree")
    chosen = args.dataset or args.dataset_name or "all"
    if args.data_path and chosen == "all":
        parser.error("--data-path requires a single dataset")
    torch.set_num_threads(1)
    try:
        for dataset in DATASETS if chosen == "all" else (chosen,):
            pretrain(
                dataset, epochs=args.epochs, alpha=args.alpha, seed=args.seed, data_path=args.data_path
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

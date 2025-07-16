"""
Bayesian optimisation of corr(coverage, F1)  with BoTorch .

Usage:
from search import BOSearch
searcher = BOSearch(csv_file, train_loader, model, idc_cfg)
best_cfg = searcher.optimize(n_trials=40, init_points=8)
"""
from __future__ import annotations
from typing import Dict, Any, List

import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from scipy.stats import pearsonr

from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from .clustering import CLUSTERS, make
from .idc import IDC
# from .utils import load_important_neurons_from_csv


def _unique_hparams() -> List[str]:
    """Collect the superset of hyper‑params across all algorithms."""
    keys = set()
    for spec in CLUSTERS.values():
        keys.update(p["name"] for p in spec["space"])
    return sorted(keys)


_ALL_PARAMS = _unique_hparams()                # length = d‑1  (first dim = algo)
_CAT_PARAMS = {                               # categorical hyper‑params
    "algo",
    "init", "linkage", "affinity",
}
# continuous/int range params will be treated as continuous and rounded later

class BOSearch:
    def __init__(
        self,
        csv_file: str,
        train_loader,
        model,
        idc_cfg: Dict[str, Any],
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.csv = csv_file
        self.train_loader = train_loader
        self.model = model.to(device).eval()
        self.idc_cfg = idc_cfg
        self.device = device
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # ---- space description ------------------------------------------- #
        self.algos = list(CLUSTERS.keys())
        self.d = 1 + len(_ALL_PARAMS)                # algo + every hyper‑param
        self.bounds = torch.stack([torch.zeros(self.d), torch.ones(self.d)]).double()
        self.cat_dims = [0] + [
            1 + _ALL_PARAMS.index(p) for p in _ALL_PARAMS if p in _CAT_PARAMS
        ]

        self.X, self.Y = None, None   # will hold evaluated points / objectives

        # cache once for speed
        self._labels, self._f1_of_cls = self._cache_f1()
    
    def _sample_loader(self, num_splits: int = 3, batch_size: int = 32):
        dataset = self.train_loader.dataset
        split_sizes = [len(dataset) // num_splits] * num_splits
        for i in range(len(dataset) % num_splits):
            split_sizes[i] += 1
        subsets = random_split(dataset, split_sizes)
        subset_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in subsets
        ]
        return subset_loaders


    # ===================================================================== #
    #            encoding / decoding between tensors and dicts
    # ===================================================================== #
    def _tensor_to_cfg(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Decode a point in [0,1]^d to a configuration dict.
        """
        x = x.double()
        # --- algorithm --------------------------------------------------- #
        algo_idx = int(torch.round(x[0] * (len(self.algos) - 1)).item())
        cfg = {"algo": self.algos[algo_idx]}

        # --- hyper‑params ------------------------------------------------- #
        for j, name in enumerate(_ALL_PARAMS, start=1):
            v = x[j].item()
            space_union = {}
            # find this param definition in any algo (all share same scale 0‑1)
            for spec in CLUSTERS.values():
                for p in spec["space"]:
                    if p["name"] == name:
                        space_union = p
                        break
                if space_union:
                    break

            if not space_union:        # should not happen
                continue

            if space_union["type"] == "choice":
                # idx = int(torch.round(v * (len(space_union["values"]) - 1)))
                idx = int(round(v * (len(space_union["values"]) - 1)))
                cfg[name] = space_union["values"][idx]
            else:                      # "range"
                lo, hi = space_union["bounds"]
                if space_union.get("value_type") == "int":
                    cfg[name] = int(round(lo + v * (hi - lo)))
                else:
                    cfg[name] = float(lo + v * (hi - lo))

        # remove params that the chosen algo does not accept
        allowed = {p["name"] for p in CLUSTERS[cfg["algo"]]["space"]}
        cfg = {k: v for k, v in cfg.items() if k == "algo" or k in allowed}
        return cfg

    # --------------------------------------------------------------------- #
    def _cfg_to_tensor(self, cfg: Dict[str, Any]) -> torch.Tensor:
        """
        Encode a cfg dict back to [0,1]^d .  (Used only when logging best.)
        """
        x = torch.zeros(self.d, dtype=torch.double)
        x[0] = self.algos.index(cfg["algo"]) / (len(self.algos) - 1)

        for j, name in enumerate(_ALL_PARAMS, 1):
            if name not in cfg:
                continue
            space_union = None
            for spec in CLUSTERS.values():
                for p in spec["space"]:
                    if p["name"] == name:
                        space_union = p
                        break
                if space_union:
                    break
            if space_union["type"] == "choice":
                idx = space_union["values"].index(cfg[name])
                x[j] = idx / (len(space_union["values"]) - 1)
            else:
                lo, hi = space_union["bounds"]
                x[j] = (cfg[name] - lo) / (hi - lo)
        return x

    # ===================================================================== #
    #                      objective  f(x)  (scalar)
    # ===================================================================== #
    def _objective(self, cfg: Dict[str, Any]) -> float:
        topk = self.idc_cfg["top_m_neurons"]
        df = pd.read_csv(self.csv)
        df_sorted = df.sort_values(by='Score', ascending=False).head(topk)
        imp_neurons = {}
        for layer_name, group in df_sorted.groupby('LayerName'):
            imp_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)

        idc = IDC(
            self.model,
            top_m_neurons=topk,
            n_clusters=self.idc_cfg["n_clusters"],
            clustering_method_name=cfg["algo"],
            clustering_params={k: v for k, v in cfg.items() if k != "algo"},
            use_silhouette=False,
            test_all_classes=True,
            cache_path=None,
        )

        sample_dataloader = self._sample_loader(num_splits=30, batch_size=32)

        # ---- activations + clusters ------------------------------------- #
        _, acts = idc.get_activations_model_dataloader(self.train_loader, imp_neurons)
        acts = {k: v.half().cpu() for k, v in acts.items()}
        clusters = idc.cluster_activation_values_all(acts)
        coverage_rate = []
        f1 = []
        for i, loader in enumerate(sample_dataloader):
            # coverage per sample (batch loop for memory safety)
            with torch.no_grad():
                coverage_, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(loader, imp_neurons, clusters)
                acc, loss, f1_ = self._eval_model_dataloder(loader)
            coverage_rate.append(coverage_)
            f1.append(f1_)
        
        corr_score = pearsonr(coverage_rate, f1).statistic
        if np.isnan(corr_score):
            print(f"[BoTorch] Warning: NaN corr for cfg={cfg}  (coverage_rate={coverage_rate}, f1={f1})")
            corr_score = 0.0
        
        return corr_score

    # ===================================================================== #
    #                           helper utilities
    # ===================================================================== #
    def _cache_f1(self):
        from sklearn.metrics import f1_score

        preds, labs = [], []
        with torch.no_grad():
            for x, y in self.train_loader:
                out = self.model(x.to(self.device))
                preds.append(out.argmax(1).cpu())
                labs.append(y.cpu())
        preds = torch.cat(preds).numpy()
        labs = torch.cat(labs).numpy()
        f1 = f1_score(labs, preds, average=None)
        return labs, {i: float(f1[i]) for i in range(len(f1))}
    
    def _eval_model_dataloder(self, dataloader):
        from sklearn.metrics import f1_score
        
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)

                # Store labels and predictions for metric computation
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Compute average loss
        avg_loss = running_loss / len(dataloader.dataset)

        # Compute accuracy
        correct_predictions = sum(p == t for p, t in zip(all_preds, all_labels))
        accuracy = correct_predictions / len(all_labels)

        # Compute F1 score
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return accuracy, avg_loss, f1

    # ===================================================================== #
    #                             optimisation loop
    # ===================================================================== #
    def optimize(self, n_trials: int = 40, init_points: int = 8) -> Dict[str, Any]:
        """
        Run BO and return the best configuration dict.
        """
        # 1. Sobol initial design ----------------------------------------- #
        sobol = draw_sobol_samples(self.bounds, n=init_points, q=1).squeeze(1)
        X = sobol.clone()
        Y = torch.tensor(
            [[self._objective(self._tensor_to_cfg(x))] for x in X],
            dtype=torch.double,
        )

        # 2. BO iterations ------------------------------------------------- #
        while X.shape[0] < n_trials:
            # --- fit GP -------------------------------------------------- #
            model = MixedSingleTaskGP(
                X, Y, cat_dims=self.cat_dims
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # --- acquisition ------------------------------------------- #
            ei = LogExpectedImprovement(model, best_f=Y.max())
            cand, _ = optimize_acqf(
                acq_function=ei,
                bounds=self.bounds,
                q=1,
                num_restarts=5,
                raw_samples=64,
            )
            new_y = torch.tensor(
                [[self._objective(self._tensor_to_cfg(cand.squeeze(0)))]],
                dtype=torch.double,
            )
            X = torch.cat([X, cand])
            Y = torch.cat([Y, new_y])

        # 3. pick best ----------------------------------------------------- #
        best_idx = torch.argmax(Y).item()
        best_cfg = self._tensor_to_cfg(X[best_idx])
        best_val = Y[best_idx].item()
        print(f"[BoTorch]  best corr = {best_val:.4f}  ->  {best_cfg}")
        return best_cfg
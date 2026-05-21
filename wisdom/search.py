"""Bayesian-style hyperparameter search for WISDOM coverage workflows.

The cleaned core prefers a PyTorch-native BoTorch backend when it is available,
but keeps a scikit-learn GP fallback so BO still works in lighter environments.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


ObjectiveFn = Callable[[Dict[str, Any]], float]


@dataclass
class SearchResult:
    best_config: Dict[str, Any]
    best_score: float
    history: List[tuple[Dict[str, Any], float]]
    backend: str = 'sklearn'


class BOSearch:
    """A small BO helper for clustering/search spaces.

    Backends:
      - ``botorch``: better mixed-space BO when BoTorch / GPyTorch are available
      - ``sklearn``: dependency-light Gaussian-process fallback
      - ``auto``: prefer BoTorch, otherwise fall back to sklearn
    """

    def __init__(self, search_space: Dict[str, Any], random_state: int = 42, backend: str = 'auto'):
        if not search_space:
            raise ValueError('search_space must not be empty')
        self.search_space = dict(search_space)
        self.random_state = int(random_state)
        self.backend = str(backend)
        self._rng = np.random.default_rng(self.random_state)

    def _sample_value(self, spec: Any):
        if isinstance(spec, list):
            return spec[int(self._rng.integers(0, len(spec)))]
        if isinstance(spec, tuple):
            if len(spec) == 2:
                low, high = spec
                return float(self._rng.uniform(float(low), float(high)))
            if len(spec) == 3:
                low, high, kind = spec
                if str(kind).lower() in {'int', 'integer'}:
                    return int(self._rng.integers(int(low), int(high) + 1))
                return float(self._rng.uniform(float(low), float(high)))
        raise TypeError(f'Unsupported search space spec: {spec!r}')

    def sample_config(self) -> Dict[str, Any]:
        return {name: self._sample_value(spec) for name, spec in self.search_space.items()}

    def _config_key(self, config: Dict[str, Any]) -> tuple:
        return tuple((key, config[key]) for key in sorted(config))

    def _encode_value(self, spec: Any, value: Any) -> float:
        if isinstance(spec, list):
            if len(spec) == 1:
                return 0.0
            idx = spec.index(value)
            return float(idx) / float(len(spec) - 1)
        if isinstance(spec, tuple):
            low, high = float(spec[0]), float(spec[1])
            if high == low:
                return 0.0
            return (float(value) - low) / (high - low)
        raise TypeError(f'Unsupported search space spec: {spec!r}')

    def _encode_config(self, config: Dict[str, Any]) -> np.ndarray:
        return np.array(
            [self._encode_value(self.search_space[name], config[name]) for name in sorted(self.search_space)],
            dtype=np.float64,
        )

    def _decode_value(self, spec: Any, encoded: float):
        encoded = float(np.clip(encoded, 0.0, 1.0))
        if isinstance(spec, list):
            if len(spec) == 1:
                return spec[0]
            idx = int(round(encoded * float(len(spec) - 1)))
            idx = max(0, min(idx, len(spec) - 1))
            return spec[idx]
        if isinstance(spec, tuple):
            low, high = float(spec[0]), float(spec[1])
            raw = low + encoded * (high - low)
            if len(spec) == 3 and str(spec[2]).lower() in {'int', 'integer'}:
                return int(round(raw))
            return float(raw)
        raise TypeError(f'Unsupported search space spec: {spec!r}')

    def _decode_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        names = sorted(self.search_space)
        if vector.shape[0] != len(names):
            raise ValueError(f'Expected {len(names)} encoded dims, got {vector.shape[0]}')
        return {
            name: self._decode_value(self.search_space[name], vector[idx])
            for idx, name in enumerate(names)
        }

    def _categorical_dims(self) -> List[int]:
        return [idx for idx, name in enumerate(sorted(self.search_space)) if isinstance(self.search_space[name], list)]

    def _discrete_values(self, spec: Any) -> List[Any] | None:
        if isinstance(spec, list):
            return list(spec)
        if isinstance(spec, tuple) and len(spec) == 3 and str(spec[2]).lower() in {'int', 'integer'}:
            low, high = int(spec[0]), int(spec[1])
            return list(range(low, high + 1))
        return None

    def _enumerate_finite_configs(self, max_configs: int = 10000) -> List[Dict[str, Any]] | None:
        names = sorted(self.search_space)
        value_lists: List[List[Any]] = []
        total = 1
        for name in names:
            values = self._discrete_values(self.search_space[name])
            if values is None:
                return None
            total *= len(values)
            if total > max_configs:
                return None
            value_lists.append(values)
        return [
            {name: value for name, value in zip(names, combo)}
            for combo in product(*value_lists)
        ]

    def _resolve_backend(self, requested: str | None = None) -> str:
        backend = str(requested or self.backend).lower()
        if backend not in {'auto', 'sklearn', 'botorch'}:
            raise ValueError(f'Unsupported BO backend: {backend}')
        if backend == 'sklearn':
            return 'sklearn'
        try:
            import botorch  # noqa: F401
            import gpytorch  # noqa: F401
        except Exception:
            if backend == 'botorch':
                raise
            return 'sklearn'
        return 'botorch'

    def _random_unseen_config(self, seen: set[tuple]) -> Dict[str, Any]:
        for _ in range(1024):
            cfg = self.sample_config()
            if self._config_key(cfg) not in seen:
                return cfg
        return self.sample_config()

    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        erf = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

    def _expected_improvement(self, mean: np.ndarray, std: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
        safe_std = np.where(std <= 1e-12, 1e-12, std)
        improvement = mean - best - xi
        z = improvement / safe_std
        ei = improvement * self._norm_cdf(z) + safe_std * self._norm_pdf(z)
        ei = np.where(std <= 1e-12, 0.0, ei)
        return ei

    def _optimize_sklearn(
        self,
        objective: ObjectiveFn,
        n_init: int,
        n_iter: int,
        candidate_pool_size: int,
    ) -> SearchResult:
        seen = set()
        history: List[tuple[Dict[str, Any], float]] = []
        finite_configs = self._enumerate_finite_configs(max_configs=max(10000, candidate_pool_size * 32))

        def evaluate(config: Dict[str, Any]) -> float:
            score = float(objective(config))
            history.append((dict(config), score))
            seen.add(self._config_key(config))
            return score

        for _ in range(max(1, n_init)):
            if finite_configs is not None and len(seen) >= len(finite_configs):
                break
            evaluate(self._random_unseen_config(seen))

        for _ in range(max(0, n_iter)):
            if finite_configs is not None:
                candidates = [cfg for cfg in finite_configs if self._config_key(cfg) not in seen]
                if not candidates:
                    break
            else:
                candidates = []
                candidate_keys = set()
                target = max(16, candidate_pool_size)
                max_attempts = max(target * 32, 256)
                attempts = 0
                while len(candidates) < target and attempts < max_attempts:
                    cfg = self._random_unseen_config(seen)
                    key = self._config_key(cfg)
                    attempts += 1
                    if key in seen or key in candidate_keys:
                        continue
                    candidate_keys.add(key)
                    candidates.append(cfg)
                if not candidates:
                    break

            x_train = np.vstack([self._encode_config(cfg) for cfg, _ in history])
            y_train = np.array([score for _, score in history], dtype=np.float64)

            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(x_train.shape[1]), nu=2.5)
            kernel += WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=self.random_state)

            try:
                gp.fit(x_train, y_train)
            except Exception:
                if finite_configs is not None:
                    remaining = [cfg for cfg in finite_configs if self._config_key(cfg) not in seen]
                    if not remaining:
                        break
                evaluate(self._random_unseen_config(seen))
                continue

            x_candidates = np.vstack([self._encode_config(cfg) for cfg in candidates])
            mean, std = gp.predict(x_candidates, return_std=True)
            best_observed = float(np.max(y_train))
            ei = self._expected_improvement(mean, std, best_observed)
            next_cfg = candidates[int(np.argmax(ei))]
            evaluate(next_cfg)

        best_config, best_score = max(history, key=lambda item: item[1])
        return SearchResult(best_config=dict(best_config), best_score=float(best_score), history=history, backend='sklearn')

    def _optimize_botorch(
        self,
        objective: ObjectiveFn,
        n_init: int,
        n_iter: int,
    ) -> SearchResult:
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import MixedSingleTaskGP, SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.optim import optimize_acqf
        from botorch.utils.sampling import draw_sobol_samples
        from gpytorch.mlls import ExactMarginalLogLikelihood

        try:
            from botorch.acquisition.analytic import LogExpectedImprovement as AcquisitionCls
        except Exception:  # pragma: no cover - older / alternate botorch APIs
            from botorch.acquisition.analytic import ExpectedImprovement as AcquisitionCls

        seen = set()
        history: List[tuple[Dict[str, Any], float]] = []

        def evaluate(config: Dict[str, Any]) -> float:
            score = float(objective(config))
            history.append((dict(config), score))
            seen.add(self._config_key(config))
            return score

        dim = len(self.search_space)
        bounds = torch.stack(
            [torch.zeros(dim, dtype=torch.double), torch.ones(dim, dtype=torch.double)],
        )
        sobol = draw_sobol_samples(bounds=bounds, n=max(4, n_init * 4), q=1).squeeze(1)
        for x in sobol:
            cfg = self._decode_vector(x.detach().cpu().numpy())
            if self._config_key(cfg) in seen:
                continue
            evaluate(cfg)
            if len(history) >= max(1, n_init):
                break
        while len(history) < max(1, n_init):
            evaluate(self._random_unseen_config(seen))

        cat_dims = self._categorical_dims()
        for _ in range(max(0, n_iter)):
            x_train = torch.tensor(
                [self._encode_config(cfg).tolist() for cfg, _ in history],
                dtype=torch.double,
            )
            y_train = torch.tensor([[score] for _, score in history], dtype=torch.double)

            try:
                if cat_dims:
                    gp = MixedSingleTaskGP(x_train, y_train, cat_dims=cat_dims)
                else:
                    gp = SingleTaskGP(
                        x_train,
                        y_train,
                        input_transform=Normalize(d=x_train.shape[1]),
                        outcome_transform=Standardize(m=1),
                    )
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                acquisition = AcquisitionCls(gp, best_f=y_train.max())
                candidate, _ = optimize_acqf(
                    acq_function=acquisition,
                    bounds=bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                )
                next_cfg = self._decode_vector(candidate.squeeze(0).detach().cpu().numpy())
                if self._config_key(next_cfg) in seen:
                    next_cfg = self._random_unseen_config(seen)
            except Exception:
                next_cfg = self._random_unseen_config(seen)
            evaluate(next_cfg)

        best_config, best_score = max(history, key=lambda item: item[1])
        return SearchResult(best_config=dict(best_config), best_score=float(best_score), history=history, backend='botorch')

    def optimize(
        self,
        objective: ObjectiveFn,
        n_init: int = 5,
        n_iter: int = 10,
        candidate_pool_size: int = 256,
        backend: str | None = None,
    ) -> SearchResult:
        resolved_backend = self._resolve_backend(backend)
        if resolved_backend == 'botorch':
            try:
                return self._optimize_botorch(objective, n_init=n_init, n_iter=n_iter)
            except Exception:
                if str(backend or self.backend).lower() == 'botorch':
                    raise
        return self._optimize_sklearn(
            objective,
            n_init=n_init,
            n_iter=n_iter,
            candidate_pool_size=candidate_pool_size,
        )


def run_bo(
    search_space: Dict[str, Any],
    objective: ObjectiveFn,
    *,
    random_state: int = 42,
    backend: str = 'auto',
    n_init: int = 5,
    n_iter: int = 10,
    candidate_pool_size: int = 256,
    out_path: str | Path | None = None,
    payload_extras: Dict[str, Any] | None = None,
) -> tuple[SearchResult, str | None]:
    """Run BO and optionally persist a standard JSON summary.

    This is the packaged BO entrypoint used by user-facing scripts. It keeps
    the optimization logic inside ``wisdom`` while allowing callers to supply
    their own objective and any extra metadata they want written alongside the
    standard search results.
    """
    result = BOSearch(search_space, random_state=random_state, backend=backend).optimize(
        objective,
        n_init=n_init,
        n_iter=n_iter,
        candidate_pool_size=candidate_pool_size,
        backend=backend,
    )

    written_path = None
    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'backend_used': result.backend,
            'best_config': result.best_config,
            'best_score': result.best_score,
            'num_evaluations': len(result.history),
            'search_space': search_space,
            'history': [{'config': cfg, 'score': float(score)} for cfg, score in result.history],
        }
        if payload_extras:
            payload.update(payload_extras)
        out_file.write_text(json.dumps(payload, indent=2))
        written_path = str(out_file)

    return result, written_path


__all__ = ['BOSearch', 'SearchResult', 'run_bo']

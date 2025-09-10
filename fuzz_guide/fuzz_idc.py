import torch
import pandas as pd
from typing import Dict, Optional, Union

from torch.utils.data import DataLoader, TensorDataset

# NEW IMPORTS: use your updated IDC implementations
from src.deepidc import DeepIDC
from src.wisdom import WisdomIDC
from src.attribution import get_relevance_scores_dataloader


def _to_dataloader(x: Union[DataLoader, torch.Tensor], batch_size: int = 64) -> DataLoader:
    """
    Accept either a DataLoader or a 4D torch.Tensor (B, C, H, W).
    If Tensor, wrap into a DataLoader with dummy labels.
    """
    if isinstance(x, DataLoader):
        return x
    if torch.is_tensor(x):
        if x.dim() != 4:
            raise ValueError(f"Expected input tensor of shape (B,C,H,W), got {tuple(x.shape)}")
        # dummy labels; IDC doesn't use labels, just forward pass for activations
        ds = TensorDataset(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # pin_memory off here to avoid accidental host RAM pressure in huge runs
        return DataLoader(ds, batch_size=min(batch_size, x.size(0)), shuffle=False)
    raise TypeError(f"Unsupported input type: {type(x)}. Expected DataLoader or 4D Tensor.")


class _BaseIDCWrapperNew:
    def __init__(self):
        self.current: float = 0.0
        self.idc = None
        self.selected_neurons: Optional[Dict[str, torch.Tensor]] = None
        self.cluster_groups = None
        # cumulative IDC state
        self._seen_tuples: set = set()   # set of tuples across all accepted samples
        self._total_comb: Optional[int] = None
        self._last_new_tuples: list = [] # stash from last calculate() for update()
    
    def _fit_clusters(self, trainloader: DataLoader):
        """
        Pre-compute per-neuron clusters on train set.
        """
        if self.idc is None or self.selected_neurons is None:
            raise RuntimeError("IDC and selected_neurons must be set before fitting clusters.")
        train_acts = self.idc.get_selected_activations(trainloader, self.selected_neurons)
        self.cluster_groups = self.idc.cluster_per_neuron(train_acts)
    
    def calculate(self, input_data: Union[DataLoader, torch.Tensor]) -> Dict[str, float]:
        """
        Compute **incremental** IDC:
          - build tuples for the given batch
          - count how many are NEW vs self._seen_tuples
          - report new coverage ratio = (|seen| + new) / total_comb
        """
        if self.cluster_groups is None:
            raise RuntimeError("Clusters not prepared. Call _fit_clusters(trainloader) first.")
        dl = _to_dataloader(input_data)
        # 1) get per-layer activations for selected neurons
        test_acts = self.idc.get_selected_activations(dl, self.selected_neurons)
        # lazy init of total combinations
        if self._total_comb is None:
            total = 1
            for _, models in self.cluster_groups.items():
                for m in models:
                    total *= int(getattr(m, "n_clusters", 1))
            self._total_comb = max(1, total)
        # 2) make tuples for this batch
        tuples = []
        num_samples = None
        # get any layer tensor to deduce batch size
        for v in test_acts.values():
            num_samples = v.shape[0]; break
        num_samples = int(num_samples or 0)
        for i in range(num_samples):
            tup = []
            for lname, models in self.cluster_groups.items():
                # a_row = test_acts[lname][i].cpu().numpy()  # shape (K_l,)
                a_row = test_acts[lname][i].detach().cpu().numpy().astype("float64", copy=False)
                for j, m in enumerate(models):
                    cid = self.idc._predict_one(m, a_row[j:j+1].reshape(1, -1))
                    # cid = self.idc._predict_one(m, a_row[j:j+1])
                    tup.append(int(cid))
            tuples.append(tuple(tup))
        # 3) incremental accounting
        new_tuples = [t for t in tuples if t not in self._seen_tuples]
        inc = len(new_tuples)
        new_ratio = (len(self._seen_tuples) + inc) / float(self._total_comb)
        self._last_new_tuples = new_tuples
        return {"ratio": float(new_ratio), "new": int(inc)}

    def gain(self, cov_dict: Dict[str, float]) -> float:
        # treat "how many new tuples" as gain; positive -> accept
        return float(cov_dict.get("new", 0))

    def update(self, cov_dict: Dict[str, float], gain: float):
        if gain > 0:
            # merge new tuples and advance coverage
            self._seen_tuples.update(self._last_new_tuples)
            self.current = float(cov_dict["ratio"])
        self._last_new_tuples = []
    
    def save(self, path: str):
        torch.save({"coverage": float(self.current)}, path)

class _BaseIDCWrapper:
    """
    Minimal wrapper that the fuzzer expects:
      - has .current (float)
      - calculate(input) -> {"ratio": coverage_value}
      - gain(cov_dict)   -> float
      - update(cov_dict, gain) -> None
      - save(path) -> None

    Internally holds:
      - self.idc (DeepIDC or WisdomIDC)
      - self.selected_neurons: Dict[layer_name, LongTensor of indices]
      - self.cluster_groups: Dict[layer_name, List[cluster_model]]
    """
    def __init__(self):
        self.current: float = 0.0
        self.idc = None
        self.selected_neurons: Optional[Dict[str, torch.Tensor]] = None
        self.cluster_groups = None

    def _fit_clusters(self, trainloader: DataLoader):
        """
        Pre-compute per-neuron clusters on train set.
        """
        if self.idc is None or self.selected_neurons is None:
            raise RuntimeError("IDC and selected_neurons must be set before fitting clusters.")
        train_acts = self.idc.get_selected_activations(trainloader, self.selected_neurons)
        self.cluster_groups = self.idc.cluster_per_neuron(train_acts)

    # ----------------- API used by the Fuzzer -----------------

    def calculate(self, input_data: Union[DataLoader, torch.Tensor]) -> Dict[str, float]:
        """
        Compute coverage for a batch (Tensor or DataLoader) **against existing clusters**.
        Returns {"ratio": coverage_rate}.
        """
        if self.cluster_groups is None:
            raise RuntimeError("Clusters not prepared. Call _fit_clusters(trainloader) first.")

        dl = _to_dataloader(input_data)
        test_acts = self.idc.get_selected_activations(dl, self.selected_neurons)
        coverage_rate, _, _ = self.idc.compute_coverage(test_acts, self.cluster_groups, save_json=False)
        return {"ratio": float(coverage_rate)}

    def gain(self, cov_dict: Dict[str, float]) -> float:
        return float(cov_dict["ratio"] - self.current)

    def update(self, cov_dict: Dict[str, float], gain: float):
        if gain > 0.0:
            self.current = float(cov_dict["ratio"])

    def save(self, path: str):
        torch.save({"coverage": float(self.current)}, path)


# ==================================================================
# DeepImportance wrapper (LRP-based neuron selection)
# ==================================================================
class DeepImportance(_BaseIDCWrapperNew):
    """
    DeepImportance criterion for fuzzing:
      - Pick top-m neurons with LRP on the train set (optionally excluding the last layer)
      - Cluster each selected neuron’s activations on train set
      - During fuzzing, compute coverage of mutated inputs w.r.t. those clusters
    """
    def __init__(
        self,
        model: torch.nn.Module,
        top_m_neurons: int,
        n_clusters: int,
        coverage_method: str,            # e.g., "KMeans" (kept for compatibility)
        trainloader: DataLoader,
        final_layer_name: Optional[str], # name of last trainable layer; may be excluded
        device: Union[str, torch.device] = "cpu",
        use_silhouette: bool = False,
        test_all_classes: bool = True,
        cache_path: Optional[str] = None,
    ):
        super().__init__()
        dev = torch.device(device) if not isinstance(device, torch.device) else device

        # Construct the new DeepIDC
        extra = dict(n_clusters=n_clusters, random_state=42, n_init=10)
        self.idc = DeepIDC(
            model=model,
            top_m_neurons=top_m_neurons,
            n_clusters=n_clusters,
            use_silhouette=use_silhouette,
            test_all_classes=test_all_classes,
            clustering_method_name=coverage_method,
            device=str(dev),
            clustering_params=extra,
            cache_path=cache_path,
        )

        # 1) layer-wise relevance (LRP) on train
        layer_scores = get_relevance_scores_dataloader(model, trainloader, dev, attribution_method="lrp")

        # 2) select top neurons (optionally exclude last layer)
        self.selected_neurons = self.idc.select_top_neurons(layer_scores, exclude_last=final_layer_name)

        # 3) cluster on train activations
        self._fit_clusters(trainloader)


# ==================================================================
# Wisdom wrapper (CSV-based neuron selection)
# ==================================================================
class Wisdom(_BaseIDCWrapperNew):
    """
    Wisdom criterion for fuzzing:
      - Read top-m neurons from a Wisdom CSV (LayerName, NeuronIndex, Score)
      - Cluster each selected neuron’s activations on train set
      - During fuzzing, compute coverage of mutated inputs w.r.t. those clusters
    """
    def __init__(
        self,
        model: torch.nn.Module,
        top_m_neurons: int,
        n_clusters: int,
        coverage_method: str,           # e.g., "KMeans" (kept for compatibility)
        trainloader: DataLoader,
        csv_file: str,                  # Wisdom ranking CSV
        device: Union[str, torch.device] = "cpu",
        use_silhouette: bool = False,
        test_all_classes: bool = True,
        cache_path: Optional[str] = None,
    ):
        super().__init__()
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        self.model = model

        # Construct the new WisdomIDC
        extra = dict(n_clusters=n_clusters, random_state=42, n_init=10)
        self.idc = WisdomIDC(
            model=model,
            top_m_neurons=top_m_neurons,
            n_clusters=n_clusters,
            use_silhouette=use_silhouette,
            test_all_classes=test_all_classes,
            clustering_method_name=coverage_method,
            device=str(dev),
            clustering_params=extra,
            cache_path=cache_path,
        )

        # 1) read top-m neurons from CSV
        df = pd.read_csv(csv_file)
        df = df.sort_values(by="Score", ascending=False).head(top_m_neurons)
        selected: Dict[str, torch.Tensor] = {}
        for lname, grp in df.groupby("LayerName"):
            idx = torch.as_tensor(grp["NeuronIndex"].values, dtype=torch.long)
            selected[lname] = idx
        self.selected_neurons = selected

        # 2) cluster on train activations
        self._fit_clusters(trainloader)

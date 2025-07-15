import torch
from src.idc import IDC
from src.attribution import get_relevance_scores_dataloader
import pandas as pd

# TODO: Coverage methods and configurations should be added here
class BaseDI:
    def __init__(self, model, top_m_neurons, n_clusters, coverage_method):
        self.top_m_neurons = top_m_neurons
        self.n_clusters = n_clusters
        self.coverage_method = coverage_method
        self.model = model
        self.idc = IDC(self.model, top_m_neurons, n_clusters, False, True, coverage_method, None, None)
        self.current = 0.0

        # to be filled by subclasses
        self.important_neurons = None
        self.cluster_groups = None

    # ---------------------------------------------
    def _prepare_clusters(self, trainloader):
        activation_dict, selected_activations = self.idc.get_activations_model_dataloader(trainloader, self.important_neurons)

        selected_activations = {k: v.half().cpu() for k, v in selected_activations.items()}
        self.cluster_groups = self.idc.cluster_activation_values_all(selected_activations)

    # ---------------------------------------------
    #  API expected by Fuzzer / run_fuzz.py
    # ---------------------------------------------
    def calculate(self, input_data):
        from torch.utils.data import DataLoader
        
        if isinstance(input_data, DataLoader):
            # Handle DataLoader input
            coverage_rate, total_combination, max_coverage = self.idc.compute_idc_test_whole_dataloader(
                input_data, self.important_neurons, self.cluster_groups
            )
        elif isinstance(input_data, torch.Tensor):
            # Handle Tensor input (B,C,H,W) already on same device as model
            coverage_rate, total_combination, max_coverage = self.idc.compute_idc_test_whole(
                input_data, self.important_neurons, self.cluster_groups
            )
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}. Expected DataLoader or Tensor.")
        
        return {"ratio": coverage_rate}

    def gain(self, cov_dict):
        return cov_dict["ratio"] - self.current

    def update(self, cov_dict, gain):
        if gain > 0:
            self.current = cov_dict["ratio"]

    def save(self, path):
        torch.save({"coverage": self.current}, path)


# ==================================================================
class DeepImportance(BaseDI):
    """
    Implements the full pipeline described in run_rq_2_demo.py / idc_coverage().
    """
    def __init__(self, model, top_m_neurons, n_clusters, coverage_method, trainloader, final_layer, device):
        super().__init__(model, top_m_neurons, n_clusters, coverage_method)

        # Relevance scores on train set
        layer_scores = get_relevance_scores_dataloader(model, trainloader, device, attribution_method='lrp')
        # Important neurons
        self.important_neurons, _ = self.idc.select_top_neurons_all(layer_scores, final_layer)
        # Build clusters from activations (parent class helper)
        self._prepare_clusters(trainloader)


# ==================================================================
class Wisdom(BaseDI):
    """
    Same idea but the 'important neurons' come from a CSV produced by
    the Wisdom heuristic (see wisdom_coverage()).
    """
    def __init__(self, model, top_m_neurons, n_clusters, coverage_method, trainloader, csv_file):
        super().__init__(model, top_m_neurons, n_clusters, coverage_method)

        # Read pre-selected neurons (list of tuples [(layer, idx), ...])
        df = pd.read_csv(csv_file)
        df_sorted = df.sort_values(by='Score', ascending=False).head(self.top_m_neurons)
        self.important_neurons = {}
        for layer_name, group in df_sorted.groupby('LayerName'):
            self.important_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)
        
        # Build clusters over train activations
        self._prepare_clusters(trainloader)
import os
import sys
import pathlib

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from run_cases.support import datasets as support_datasets
import wisdom
from wisdom import BOSearch, ClusteringConfig, run_bo, WisdomConfig, WisdomIDC
from wisdom import utils as wisdom_utils


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_public_api_exports():
    assert WisdomIDC is not None
    assert BOSearch is not None
    assert run_bo is not None
    assert not hasattr(wisdom, 'IDC')
    assert hasattr(wisdom_utils, 'load_detection_model')
    assert hasattr(support_datasets, 'load_cifar100')


def test_wisdom_idc_runs_on_tiny_model():
    torch.manual_seed(0)
    model = TinyNet()
    inputs = torch.randn(8, 3, 8, 8)
    labels = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)

    layer_scores = {
        'conv': torch.tensor([0.9, 0.8, 0.3, 0.1]),
        'fc': torch.tensor([0.2, 0.1]),
    }

    idc = WisdomIDC(
        model,
        impl='idc',
        cfg=WisdomConfig(top_m_neurons=3, test_all_classes=True, cache_path=None),
        cluster=ClusteringConfig(method='KMeans', params={'n_clusters': 2, 'random_state': 42}, use_silhouette=True),
    )
    selected = idc.fit(loader, layer_scores, device='cpu')
    assert selected
    rate, total, max_cov = idc.coverage(loader, selected=selected, device='cpu')
    assert total >= 1
    assert 0.0 <= rate <= 1.0
    assert 0.0 <= max_cov <= 1.0

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wisdom.core.wisdom import (
    build_layer_groups,
    get_group_names,
    load_groupwise_top_neurons,
    load_layerwise_top_neurons,
    split_selected_by_group,
)


def test_get_group_names_default():
    assert get_group_names(3) == ('early', 'middle', 'late')


def test_build_layer_groups_splits_in_order():
    groups = build_layer_groups(['model.0.conv', 'model.1.conv', 'model.2.conv', 'model.3.conv'], n_groups=2)
    assert groups == {
        'front': ['model.0.conv', 'model.1.conv'],
        'back': ['model.2.conv', 'model.3.conv'],
    }


def test_load_layerwise_top_neurons(tmp_path):
    csv_path = tmp_path / 'scores.csv'
    pd.DataFrame(
        [
            {'LayerName': 'yolo_model.model.0.conv', 'NeuronIndex': 0, 'Score': 0.2},
            {'LayerName': 'yolo_model.model.0.conv', 'NeuronIndex': 1, 'Score': 0.5},
            {'LayerName': 'yolo_model.model.1.conv', 'NeuronIndex': 0, 'Score': 0.7},
            {'LayerName': 'yolo_model.model.1.conv', 'NeuronIndex': 1, 'Score': 0.0},
        ]
    ).to_csv(csv_path, index=False)

    selected = load_layerwise_top_neurons(str(csv_path), per_layer_k=1)
    assert selected == {
        'model.0.conv': [1],
        'model.1.conv': [0],
    }


def test_load_groupwise_top_neurons(tmp_path):
    csv_path = tmp_path / 'scores.csv'
    rows = []
    for layer_idx in range(6):
        rows.append({'LayerName': f'yolo_model.model.{layer_idx}.conv', 'NeuronIndex': 0, 'Score': float(100 - layer_idx)})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    selected = load_groupwise_top_neurons(str(csv_path), per_group_k=1, n_groups=3)
    assert sum(len(indices) for indices in selected.values()) == 3


def test_split_selected_by_group():
    grouped = split_selected_by_group(
        {
            'model.0.conv': [0],
            'model.1.conv': [1],
            'model.2.conv': [2],
            'model.3.conv': [3],
        },
        n_groups=2,
    )
    assert grouped == {
        'front': {
            'model.0.conv': [0],
            'model.1.conv': [1],
        },
        'back': {
            'model.2.conv': [2],
            'model.3.conv': [3],
        },
    }

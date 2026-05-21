import pathlib
import sys

import pandas as pd
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from repair.workflow import (
    _build_augmented_train_loader,
    _coverage_change_score,
    _infer_task,
    _load_detection_samples,
    _plot_candidate_sweep,
    _prepare_detection_data_yaml,
    _select_ranked_rows,
    perturb_important_pixels,
)


def test_coverage_change_score_combines_novelty_and_changed_fraction():
    assert _coverage_change_score(1.0, 0.25) == 1.25
    assert _coverage_change_score(0.0, 0.5) == 0.5


def test_select_ranked_rows_overall_picks_top_scores():
    df = pd.DataFrame(
        [
            {'dataset_index': 0, 'label': 0, 'repair_score': 0.3, 'changed_fraction': 0.3, 'novel_combo': 0.0},
            {'dataset_index': 1, 'label': 1, 'repair_score': 1.2, 'changed_fraction': 0.2, 'novel_combo': 1.0},
            {'dataset_index': 2, 'label': 0, 'repair_score': 0.8, 'changed_fraction': 0.8, 'novel_combo': 0.0},
        ]
    )
    selected = _select_ranked_rows(df, 2, 'overall')
    assert selected['dataset_index'].tolist() == [1, 2]


def test_select_ranked_rows_class_balanced_spreads_across_labels():
    df = pd.DataFrame(
        [
            {'dataset_index': 0, 'label': 0, 'repair_score': 1.0, 'changed_fraction': 0.0, 'novel_combo': 1.0},
            {'dataset_index': 1, 'label': 0, 'repair_score': 0.9, 'changed_fraction': 0.0, 'novel_combo': 1.0},
            {'dataset_index': 2, 'label': 1, 'repair_score': 0.8, 'changed_fraction': 0.0, 'novel_combo': 1.0},
            {'dataset_index': 3, 'label': 1, 'repair_score': 0.7, 'changed_fraction': 0.0, 'novel_combo': 1.0},
        ]
    )
    selected = _select_ranked_rows(df, 2, 'class-balanced')
    assert sorted(selected['label'].tolist()) == [0, 1]


def test_perturb_important_pixels_preserves_shape():
    images = torch.zeros(2, 1, 4, 4)
    importance = torch.ones(2, 4, 4)
    perturbed = perturb_important_pixels(images, importance, frac=0.25, std=0.1)
    assert perturbed.shape == images.shape
    assert torch.any(perturbed != images)


def test_build_augmented_train_loader_normalizes_label_types():
    dataset = TensorDataset(torch.randn(4, 1, 4, 4), torch.tensor([0, 1, 2, 3], dtype=torch.long))
    selected_rows = pd.DataFrame([{'dataset_index': 10}, {'dataset_index': 11}])
    payloads = {
        10: {'perturbed': torch.randn(1, 4, 4), 'label': 4},
        11: {'perturbed': torch.randn(1, 4, 4), 'label': 5},
    }

    loader = _build_augmented_train_loader(dataset, [0, 1], selected_rows, payloads, batch_size=4)
    images, labels = next(iter(loader))

    assert images.shape[0] == 4
    assert labels.dtype == torch.int64


def test_infer_task_detects_yolo_from_weights_and_yaml():
    assert _infer_task('auto', 'weights/yolo11n.pt', None) == 'detection'
    assert _infer_task('auto', 'models/lenet_MNIST_whole.pth', '/tmp/data.yaml') == 'detection'
    assert _infer_task('auto', 'models/lenet_MNIST_whole.pth', None) == 'classification'


def test_load_detection_samples_supports_txt_entries(tmp_path):
    images_dir = tmp_path / 'coco' / 'images' / 'train2017'
    labels_dir = tmp_path / 'coco' / 'labels' / 'train2017'
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    image_path = images_dir / '0001.jpg'
    image_path.write_bytes(b'not-an-image')
    (labels_dir / '0001.txt').write_text('3 0.5 0.5 0.25 0.25\n')

    txt_path = tmp_path / 'train.txt'
    txt_path.write_text(str(image_path) + '\n')
    data_yaml = tmp_path / 'data.yaml'
    data_yaml.write_text(f'train: {txt_path}\nval: {txt_path}\nnc: 4\nnames: {{0: a, 1: b, 2: c, 3: d}}\n')

    samples, names = _load_detection_samples(str(data_yaml), 'train')

    assert len(samples) == 1
    assert samples[0]['image_path'] == str(image_path)
    assert samples[0]['label'] == 3
    assert names[3] == 'd'


def test_prepare_detection_data_yaml_writes_train_and_val_layout(tmp_path):
    src_images = tmp_path / 'src' / 'images'
    src_labels = tmp_path / 'src' / 'labels'
    src_images.mkdir(parents=True)
    src_labels.mkdir(parents=True)

    clean_image = src_images / 'clean.jpg'
    clean_image.write_bytes(b'clean')
    clean_label = src_labels / 'clean.txt'
    clean_label.write_text('0 0.5 0.5 0.2 0.2\n')
    pert_label = src_labels / 'pert.txt'
    pert_label.write_text('1 0.5 0.5 0.2 0.2\n')

    train_items = [{'image_path': str(clean_image), 'label_path': str(clean_label), 'image_name': 'clean.jpg'}]
    val_items = [{'image_name': 'pert.png', 'label_path': str(pert_label), 'image_tensor': torch.rand(3, 8, 8)}]

    data_yaml = _prepare_detection_data_yaml(train_items, val_items, str(tmp_path / 'subset'), {0: 'a', 1: 'b'})
    subset_root = pathlib.Path(data_yaml).parent

    assert (subset_root / 'images' / 'train2017' / 'clean.jpg').exists()
    assert (subset_root / 'labels' / 'train2017' / 'clean.txt').exists()
    assert (subset_root / 'images' / 'val2017' / 'pert.png').exists()
    assert (subset_root / 'labels' / 'val2017' / 'pert.txt').exists()


def test_plot_candidate_sweep_supports_detection_metrics(tmp_path):
    df = pd.DataFrame(
        [
            {'candidate_count': 0, 'eval_mean_repair_score': 0.4, 'eval_clean_map': 0.2, 'eval_perturbed_map': 0.1},
            {'candidate_count': 1, 'eval_mean_repair_score': 0.3, 'eval_clean_map': 0.25, 'eval_perturbed_map': 0.2},
        ]
    )

    out_path = tmp_path / 'detection_sweep.pdf'
    generated = _plot_candidate_sweep(df, out_path)

    assert generated == str(out_path)
    assert out_path.exists()

"""Dataset and model-path helpers for run-case scripts."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST


def _train_val_dataset(dataset, val_split: float = 0.25):
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    return {'train': train_subset, 'val': val_subset}, train_subset, val_subset


def _imagenet_loaders(root: str, batch_size: int = 32, num_workers: int = 2):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_dataset, val_dataset


def load_imagenet(
    batch_size: int = 32,
    root: str = './datasets/ImageNet',
    num_workers: int = 2,
    use_val: bool = False,
    label_path: str = './datasets/imagenet_labels.json',
):
    """Load ImageNet from the expected folder structure."""

    if not os.path.exists(label_path):
        url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
        urllib.request.urlretrieve(url, 'imagenet_labels.json')
        label_path = 'imagenet_labels.json'

    with open(label_path, 'r') as handle:
        classes = json.load(handle)

    if use_val:
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'val'), transform=transform)
        splits, train_dataset, test_dataset = _train_val_dataset(val_dataset, val_split=0.25)
        train_loader = DataLoader(splits['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(splits['val'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return train_loader, test_loader, train_dataset, test_dataset, classes

    train_loader, test_loader, train_dataset, test_dataset = _imagenet_loaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, test_loader, train_dataset, test_dataset, classes


def _cifar_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_cifar10(batch_size: int = 32, root: str = './datasets', shuffle: bool = True):
    transform = _cifar_transform()
    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader, train_dataset, test_dataset, classes


def load_cifar100(batch_size: int = 32, root: str = './datasets', shuffle: bool = True):
    transform = _cifar_transform()
    train_dataset = CIFAR100(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root=root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = tuple(test_dataset.classes)
    return train_loader, test_loader, train_dataset, test_dataset, classes


def load_mnist(
    batch_size: int = 32,
    root: str = './datasets',
    channel_first: bool = False,
    train_all: bool = False,
):
    transform_steps = [
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    if channel_first:
        transform_steps.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    transform = transforms.Compose(transform_steps)

    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)

    if train_all:
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classes = [str(i) for i in range(10)]
    return train_loader, test_loader, train_dataset, test_dataset, classes


def get_data(dataset_name: str, batch_size: int, data_path: str):
    key = dataset_name.lower()
    if key == 'cifar10':
        return load_cifar10(batch_size=batch_size, root=data_path, shuffle=True)
    if key == 'cifar100':
        return load_cifar100(batch_size=batch_size, root=data_path, shuffle=True)
    if key == 'mnist':
        return load_mnist(batch_size=batch_size, root=data_path)
    if key == 'imagenet':
        return load_imagenet(
            batch_size=batch_size,
            root=os.path.join(data_path, 'ImageNet'),
            num_workers=2,
            use_val=False,
        )
    raise ValueError(f'Invalid dataset: {dataset_name}')


def resolve_saved_model_path(saved_model: str) -> str:
    """Resolve model paths across legacy and local workspace layouts."""

    if not saved_model:
        return saved_model

    repo_root = Path(__file__).resolve().parents[2]
    home_env = os.getenv('HOME')
    home = Path(home_env).expanduser() if home_env else None
    legacy_prefix = '/torch-deepimportance/'
    raw_path = Path(saved_model)

    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(repo_root / saved_model)

    if saved_model.startswith(legacy_prefix):
        relative_path = Path(saved_model[len(legacy_prefix):])
        candidates.append(repo_root / relative_path)
        if home is not None:
            candidates.append(home / 'torch-deepimportance' / relative_path)
            candidates.append(Path(str(home) + saved_model))

    if home is not None and not raw_path.is_absolute():
        candidates.append(home / saved_model)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return saved_model


__all__ = [
    'get_data',
    'load_cifar10',
    'load_cifar100',
    'load_imagenet',
    'load_mnist',
    'resolve_saved_model_path',
]

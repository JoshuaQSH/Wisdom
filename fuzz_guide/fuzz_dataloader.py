import os
import random
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

class CIFAR10Dataset(Dataset):
    def __init__(self,
                 args,
                 image_dir='./datasets/CIFAR10/',
                 split='train'):
        super(CIFAR10Dataset).__init__()
        assert split in ['train', 'test']
        self.total_class_num = 10
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                ])

        self.image_list = []
        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        label = self.class_list.index(label)
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, label

class ImageNetDataset(Dataset):
    def __init__(self,
                 args,
                 image_dir='./datasets/ImageNet/',
                 label2index_file='./datasets/ImageNet/imagenet_labels.json',
                 split='train'):
        super(ImageNetDataset).__init__()
        assert split in ['train', 'val']
        self.total_class_num = 1000
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        self.image_list = []
        
        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)

        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        index = self.label2index[label]
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.init_param()

    def init_param(self):
        self.gpus = 1
        # self.gpus = torch.cuda.device_count()
        # TODO: multi GPU

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.args.batch_size * self.gpus,
                            num_workers=int(self.args.num_workers),
                            shuffle=shuffle
                        )
        return data_loader

def get_loader(args):
    assert args.dataset in ['CIFAR10', 'ImageNet']
    if args.dataset == 'CIFAR10':
        train_data = CIFAR10Dataset(args, split='train')
        test_data = CIFAR10Dataset(args, split='test')
        loader = DataLoader(args)
        train_loader = loader.get_loader(train_data, False)
        test_loader = loader.get_loader(test_data, False)
        seed_loader = loader.get_loader(test_data, True)
        TOTAL_CLASS_NUM = 10
    elif args.dataset == 'ImageNet':
        train_data = ImageNetFuzzDataset(args, image_dir=args.data_path, label2index_file='./datasets/imagenet_labels.json', split='train')
        test_data = ImageNetFuzzDataset(args, image_dir=args.data_path, label2index_file='./datasets/imagenet_labels.json', split='val')
        loader = DataLoader(args)
        train_loader = loader.get_loader(train_data, False)
        test_loader = loader.get_loader(test_data, False)
        seed_loader = loader.get_loader(test_data, True)
        TOTAL_CLASS_NUM = 1000

    return TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader

class FuzzDataset:
    def __init__(self):
        raise NotImplementedError

    def label2index(self):
        raise NotImplementedError

    def get_len(self):
        return len(self.image_list)

    def get_item(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        # label = self.cat_list.index(label)
        index = self.label2index(label)
        assert int(index) < self.args.num_class
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return (image, index)

    def build(self):
        image_list = []
        label_list = []
        for i in tqdm(range(self.get_len())):
            (image, label) = self.get_item(i)
            image_list.append(image)
            label_list.append(label)
        return image_list, label_list

    def to_numpy(self, image_list, is_image=True):
        image_numpy_list = []
        for i in tqdm(range(len(image_list))):
            image = image_list[i]
            if is_image:
                image_numpy = image.transpose(0, 2).numpy()
            else:
                image_numpy = image.numpy()
            image_numpy_list.append(image_numpy)
        print('Numpy: %d' % len(image_numpy_list))
        return image_numpy_list

    def to_batch(self, data_list, is_image=True):
        batch_list = []
        batch = []
        for i, data in enumerate(data_list):
            if i and i % self.args.batch_size == 0:
                batch_list.append(torch.stack(batch, 0))
                batch = []
            batch.append(self.norm(data) if is_image else data)
        if len(batch):
            batch_list.append(torch.stack(batch, 0))
        print('Batch: %d' % len(batch_list))
        return batch_list

class TorchvisionCIFAR10FuzzDataset(FuzzDataset):
    def __init__(self, args, root="./data", split="test"):
        self.args = args
        self.dataset  = torchvision.datasets.CIFAR10(
                           root=root, 
                           train=(split=="train"),
                           download=True)
        self.transform = transforms.Compose([
              transforms.Resize(args.image_size),
              transforms.CenterCrop(args.image_size),
              transforms.ToTensor(),
        ])
        self.norm = transforms.Normalize((0.4914,0.4822,0.4465),
                                         (0.2471,0.2435,0.2616))
        self.image_list = list(range(len(self.dataset)))   # indices only

    # ---------- hooks required by FuzzDataset ----------
    def label2index(self, label_name_or_int):
        return int(label_name_or_int)          # torchvision already returns int

    def get_len(self):
        return len(self.dataset)

    def get_item(self, idx):
        images, labels = self.dataset[idx]   # PIL, int
        images = self.transform(images)      # tensor in [0,1]
        labels = torch.LongTensor([labels]).squeeze()
        return images, labels

class TorchImageNetFuzzDataset(FuzzDataset):
    def __init__(self, args, root="./data", split="val"):
        self.args = args
        self.root = root
        self.split = split
        
        # Build the path to the split directory
        self.image_dir = os.path.join(root, split)
        
        # Verify the directory exists
        if not os.path.exists(self.image_dir):
            raise RuntimeError(f"Dataset directory not found: {self.image_dir}")
            
        # Use ImageFolder which automatically handles class directories
        self.dataset = torchvision.datasets.ImageFolder(
            root=self.image_dir,
            transform=transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
            ])
        )
        
        self.transform = self.dataset.transform
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        # Get class names from ImageFolder
        self.class_list = self.dataset.classes
        
        # Apply class and sample limits if specified
        if hasattr(args, 'num_class') and args.num_class > 0:
            self.class_list = self.class_list[:args.num_class]
            # Filter dataset indices to only include selected classes
            self.valid_indices = self._get_filtered_indices(args.num_class, 
                                                           getattr(args, 'num_per_class', None))
        else:
            self.valid_indices = list(range(len(self.dataset)))
            
        self.image_list = self.valid_indices  # indices only
        print(f'Total {len(self.image_list)} ImageNet Data from {len(self.class_list)} classes.')

    def _get_filtered_indices(self, num_classes, num_per_class=None):
        """Get indices for a subset of classes and optionally limit samples per class."""
        class_counts = {}
        valid_indices = []
        
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            
            # Only include first num_classes
            if label >= num_classes:
                continue
                
            # Count samples per class
            if label not in class_counts:
                class_counts[label] = 0
                
            # Limit samples per class if specified
            if num_per_class is None or class_counts[label] < num_per_class:
                valid_indices.append(idx)
                class_counts[label] += 1
                
        return valid_indices

    # ---------- hooks required by FuzzDataset ----------
    def label2index(self, label_name_or_int):
        """Convert class name to index using ImageFolder's class_to_idx mapping."""
        if isinstance(label_name_or_int, str):
            return self.dataset.class_to_idx[label_name_or_int]
        else:
            return int(label_name_or_int)  # already an index

    def get_len(self):
        return len(self.image_list)

    def get_item(self, idx):
        # Get the actual dataset index from our filtered list
        actual_idx = self.image_list[idx]
        images, labels = self.dataset[actual_idx]  # PIL, int (ImageFolder handles loading)
        labels = torch.LongTensor([labels]).squeeze()
        return images, labels

class CIFAR10FuzzDataset(FuzzDataset):
    def __init__(self,
                 args,
                 image_dir='./datasets/',
                 split='test'):
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                ])
        self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        self.image_list = []
        
        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def label2index(self, label_name):
        return self.class_list.index(label_name)

class ImageNetFuzzDataset(FuzzDataset):
    def __init__(self,
                 args,
                 image_dir='./datasets/ImageNet/',
                 label2index_file='./datasets/ImageNet/imagenet_labels.json',
                 split='val'):
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                ])
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_list = []
        
        with open(label2index_file, 'r') as f:
            self.label2index_dict = json.load(f)

        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def label2index(self, label_name):
        breakpoint()
        return self.label2index_dict[label_name]

if __name__ == '__main__':
    pass
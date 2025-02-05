import unittest
import utils
from utils import SelectorDataset
# from src.utils import parse_args

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader

from model_hub import LeNet, Net
from models_cv import *
from YOLOv5.yolo import *

class UtilsTests(unittest.TestCase):
    
    def setUp(self):
        self.parser = utils.parse_args()
    
    def test_default_args(self):
        args = self.parser
        self.assertEqual(args.model, 'lenet')
        self.assertEqual(args.model_path, 'None')
        self.assertEqual(args.saved_model, 'lenet_cifar10.pt')
        self.assertEqual(args.dataset, 'cifar10')
        self.assertEqual(args.data_path, '/data/shenghao/dataset/')
        self.assertEqual(args.importance_file, './saved_files/plane_lenet_importance.json')
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.device, 'cuda:0')
        self.assertFalse(args.large_image)
        self.assertFalse(args.random_prune)
        self.assertFalse(args.use_silhouette)
        self.assertEqual(args.n_clusters, 2)
        self.assertFalse(args.cluster_scores)
        self.assertEqual(args.top_m_neurons, 5)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.test_image, 'plane')
        self.assertFalse(args.test_all)
        self.assertFalse(args.all_class)
        self.assertFalse(args.idc_test_all)
        self.assertEqual(args.num_samples, 1000)
        self.assertEqual(args.attr, 'lc')
        self.assertEqual(args.layer_index, 1)
        self.assertFalse(args.all_attr)
        self.assertFalse(args.layer_by_layer)
        self.assertFalse(args.end2end)
        self.assertFalse(args.vis_attributions)
        self.assertFalse(args.viz)
        self.assertFalse(args.logging)
        self.assertEqual(args.log_path, './logs/')
    
    def test_model_load_cv(self):
        args = self.parser
        offer_model_name = ['vgg16', 'lenet',
                            'convnext_base', 
                            'efficientnet_v2_s', 
                            'efficientnet_v2_m', 
                            'mnasnet1_0', 
                            'googlenet',
                            'inception_v3',
                            'mobilenet_v3_small',
                            'resnet18',
                            'resnet152',
                            'resnext101_32x8d',
                            'vit_b_16']
        # offer_model_name = ['mobilenet_v3_small', 'efficientnet_v2_s', 'convnext_base']
        for model_name in offer_model_name:
            model, module_name, module = utils.get_model(model_name=model_name)
            print(model_name, len(module_name))
            # print(model_name, module_name)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters: {total_params}")
    
    def test_model_infer_cifar(self):
        args = self.parser
        model_classes = {
            'lenet': LeNet,
            'vgg16': lambda: VGG('VGG16'),
            'resnet18': ResNet18,
            'googlenet': GoogLeNet,
            'densenet': DenseNet121,
            'resnext29': ResNeXt29_2x64d,
            'mobilenetv2': MobileNetV2,
            'shufflenetv2': lambda: ShuffleNetV2(1),
            'senet': SENet18,
            'preresnet': PreActResNet18,
            'mobilenet': MobileNet,
            'DPN92': DPN92,
            'efficientnet': EfficientNetB0,
            'regnet': RegNetX_200MF,
            'simpledla': SimpleDLA,
        }
        x = torch.randn(1, 3, 32, 32)
        for model_name in model_classes:
            model = model_classes[model_name]()                
            y = model(x)
            self.assertEqual(y.shape, (1, 10))
    
    def test_yolov5(self):
        cfg = 'YOLOv5/yolov5s.yaml'
        model = Model(cfg)
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640)
        y = model(img, profile=True)
        self.assertEqual(y[0].shape, (8, 3, 80, 80, 6))
    
    def test_selector_dataloader(self):
        args = self.parser
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        train_dataset = CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
        # One-hot vector for all images
        layer_info = torch.tensor([0., 0., 1., 0., 0.]).repeat(len(train_dataset), 1)  
        # Example list of attribution methods
        attribution_labels = ['lc', 'la', 'ii', 'lc', 'ldl', 'lig', 'lgs', 'lgxa', 'lrp', 'lfa'] * 5000
        attribution_methods = ['lc', 'la', 'ii', 'lgxa', 'lgc', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp']

        # Create the custom dataset
        selector_train_dataset = SelectorDataset(train_dataset, layer_info, attribution_labels, attribution_methods)

        # Create DataLoader for training
        trainloader = DataLoader(selector_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        # Create DataLoader for testing (similarly, you need to create the corresponding test dataset)
        selector_test_dataset = SelectorDataset(test_dataset, layer_info, attribution_labels, attribution_methods)
        testloader = torch.utils.data.DataLoader(selector_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        for images, layer_info, labels in trainloader:
            print(f"Images shape: {images.shape}")
            print(f"Layer info shape: {layer_info.shape}")
            print(f"Labels shape: {labels.shape}")
            self.assertEqual(images.shape, (args.batch_size, 3, 224, 224))
            self.assertEqual(layer_info.shape, (args.batch_size, 5))
            self.assertEqual(labels.shape[0], args.batch_size)
            break
    
    def test_load_CIFAR(self):
        args = self.parser
        trainloader, testloader, test_dataset, classes = utils.load_CIFAR(batch_size=args.batch_size, root=args.data_path)
        self.assertEqual(next(iter(trainloader))[1].shape[0], args.batch_size)
        self.assertEqual(next(iter(trainloader))[0].shape, (args.batch_size, 3, 224, 224))
        self.assertEqual(classes[0], 'plane')
    
    def test_load_IMAGE(self):
        args = self.parser
        trainloader, testloader, val_dataset, classes = utils.load_ImageNet(batch_size=args.batch_size, root='/data/shenghao/dataset/ImageNet', num_workers=2, use_val=False)
        self.assertEqual(next(iter(trainloader))[1].shape[0], args.batch_size)
        self.assertEqual(next(iter(trainloader))[0].shape, (args.batch_size, 3, 224, 224))
        self.assertEqual(classes[0], 'tench')
    
    def test_load_MNIST(self):
        args = self.parser
        trainloader, testloader, test_dataset, classes = utils.load_MNIST(batch_size=args.batch_size, root='/data/shenghao/dataset/MNIST', channel_first=False, train_all=False)
        self.assertEqual(next(iter(trainloader))[1].shape[0], args.batch_size)
        self.assertEqual(next(iter(trainloader))[0].shape, (args.batch_size, 1, 28, 28))
        self.assertEqual(classes[0], '0')

if __name__ == '__main__':
    unittest.main()
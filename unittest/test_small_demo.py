import unittest

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader

from models_info.models_cv import *
from models_info.YOLOv5.yolo import *
from src.attribution import *
from src.utils import *
from src.idc import IDC

# For united test
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 3 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class UtilsTests(unittest.TestCase):
    
    def setUp(self):
        self.parser = parse_args()
    
    def test_default_args(self):
        args = self.parser
        self.assertEqual(args.model, 'lenet')
        self.assertEqual(args.saved_model, '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth')
        self.assertEqual(args.dataset, 'mnist')
        self.assertEqual(args.data_path, './datasets/')
        self.assertEqual(args.importance_file, './logs/important.json')
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.device, 'cpu')
        self.assertFalse(args.large_image)
        self.assertFalse(args.random_prune)
        self.assertFalse(args.use_silhouette)
        self.assertEqual(args.n_clusters, 2)
        self.assertEqual(args.top_m_neurons, 5)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.test_image, '1')
        self.assertFalse(args.all_class)
        self.assertFalse(args.idc_test_all)
        self.assertEqual(args.num_samples, 0)
        self.assertEqual(args.attr, 'lc')
        self.assertEqual(args.layer_index, 1)
        self.assertFalse(args.layer_by_layer)
        self.assertFalse(args.end2end)
        # self.assertFalse(args.vis_attributions)
        # self.assertFalse(args.viz)
        self.assertFalse(args.logging)
        self.assertEqual(args.log_path, './logs/TestLog')
        self.assertEqual(args.csv_file, 'demo_layer_scores.csv')

    
    def test_model_load_lenet(self):
        args = self.parser
        load_model_path = os.getenv("HOME") + args.saved_model
        model, module_name, module = get_model(load_model_path=load_model_path)
        print("LeNet Model: ", len(module_name))
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
    
    
    def test_dynamic_clustering_idc_end2end(self):
        args = self.parser
        model_path = os.getenv("HOME") + args.saved_model
        trainloader, testloader, train_dataset, test_dataset, classes = load_MNIST(batch_size=args.batch_size, root=args.data_path)
        model, module_name, module = get_model(load_model_path=model_path)
        trainable_module, trainable_module_name = get_trainable_modules_main(model)
        test_image, test_label = get_class_data(testloader, classes, args.test_image)
        images, labels = get_class_data(trainloader, classes, args.test_image)
        attribution = get_relevance_scores_for_all_layers(model, images, labels, attribution_method='lrp')        
        idc = IDC(model=model, 
                  classes=classes, 
                  top_m_neurons=10, 
                  n_clusters=2, 
                  use_silhouette=True, 
                  test_all_classes=True,
                  clustering_method_name='KMeans')
        important_neuron_indices, inorderd_neuron_indices = idc.select_top_neurons_all(attribution, 'fc3')
        activation_values, selected_activations = idc.get_activation_values_for_model(images, classes[labels[0]], important_neuron_indices)
        cluster_groups = idc.cluster_activation_values_all(selected_activations)
        unique_cluster, coverage_rate = idc.compute_idc_test_whole(test_image, 
                            test_label,
                            important_neuron_indices,
                            cluster_groups,
                            'lrp')

    def test_load_MNIST(self):
        args = self.parser
        trainloader, testloader, train_dataset, test_dataset, classes = load_MNIST(batch_size=args.batch_size, root=args.data_path, channel_first=False, train_all=False)
        self.assertEqual(next(iter(trainloader))[1].shape[0], args.batch_size)
        self.assertEqual(next(iter(trainloader))[0].shape, (args.batch_size, 1, 32, 32))
        self.assertEqual(classes[0], '0')

if __name__ == '__main__':
    unittest.main()
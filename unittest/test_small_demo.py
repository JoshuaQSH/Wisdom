import pytest

import torch
import os
import sys
from models_info.models_cv import *
from models_info.YOLOv5.yolo import *
from src.attribution import *
from src.utils import *
from src.idc import IDC

# Map model names to constructors
MODEL_CLASSES = {
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

SAVED_MODEL_PATH = os.path.join(os.getenv("HOME", ""), "torch-deepimportance", "models_info", "saved_models", "lenet_MNIST_whole.pth")
SAVED_DATA_PATH = os.path.join(os.getenv("HOME", ""), "torch-deepimportance", "datasets", "MNIST")

def test_default_args_parsing():
    sys.argv = ["script.py"]
    args = parse_args()
    assert hasattr(args, "model")
    assert args.model in ["lenet", "vgg16", "resnet18"]
    assert args.saved_model == "/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth"
    assert args.dataset in ["mnist", "cifar10", "imagenet"]
    assert args.data_path == "./datasets/"
    assert args.importance_file == "./logs/important.json"
    assert args.epochs == 10
    assert args.device == "cpu"
    assert args.large_image is False
    assert args.random_prune is False
    assert args.use_silhouette is False
    assert args.n_clusters == 2
    assert args.top_m_neurons == 5
    assert args.batch_size == 256
    assert args.test_image == "1"
    assert args.all_class is False
    assert args.idc_test_all is False
    assert args.num_samples == 0
    assert args.attr == "lc"
    assert args.layer_index == 1
    assert args.layer_by_layer is False
    assert args.end2end is False
    assert args.logging is False
    assert args.log_path == "./logs/TestLog"
    assert args.csv_file == "demo_layer_scores.csv"


@pytest.mark.parametrize("model_name", list(MODEL_CLASSES.keys()))
def test_model_infer_cifar(model_name):
    model_fn = MODEL_CLASSES[model_name]
    model = model_fn()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, f"Model {model_name} has no parameters."
    print(f"Total number of parameters: {total_params}")
    
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, 10), f"{model_name} output shape mismatch: got {y.shape}"


@pytest.mark.skipif(
    not os.path.exists(SAVED_MODEL_PATH),
    reason="LeNet saved model file not found; skipping test."
)
def test_model_load_lenet():
    model, module_name, module = get_model(load_model_path=SAVED_MODEL_PATH)
    assert model is not None
    assert module_name is not None
    assert module is not None
    print("LeNet Model: ", len(module_name))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")


@pytest.mark.parametrize("data_path, batch_size", [
    (os.path.expanduser(SAVED_DATA_PATH), 32),
])
def test_load_mnist(data_path, batch_size):
    if not os.path.exists(data_path):
        pytest.skip(f"MNIST data path does not exist: {data_path}")

    trainloader, testloader, train_dataset, test_dataset, classes = load_MNIST(
        batch_size=batch_size,
        root=data_path,
        channel_first=False,
        train_all=False
    )

    images, labels = next(iter(trainloader))
    assert labels.shape[0] == batch_size, "Batch size mismatch"
    assert images.shape == (batch_size, 1, 32, 32), "Image shape mismatch"
    assert classes[0] == "0", "Expected class label '0'"
    

def test_dynamic_clustering_idc_end2end():
        # Prepare dummy model and input
        model = LeNet()
        model.eval()
        
        batch_size = 32
        num_classes = 10
        image_size = (3, 32, 32)  # MNIST was adapted to 3-channel 32x32 in this project

        # Generate synthetic data
        torch.manual_seed(0)
        images = torch.randn(batch_size, *image_size)  # Synthetic training images
        labels = torch.randint(0, num_classes, (batch_size,))  # Synthetic labels
        test_image = torch.randn(1, *image_size)  # Single test image
        test_label = torch.randint(0, num_classes, (1,))   # Single label

        classes = [str(i) for i in range(num_classes)]

        # Simulate attribution
        attribution = get_relevance_scores_for_all_layers(
            model, images, labels, device='cpu', attribution_method='lrp'
        )

        # Instantiate IDC
        idc = IDC(
            model=model,
            classes=classes,
            top_m_neurons=10,
            n_clusters=2,
            use_silhouette=True,
            test_all_classes=True,
            clustering_method_name='KMeans'
        )

        important_neuron_indices, inorderd_neuron_indices = idc.select_top_neurons_all(attribution, 'fc3')
        assert important_neuron_indices is not None and len(important_neuron_indices) > 0
        
        activation_values, selected_activations = idc.get_activation_values_for_model(
            images, classes[labels[0]], important_neuron_indices
        )
        assert activation_values['conv1'].shape == (batch_size, 6, 28, 28)
        assert activation_values['fc3'].shape == (batch_size, 10)
        assert selected_activations is not None
        
        cluster_groups = idc.cluster_activation_values_all(selected_activations)

        unique_cluster, coverage_rate = idc.compute_idc_test_whole(
            test_image,
            test_label,
            important_neuron_indices,
            cluster_groups,
            'lrp'
        )
        assert 0.0 <= coverage_rate <= 1.0
        assert isinstance(unique_cluster, (set, tuple, list))
        assert len(unique_cluster) >= 1
        
@pytest.mark.parametrize("data_path,saved_model_path,test_image_index,batch_size", [
    (
        os.path.expanduser(SAVED_DATA_PATH),
        os.path.expanduser(SAVED_MODEL_PATH),
        '1',
        16
    ),
])
def test_dynamic_clustering_idc_end2end_mnist(data_path, saved_model_path, test_image_index, batch_size):
    if not os.path.exists(data_path):
        pytest.skip(f"MNIST data path does not exist: {data_path}")
    if not os.path.exists(saved_model_path):
        pytest.skip(f"Saved model path does not exist: {saved_model_path}")
    # Load MNIST dataset 
    trainloader, testloader, train_dataset, test_dataset, classes = load_MNIST(
        batch_size=batch_size,
        root=data_path,
        channel_first=False,
        train_all=False
    )
    # Load model
    model, module_name, module = get_model(load_model_path=saved_model_path)
    assert model is not None
    assert module_name is not None
    assert module is not None
    print("LeNet Model: ", len(module_name))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    # Get trainable modules
    trainable_module, trainable_module_name = get_trainable_modules_main(model)
    assert trainable_module is not None
    assert trainable_module_name is not None
    
    # Get test image and label
    test_image, test_label = get_class_data(testloader, classes, test_image_index)
    assert test_image is not None
    assert test_label is not None
    
    # Get training images and labels
    images, labels = get_class_data(trainloader, classes, test_image_index)
    assert images is not None
    assert labels is not None
    
    # Simulate attribution
    attribution = get_relevance_scores_for_all_layers(
        model, images, labels, device='cpu', attribution_method='lrp'
    )
    assert attribution is not None
    
    # Instantiate IDC
    idc = IDC(
        model=model,
        classes=classes,
        top_m_neurons=10,
        n_clusters=2,
        use_silhouette=True,
        test_all_classes=True,
        clustering_method_name='KMeans'
    )
    important_neuron_indices, inorderd_neuron_indices = idc.select_top_neurons_all(attribution, 'fc3')
    assert important_neuron_indices is not None
    assert len(important_neuron_indices) > 0
    
    activation_values, selected_activations = idc.get_activation_values_for_model(
        images, classes[labels[0]], important_neuron_indices
    )
    assert activation_values['conv1'].shape == (len(images), 6, 28, 28)
    assert activation_values['fc3'].shape == (len(images), 10)
    assert selected_activations is not None
    
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    assert cluster_groups is not None
    assert isinstance(cluster_groups, (set, tuple, list))
    assert len(cluster_groups) >= 1
    unique_cluster, coverage_rate = idc.compute_idc_test_whole(
        test_image,
        test_label,
        important_neuron_indices,
        cluster_groups,
        'lrp'
    )
    assert 0.0 <= coverage_rate <= 1.0
    assert isinstance(unique_cluster, (set, tuple, list))
    assert len(unique_cluster) >= 1
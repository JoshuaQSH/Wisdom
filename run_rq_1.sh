#!/bin/bash

RUN_PRE=$@
DATA_PATH='/data/shenghao/dataset/'

## Pre trained for RQ1 [run only once]

if [ $RUN_PRE -eq 1 ]; then
    echo "Running Pre-trained models for RQ1"
    # lenet MNIST top-N
    python3 prepare_data.py --data-path $DATA_PATH --saved-model '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth' --dataset mnist --batch-size 32 --end2end --model lenet --device 'cuda:0' --top-m-neurons 6 --csv-file lenet_mnist_b32 --logging --log-path './logs/PreLeNetMNISTTop-6'

    # lenet CIFAR10 top-N
    python3 prepare_data.py --data-path $DATA_PATH --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --batch-size 32 --end2end --model lenet --device 'cuda:0' --top-m-neurons 6 --csv-file lenet_cifar_b32 --logging --log-path './logs/PreLeNetCIFARTop-6'

    # VGG16 CIFAR10 top-N
    python3 prepare_data.py --data-path $DATA_PATH --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --batch-size 32 --end2end --model vgg16 --device 'cuda:0' --top-m-neurons 6 --csv-file vgg16_cifar_b32 --logging --log-path './logs/PreVGG16CIFARTop-6'

    # ResNet18 CIFAR10 top-N
    python3 prepare_data.py --data-path $DATA_PATH --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_CIFAR10_whole.pth' --dataset cifar10 --batch-size 32 --end2end --model resnet18 --device 'cuda:0' --top-m-neurons 6 --csv-file resnet18_cifar_b32 --logging --log-path './logs/PreResNet18CIFARTop-6'

    # ResNet18 ImageNet top-N
    python3 prepare_data.py --data-path $DATA_PATH --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_IMAGENET_whole.pth' --dataset imagenet --batch-size 8 --end2end --model resnet18 --top-m-neurons 6 --csv-file resnet18_imagenet_b8 --device 'cuda:0' --logging --log-path './logs/PreResNet18ImageNetTop-6'
else
    echo "Skip Pre-trained models for RQ1"
fi

# Results for RQ1
python run_rq_1_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path $DATA_PATH --batch-size 32 --device 'cuda:0'  --csv-file './saved_files/pre_csv/lenet_cifar_b32.csv'
python run_rq_1_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth' --dataset mnist --data-path $DATA_PATH --batch-size 32 --device 'cuda:0' --csv-file './saved_files/pre_csv/lenet_mnist_b32.csv'
python run_rq_1_demo.py --model vgg16 --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --data-path $DATA_PATH --batch-size 32 --device 'cuda:0' --csv-file './saved_files/pre_csv/vgg16_cifar_b32.csv'
python run_rq_1_demo.py --model resnet18 --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_CIFAR10_whole.pth' --dataset cifar10 --data-path $DATA_PATH --batch-size 32 --device 'cuda:0' --csv-file './saved_files/pre_csv/resnet18_cifar_b32.csv'
python run_rq_1_demo.py --model resnet18 --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_IMAGENET_patched_whole.pth' --dataset imagenet --data-path $DATA_PATH --batch-size 8 --device 'cuda:0' --csv-file './saved_files/pre_csv/resnet18_imagenet_b8.csv'
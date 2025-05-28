#!/bin/bash

RUN_PRE=$@

## Pre trained for RQ1 [run only once]

if [ $RUN_PRE -eq 1 ]; then
    echo "Running Pre-trained models for RQ1"
    # lenet MNIST top-N
    python3 prepare_data.py --saved-model '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth' --dataset mnist --batch-size 32 --end2end --model lenet --device 'cuda:0' --top-m-neurons 6 --n-clusters 2 --csv-file lenet_mnist_b32 --logging --log-path './logs/PreLeNetMNISTTopNew-6'

    # lenet CIFAR10 top-N
    python3 prepare_data.py --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --batch-size 32 --end2end --model lenet --device 'cuda:0' --top-m-neurons 6 --n-clusters 2 --csv-file lenet_cifar_b32 --logging --log-path './logs/PreLeNetCIFARTopNew-6'

    # VGG16 CIFAR10 top-N
    python3 prepare_data.py --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --batch-size 32 --end2end --model vgg16 --device 'cuda:0' --top-m-neurons 6 --n-clusters 2 --csv-file vgg16_cifar_b32 --logging --log-path './logs/PreVGG16CIFARTopNew-6'

    # ResNet18 CIFAR10 top-N
    python3 prepare_data.py --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_CIFAR10_whole.pth' --dataset cifar10 --batch-size 32 --end2end --model resnet18 --device 'cuda:0' --top-m-neurons 6 --n-clusters 2 --csv-file resnet18_cifar_b32 --logging --log-path './logs/PreResNet18CIFARTopNew-6'
else
    echo "Skip Pre-trained models for RQ1"
fi

# Results for RQ1
python run_rq_1_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --batch-size 32 --device 'cuda:0'  --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/lenet_cifar_b32.csv'
python run_rq_1_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth' --dataset mnist --data-path '/data/shenghao/dataset/' --batch-size 32 --device 'cuda:0' --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/lenet_mnist_b32.csv'
python run_rq_1_demo.py --model vgg16 --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --batch-size 32 --device 'cuda:0' --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/vgg16_cifar_b32.csv'
python run_rq_1_demo.py --model resnet18 --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --batch-size 32 --device 'cuda:0' --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/resnet18_cifar_b32.csv'
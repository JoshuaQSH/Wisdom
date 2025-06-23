#!/bin/bash

DATA_PATH='/data/shenghao/dataset/'

python run_rq_3_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth' --dataset mnist --data-path $DATA_PATH --batch-size 32 --device 'cuda:1' --csv-file './saved_files/pre_csv/lenet_mnist_b32.csv' --idc-test-all --attr lrp --top-m-neurons 10
python run_rq_3_demo.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path $DATA_PATH --batch-size 32 --device 'cuda:1' --csv-file './saved_files/pre_csv/lenet_cifar_b32.csv' --idc-test-all --attr lrp --top-m-neurons 10
python run_rq_3_demo.py --model vgg16 --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --data-path $DATA_PATH --batch-size 32 --device 'cuda:1' --csv-file './saved_files/pre_csv/vgg16_cifar_b32.csv' --idc-test-all --attr lrp --top-m-neurons 10
python run_rq_3_demo.py --model resnet18 --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_CIFAR10_whole.pth' --dataset cifar10 --data-path $DATA_PATH --batch-size 32 --device 'cuda:1' --csv-file './saved_files/pre_csv/resnet18_cifar_b32.csv' --idc-test-all --attr lrp --top-m-neurons 10
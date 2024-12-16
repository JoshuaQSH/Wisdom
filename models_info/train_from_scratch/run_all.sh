#!/usr/bin/bash
python train.py --lr=0.01 --model vgg16
python train.py --lr=0.01 --model resnet18
python train.py --lr=0.01 --model densenet
python train.py --lr=0.01 --model mobilenetv2
python train.py --lr=0.01 --model efficientnet
python train.py --lr=0.01 --model shufflenetv2

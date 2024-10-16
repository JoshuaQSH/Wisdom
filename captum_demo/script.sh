#!/usr/bin/bash

RUN_TEST=$@

if [ $RUN_TEST == "fc1" ]
then
    echo "Running LeNet with fc1 layer"
    python run_demo.py --capture-all \
        --is-cifar10 \
        --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_lenet_importance_fc1.json \
        --layer-index 3 \
        --model lenet \
        --top-m-neurons -1
elif [ $RUN_TEST == "conv1" ]
then
    echo "Running LeNet with conv1 layer"
    python run_demo.py --capture-all \
    --is-cifar10 \
    --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_lenet_importance_conv1.json \
    --layer-index 1 \
    --model lenet
elif [ $RUN_TEST == "fc2" ]
then
    echo "Running LeNet with fc2 layer"
    python run_demo.py --capture-all \
        --is-cifar10 \
        --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_lenet_importance_fc2.json \
        --layer-index 4 \
        --model lenet
elif [ $RUN_TEST == "conv2" ]
then
    echo "Running LeNet with conv2 layer"
    python run_demo.py --capture-all \
        --is-cifar10 \
        --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_lenet_importance_conv2.json \
        --layer-index 2 \
        --model lenet
else
    echo "Running a custom model with fixed layer"
    python run_demo.py --model custom --is-cifar10 --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_importance.json
fi
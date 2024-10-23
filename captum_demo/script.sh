#!/usr/bin/bash

RUN_TEST=$@
CLASSES=('plane' 'car' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'ship' 'truck')
METHODS=('lc' 'la' 'ii' 'lgxa' 'lgc' 'ldl' 'ldls' 'lgs' 'lig' 'lfa' 'lrp')
TEST_ATTR="lgxa"
TEST_CLASS="plane"
TEST_MODEL="lenet"
NUM_CLUSTERS=2
NUM_NEURONS=5

if [ $RUN_TEST == "fc1" ]
then
    echo "Running LeNet with fc1 layer"
    python run_demo.py \
        --is-cifar10 \
        --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_lenet_${TEST_ATTR}_fc1.json \
        --layer-index 3 \
        --model lenet \
        --top-m-neurons 10 \
        --n-clusters 2 \
        --attr $TEST_ATTR > output_file.txt
elif [ $RUN_TEST == "conv1" ]
then
    echo "Running LeNet with conv1 layer"
    python run_demo.py --capture-all \
    --is-cifar10 \
    --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_lenet_${TEST_ATTR}_conv1.json \
    --layer-index 1 \
    --top-m-neurons -1 \
    --model lenet \
    --attr $TEST_ATTR
elif [ $RUN_TEST == "fc2" ]
then
    echo "Running LeNet with fc2 layer"
    python run_demo.py --capture-all \
        --is-cifar10 \
        --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_lenet_${TEST_ATTR}_fc2.json \
        --layer-index 4 \
        --top-m-neurons 10 \
        --n-clusters 2 \
        --model lenet \
        --attr $TEST_ATTR
elif [ $RUN_TEST == "conv2" ]
then
    echo "Running LeNet with conv2 layer"
    python run_demo.py --capture-all \
        --is-cifar10 \
        --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_lenet_${TEST_ATTR}_conv2.json \
        --layer-index 2 \
        --top-m-neurons -1 \
        --model lenet \
        --attr $TEST_ATTR
elif [ $RUN_TEST == "cifartest" ]
then
    echo "Running LeNet with fc1 layer"
    for class in "${CLASSES[@]}"
    do
        echo "--- Processing class: $class ---"
        python run_demo.py --capture-all \
            --is-cifar10 \
            --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_lenet_${TEST_ATTR}_fc1.json \
            --layer-index 3 \
            --model lenet \
            --top-m-neurons 5 \
            --n-clusters 2 \
            --test-image $class \
            --attr $TEST_ATTR
    done
elif [ $RUN_TEST == "cifarmethods" ]
then
    echo "Running LeNet with fc1 layer"
    for attribution in "${METHODS[@]}"
    do
        echo "--- Processing class: $attribution ---"
        python run_demo.py --capture-all \
            --is-cifar10 \
            --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_lenet_${attribution}_fc1.json \
            --layer-index 3 \
            --model $TEST_MODEL \
            --top-m-neurons $NUM_NEURONS \
            --n-clusters $NUM_CLUSTERS \
            --test-image $TEST_CLASS \
            --attr $attribution > ${TEST_MODEL}_${TEST_CLASS}_N-${NUM_NEURONS}_C-${NUM_CLUSTERS}.log
    done
else
    echo "Running a custom model with fixed layer"
    python run_demo.py --model custom --is-cifar10 --importance-file /home/shenghao/torch-deepimportance/captum_demo/saved_files/plane_importance.json
fi
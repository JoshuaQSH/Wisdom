#!/usr/bin/bash

RUN_TEST=$@
CLASSES=('plane' 'car' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'ship' 'truck')
# Check "lgc"
METHODS=('lc' 'la' 'ii' 'lgxa' 'lgc' 'ldl' 'ldls' 'lgs' 'lig' 'lfa' 'lrp')
TEST_ATTR="lrp"
DATASET="cifar10"
TEST_CLASS="plane"
TEST_MODEL="lenet"
NUM_CLUSTERS=2
NUM_NEURONS=10
I_PATH=/home/shenghao/torch-deepimportance

if [ $RUN_TEST == "fc1" ]
then
    echo "Running LeNet with fc1 layer"
    # layer-index 3 is the fc1 layer
    python run.py \
        --dataset cifar10 \
        --batch-size 4 \
        --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_fc1.json \
        --layer-index 3 \
        --model $TEST_MODEL \
        --top-m-neurons $NUM_NEURONS \
        --n-clusters $NUM_CLUSTERS \
        --attr ldls

elif [ $RUN_TEST == "end2end" ]
then
    echo "Running ${TEST_MODEL} End2End analysis"
    python run.py \
        --dataset $DATASET \
        --batch-size 4 \
        --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_end2end.json \
        --model $TEST_MODEL \
        --top-m-neurons $NUM_NEURONS \
        --n-clusters $NUM_CLUSTERS \
        --attr $TEST_ATTR \
        --end2end

elif [ $RUN_TEST == "conv1" ]
then
    echo "Running LeNet with conv1 layer"
    python run.py --dataset $DATASET \
    --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_conv1.json \
    --layer-index 1 \
    --top-m-neurons 2 \
    --model $TEST_MODEL \
    --attr $TEST_ATTR

elif [ $RUN_TEST == "fc2" ]
then
    echo "Running LeNet with fc2 layer"
    python run.py --dataset $DATASET \
        --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_fc2.json \
        --layer-index 4 \
        --top-m-neurons 10 \
        --n-clusters 2 \
        --model $TEST_MODEL \
        --attr $TEST_ATTR

elif [ $RUN_TEST == "conv2" ]
then
    echo "Running LeNet with conv2 layer"
    python run.py --dataset $DATASET \
        --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_conv2.json \
        --layer-index 2 \
        --top-m-neurons 2 \
        --model $TEST_MODEL \
        --attr $TEST_ATTR

elif [ $RUN_TEST == "cifartest" ]
then
    echo "Running LeNet with fc1 layer"
    for class in "${CLASSES[@]}"
    do
        echo "--- Processing class: $class ---"
        python run.py --capture-all \
            --dataset $DATASET \
            --large-image \
            --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_fc1.json \
            --layer-index 3 \
            --model $TEST_MODEL \
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
        python run.py --dataset $DATASET \
            --large-image \
            --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${attribution}_fc1.json \
            --layer-index 3 \
            --model $TEST_MODEL \
            --top-m-neurons $NUM_NEURONS \
            --n-clusters $NUM_CLUSTERS \
            --test-image $TEST_CLASS \
            --attr $attribution
        echo "Done with $attribution"
    done

elif [ $RUN_TEST == "imagenet" ]
then
    echo "Running ImageNet with vgg16"
    python run.py --capture-all \
        --dataset imagenet \
        --importance-file ${I_PATH}/saved_files/plane_vgg16_${TEST_ATTR}_fc1.json \
        --layer-index 3 \
        --model vgg16 \
        --top-m-neurons 5 \
        --n-clusters 2 \
        --test-image $TEST_CLASS \
        --attr $TEST_ATTR

elif [ $RUN_TEST == "attrtest" ]
then
    echo "Running ${DATASET} with ${TEST_MODEL}-FC1 and test each attribution"
    for attribution in "${METHODS[@]}"
    do
        echo "--- Processing class: $attribution ---"
        python run.py --dataset $DATASET \
            --layer-index 3 \
            --model $TEST_MODEL \
            --top-m-neurons $NUM_NEURONS \
            --n-clusters $NUM_CLUSTERS \
            --test-image $TEST_CLASS \
            --attr $attribution
        echo "Done with $attribution"
    done

# 1->conv1, 2->conv2, 3->fc1, 4->fc2
elif [ $RUN_TEST == "attr4class" ]
then
    echo "Running ${DATASET} with ${TEST_MODEL}-FC1 and test each attribution"
    python eval_attr.py \
        --dataset $DATASET \
        --batch-size 4 \
        --layer-index 3 \
        --model $TEST_MODEL \
        --top-m-neurons 10 \
        --n-clusters $NUM_CLUSTERS \
        --capture-all \
        --logging
        
elif [ $RUN_TEST == "attr4class2" ]
then
    echo "Running ${DATASET} with ${TEST_MODEL}-Conv2 and test each attribution"
    python eval_attr.py \
        --dataset $DATASET \
        --batch-size 4 \
        --layer-index 2 \
        --model $TEST_MODEL \
        --top-m-neurons 2 \
        --n-clusters $NUM_CLUSTERS \
        --capture-all \
        --logging

elif [ $RUN_TEST == "attr4class3" ]
then
    echo "Running ${DATASET} with ${TEST_MODEL}-FC2 and test each attribution"
    python eval_attr.py \
        --dataset $DATASET \
        --batch-size 4 \
        --layer-index 4 \
        --model $TEST_MODEL \
        --top-m-neurons $NUM_NEURONS \
        --n-clusters $NUM_CLUSTERS \
        --capture-all \
        --logging

elif [ $RUN_TEST == "attr4class4" ]
then
    echo "Running ${DATASET} with ${TEST_MODEL}-Conv1 and test each attribution"
    python eval_attr.py \
        --dataset $DATASET \
        --batch-size 4 \
        --layer-index 1 \
        --model $TEST_MODEL \
        --top-m-neurons 1 \
        --n-clusters $NUM_CLUSTERS \
        --capture-all \
        --logging

else
    echo "Running a custom model with fixed layer"
    python run.py --model custom --large-image --importance-file /home/shenghao/torch-deepimportance/saved_files/plane_importance.json
fi
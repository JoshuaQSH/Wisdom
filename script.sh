#!/usr/bin/bash

RUN_TEST=$@
CLASSES=('plane' 'car' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'ship' 'truck')
# Check "lgc"
METHODS=('lc' 'la' 'ii' 'lgxa' 'lgc' 'ldl' 'ldls' 'lgs' 'lig' 'lfa' 'lrp')
TEST_ATTR="la"
# cifar10, mnist, imagenet
DATASET="cifar10"
TEST_CLASS="plane"
TEST_MODEL="lenet"
NUM_CLUSTERS=2
NUM_NEURONS=10
I_PATH=/home/shenghao/torch-deepimportance

# For the CIFAR-10 dataset and LeNet model, parts of the unit tests
# 0-'conv1', 1-'conv2', 2-'fc1', 3-'fc2', 4-'fc3'
if [ $RUN_TEST == "lenetfc1" ]
then
    echo "Running LeNet with fc1 layer"
    # layer-index 3 is the fc1 layer
    python run.py \
        --dataset $DATASET \
        --batch-size 4 \
        --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_fc1.json \
        --layer-index 2 \
        --model $TEST_MODEL \
        --top-m-neurons $NUM_NEURONS \
        --n-clusters $NUM_CLUSTERS \
        --attr $TEST_ATTR \
        --all-class \
        --idc-test-all

elif [ $RUN_TEST == "othernet" ]
then
    layerindex=4
    testmodel="efficientnet"
    echo "Running ${testmodel} with Layer Index ${layerindex}"
    python run.py \
        --dataset $DATASET \
        --batch-size 4 \
        --layer-index $layerindex \
        --model $testmodel \
        --top-m-neurons 5 \
        --n-clusters $NUM_CLUSTERS \
        --attr lrp \
        --saved-model ${testmodel}_CIFAR10-new.pt \
        --test-image cat

elif [ $RUN_TEST == "lenetfc2" ]
then
    echo "Running LeNet with fc2 layer"
    python run.py \
        --dataset $DATASET \
        --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_fc2.json \
        --layer-index 3 \
        --top-m-neurons 10 \
        --n-clusters $NUM_CLUSTERS \
        --model $TEST_MODEL \
        --attr $TEST_ATTR

elif [ $RUN_TEST == "lenetconv2" ]
then
    echo "Running LeNet with conv2 layer"
    python run.py \
        --dataset $DATASET \
        --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_conv2.json \
        --layer-index 0 \
        --top-m-neurons 3 \
        --n-clusters $NUM_CLUSTERS \
        --model $TEST_MODEL \
        --attr $TEST_ATTR

elif [ $RUN_TEST == "lenetconv1" ]
then
    echo "Running LeNet with conv1 layer"
    python run.py \
    --dataset $DATASET \
    --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_conv1.json \
    --layer-index 1 \
    --top-m-neurons 2 \
    --n-clusters $NUM_CLUSTERS \
    --model $TEST_MODEL \
    --attr $TEST_ATTR

elif [ $RUN_TEST == "lenet-1" ]
then
    topkneurons=(2 5 10 15 20)
    for num in "${topkneurons[@]}";do
        for meth in "${METHODS[@]}";do
            echo "Running LeNet with FC1 layer, top-m-neurons: $num, method: $meth"
            python run.py \
                --dataset $DATASET \
                --batch-size 4 \
                --layer-index 2 \
                --top-m-neurons $num \
                --model $TEST_MODEL \
                --n-clusters $NUM_CLUSTERS \
                --attr $meth \
                --all-class \
                --idc-test-all \
                --logging
        done
    done

elif [ $RUN_TEST == "lenet-2" ]
then
    topkneurons=(2 5 10 15 20)
    for num in "${topkneurons[@]}";do
        for meth in "${METHODS[@]}";do
            echo "Running LeNet with FC2 layer, top-m-neurons: $num, method: $meth"
            python run.py \
                --dataset $DATASET \
                --batch-size 4 \
                --layer-index 3 \
                --top-m-neurons $num \
                --model $TEST_MODEL \
                --n-clusters $NUM_CLUSTERS \
                --attr $meth \
                --all-class \
                --idc-test-all \
                --logging
        done
    done

elif [ $RUN_TEST == "lenet-3" ]
then
    topkneurons=(2 3 5)
    for num in "${topkneurons[@]}";do
        for meth in "${METHODS[@]}";do
            echo "Running LeNet with Conv1 layer, top-m-neurons: $num, method: $meth"
            python run.py \
                --dataset $DATASET \
                --batch-size 4 \
                --layer-index 0 \
                --top-m-neurons $num \
                --model $TEST_MODEL \
                --n-clusters $NUM_CLUSTERS \
                --attr $meth \
                --all-class \
                --idc-test-all \
                --logging
        done
    done

elif [ $RUN_TEST == "lenet-4" ]
then
    topkneurons=(2 3 5)
    for num in "${topkneurons[@]}";do
        for meth in "${METHODS[@]}";do
            echo "Running LeNet with Conv2 layer, top-m-neurons: $num, method: $meth"
            python run.py \
                --dataset $DATASET \
                --batch-size 4 \
                --layer-index 1 \
                --top-m-neurons $num \
                --model $TEST_MODEL \
                --n-clusters $NUM_CLUSTERS \
                --attr $meth \
                --all-class \
                --idc-test-all \
                --logging
        done
    done

elif [ $RUN_TEST == "lenetallclass" ]
then
    echo "Running LeNet with fc1 layer"
    for class in "${CLASSES[@]}"
    do
        echo "--- Processing class: $class ---"
        python run.py --dataset $DATASET \
            --large-image \
            --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_fc1.json \
            --layer-index 3 \
            --model $TEST_MODEL \
            --top-m-neurons 5 \
            --n-clusters 2 \
            --test-image $class \
            --attr $TEST_ATTR
    done

elif [ $RUN_TEST == "end2end" ]
then
    echo "Running ${TEST_MODEL} End2End analysis"
    for class in "${CLASSES[@]}"
    do
        echo "--- Processing class: $class ---"
        python run.py \
            --dataset $DATASET \
            --batch-size 4 \
            --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_end2end.json \
            --model $TEST_MODEL \
            --top-m-neurons $NUM_NEURONS \
            --n-clusters $NUM_CLUSTERS \
            --attr $TEST_ATTR \
            --test-image $class \
            --saved-model ${TEST_MODEL}_CIFAR10-new.pt \
            --end2end
    done

elif [ $RUN_TEST == "end2endSelect" ]
then
    echo "Running ${TEST_MODEL} End2End analysis"
    for class in "${CLASSES[@]}"
    do
        echo "--- Processing class: $class ---"
        if [ $class == "plane" ]
        then
            CHOOSE_ATTR="lrp"
        elif [ $class == "car" ]
        then
            CHOOSE_ATTR="ii"
        elif [ $class == "bird" ]
        then
            CHHOSE_ATTR="lrp"
        elif [ $class == "cat" ]
        then
            CHOOSE_ATTR="lrp"
        elif [ $class == "deer" ]
        then
            CHOOSE_ATTR="ldl"
        elif [ $class == "dog" ]
        then
            CHOOSE_ATTR="lrp"
        elif [ $class == "frog" ]
        then
            CHOOSE_ATTR="lrp"
        elif [ $class == "horse" ]
        then
            CHOOSE_ATTR="la"
        elif [ $class == "ship" ]
        then
            CHOOSE_ATTR="lrp"
        elif [ $class == "truck" ]
        then
            CHOOSE_ATTR="lfa"
        fi
        python run.py \
            --dataset $DATASET \
            --batch-size 4 \
            --importance-file ${I_PATH}/saved_files/plane_${TEST_MODEL}_${TEST_ATTR}_end2end.json \
            --model lenet \
            --top-m-neurons $NUM_NEURONS \
            --n-clusters $NUM_CLUSTERS \
            --attr $CHOOSE_ATTR \
            --test-image $class \
            --end2end \
            --logging
    done

# TODO: For the ImageNet and more models
# TODO: 1) test three models with different attributions
# TODO: 2) test one label and then all the labels with --test-all
elif [ $RUN_TEST == "imagenet" ]
then
    # ['vgg16', 'convnext_base', 'efficientnet_v2_s', 'mobilenet_v3_small', 'efficientnet_v2_s']  
    IMAGE_MODEL="resnet18"
    IMAGE_CLASS="tench"
    echo "Running ImageNet with ${IMAGE_MODEL}"
    python run.py --dataset imagenet \
        --importance-file ${I_PATH}/saved_files/${IMAGE_MODEL}_${TEST_ATTR}.json \
        --layer-index 4 \
        --batch-size 4 \
        --model $IMAGE_MODEL \
        --top-m-neurons 10 \
        --n-clusters 2 \
        --test-image $IMAGE_CLASS \
        --attr $TEST_ATTR

elif [ $RUN_TEST == "imagenet-all" ]
then
    # ['vgg16', 'convnext_base', 'efficientnet_v2_s', 'mobilenet_v3_small', 'efficientnet_v2_s']  
    IMAGE_MODEL="resnet18"
    IMAGE_CLASS="tench"
    echo "Running ImageNet with ${IMAGE_MODEL}"
    python run.py --dataset imagenet \
        --importance-file ${I_PATH}/saved_files/${IMAGE_MODEL}_${TEST_ATTR}.json \
        --layer-index 4 \
        --batch-size 4 \
        --model $IMAGE_MODEL \
        --top-m-neurons 3 \
        --n-clusters 2 \
        --test-image $IMAGE_CLASS \
        --attr $TEST_ATTR \
        --test-all

## Finegrained evaluations with pruning and attributions selection
# 0-'conv1', 1-'conv2', 2-'fc1', 3-'fc2', 4-'fc3'
elif [ $RUN_TEST == "attr4demo" ]
then
    echo "Running ${DATASET} with ${TEST_MODEL} and test each attribution"
    python selector_demo.py \
        --dataset $DATASET \
        --batch-size 128 \
        --layer-index 0 \
        --model lenet \
        --top-m-neurons 10 \
        --n-clusters $NUM_CLUSTERS \
        --test-image plane \
        --all-attr \
        --logging

    python selector_demo.py \
        --dataset $DATASET \
        --batch-size 128 \
        --layer-index 1 \
        --model lenet \
        --top-m-neurons 10 \
        --n-clusters $NUM_CLUSTERS \
        --test-image plane \
        --all-attr \
        --logging
    
    python selector_demo.py \
        --dataset $DATASET \
        --batch-size 128 \
        --layer-index 2 \
        --model lenet \
        --top-m-neurons 10 \
        --n-clusters $NUM_CLUSTERS \
        --test-image plane \
        --all-attr \
        --logging
    
    python selector_demo.py \
        --dataset $DATASET \
        --batch-size 128 \
        --layer-index 3 \
        --model lenet \
        --top-m-neurons 10 \
        --n-clusters $NUM_CLUSTERS \
        --test-image plane \
        --all-attr \
        --logging

elif [ $RUN_TEST == "attr4lenetf1" ]
then
    topkneurons=(2 5 10 15 20)
    echo "Running ${DATASET} with ${TEST_MODEL}-FC1 and test each attribution"
    for num in "${topkneurons[@]}";do
        python eval_attr.py \
            --dataset $DATASET \
            --batch-size 4 \
            --layer-index 2 \
            --model $TEST_MODEL \
            --top-m-neurons $num \
            --n-clusters $NUM_CLUSTERS \
            --all-attr \
            --logging
    done

elif [ $RUN_TEST == "attr4lenetf2" ]
then
    topkneurons=(2 5 10 15 20)
    echo "Running ${DATASET} with ${TEST_MODEL}-FC2 and test each attribution"
    for num in "${topkneurons[@]}";do
        python eval_attr.py \
            --dataset $DATASET \
            --batch-size 4 \
            --layer-index 3 \
            --model $TEST_MODEL \
            --top-m-neurons $num \
            --n-clusters $NUM_CLUSTERS \
            --all-attr
    done

elif [ $RUN_TEST == "attr4lenetc1" ]
then
    topkneurons=(2 3 5)
    echo "Running ${DATASET} with ${TEST_MODEL}-Conv1 and test each attribution"
    for num in "${topkneurons[@]}";do
        python eval_attr.py \
            --dataset $DATASET \
            --batch-size 4 \
            --layer-index 0 \
            --model $TEST_MODEL \
            --top-m-neurons $num \
            --n-clusters $NUM_CLUSTERS \
            --all-attr \
            --logging
    done    
    
elif [ $RUN_TEST == "attr4lenetc2" ]
then
    topkneurons=(2 3 5)
    echo "Running ${DATASET} with ${TEST_MODEL}-Conv2 and test each attribution"
    for num in "${topkneurons[@]}";do
        python eval_attr.py \
            --dataset $DATASET \
            --batch-size 4 \
            --layer-index 1 \
            --model $TEST_MODEL \
            --top-m-neurons $num \
            --n-clusters $NUM_CLUSTERS \
            --all-attr \
            --logging
    done
else
    echo "Running a custom model with fixed layer"
    python run.py --model custom --large-image --importance-file /home/shenghao/torch-deepimportance/saved_files/plane_importance.json
fi
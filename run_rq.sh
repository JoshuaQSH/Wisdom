#!/bin/bash

# Usage:
# ./run_rq.sh --rq [1|2|3|4|5] [--pretrain 1] [--wisdom 1]

DATA_PATH='/data/shenghao/dataset/'
DEVICE_CUDA='cuda:0'
DEVICE_CPU='cpu'

run_rq1() {
    if [ "$PRETRAIN" -eq 1 ]; then
        echo "Running pre-trained setup for RQ1"
        python3 prepare_data.py --data-path $DATA_PATH --saved-model '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth' --dataset mnist --batch-size 128 --end2end --model lenet --device $DEVICE_CUDA --top-m-neurons 10 --csv-file lenet_mnist  --use-silhouette --logging --log-path './logs/PreLeNetMNISTTop-10'
        python3 prepare_data.py --data-path $DATA_PATH --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --batch-size 128 --end2end --model lenet --device $DEVICE_CUDA --top-m-neurons 10 --csv-file lenet_cifar --use-silhouette --logging --log-path './logs/PreLeNetCIFARTop-10'
        python3 prepare_data.py --data-path $DATA_PATH --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --batch-size 32 --end2end --model vgg16 --device $DEVICE_CUDA --top-m-neurons 10 --csv-file vgg16_cifar --use-silhouette --logging --log-path './logs/PreVGG16CIFARTop-10'
        python3 prepare_data.py --data-path $DATA_PATH --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_CIFAR10_whole.pth' --dataset cifar10 --batch-size 32 --end2end --model resnet18 --device $DEVICE_CUDA --top-m-neurons 10 --csv-file resnet18_cifar --use-silhouette --logging --log-path './logs/PreResNet18CIFARTop-10'
        python3 prepare_data.py --data-path $DATA_PATH --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_IMAGENET_whole.pth' --dataset imagenet --batch-size 8 --end2end --model resnet18 --device $DEVICE_CUDA --top-m-neurons 10 --csv-file resnet18_imagenet --use-silhouette --logging --log-path './logs/PreResNet18ImageNetTop-10'
    else
        echo "Skipping pre-training for RQ1"
    fi

    for entry in \
        "lenet mnist 128 $DEVICE_CUDA" \
        "lenet cifar10 128 $DEVICE_CUDA" \
        "vgg16 cifar10 32 $DEVICE_CUDA" \
        "resnet18 cifar10 32 $DEVICE_CUDA" \
        "resnet18 imagenet 8 $DEVICE_CUDA"
    do
        set -- $entry
        python run_rq_1_demo.py --model $1 --saved-model "/torch-deepimportance/models_info/saved_models/${1}_${2^^}_whole.pth" --dataset $2 --data-path $DATA_PATH --batch-size $3 --device $4 --top-m-neurons 10 --csv-file "./saved_files/pre_csv/${1}_${2}.csv" --logging --log-path "./logs/RQ1_${1}_${2}Log"
    done
}

run_rq2() {
    ATTR="lrp"
    DEVICE=$DEVICE_CUDA
    if [ "$WISDOM" -eq 1 ]; then
        ATTR="wisdom"
        DEVICE=$DEVICE_CUDA
        echo "Running Wisdom-driven perturbation for RQ2"
    else
        echo "Running LRP-driven perturbation for RQ2"
    fi

    for entry in \
        "lenet mnist 128" \
        "lenet cifar10 128" \
        "vgg16 cifar10 32" \
        "resnet18 cifar10 32"
    do
        set -- $entry
        python run_rq_2_demo.py --model $1 --saved-model "/torch-deepimportance/models_info/saved_models/${1}_${2^^}_whole.pth" --dataset $2 --data-path $DATA_PATH --batch-size $3 --device $DEVICE --csv-file "./saved_files/pre_csv/${1}_${2}.csv" --attr $ATTR --top-m-neurons 10 --use-silhouette --logging --log-path "./logs/RQ2_${1}_${2}Log"
    done
}

run_rq3() {
    echo "Running RQ3"
    for entry in \
        "lenet mnist 128" \
        "lenet cifar10 128" \
        "vgg16 cifar10 32" \
        "resnet18 cifar10 32"
    do
        set -- $entry
        python run_rq_3_demo.py --model $1 --saved-model "/torch-deepimportance/models_info/saved_models/${1}_${2^^}_whole.pth" --dataset $2 --data-path $DATA_PATH --batch-size $3 --device $DEVICE_CUDA --csv-file "./saved_files/pre_csv/${1}_${2}.csv" --attr lrp --top-m-neurons 10 --use-silhouette --logging --log-path "./logs/RQ3_${1}_${2}Log"
    done
}

run_rq4() {
    echo "Running RQ4"
    for entry in \
        "lenet mnist 128 $DEVICE_CUDA" \
        "lenet cifar10 128 $DEVICE_CUDA" \
        "vgg16 cifar10 32 $DEVICE_CPU" \
        "resnet18 cifar10 32 $DEVICE_CPU"
    do
        set -- $entry
        python run_rq_4_demo.py --model $1 --saved-model "/torch-deepimportance/models_info/saved_models/${1}_${2^^}_whole.pth" --dataset $2 --data-path $DATA_PATH --batch-size $3 --device $4 --csv-file "./saved_files/pre_csv/${1}_${2}.csv" --attr lrp --top-m-neurons 10 --use-silhouette --logging --log-path "./logs/RQ4_${1}_${2}Log"
    done
}

run_rq5() {
    echo "Running RQ5"
    for entry in \
        "lenet mnist 128 $DEVICE_CUDA" \
        "lenet cifar10 64 $DEVICE_CUDA" \
        "lenet synthetic 128 $DEVICE_CUDA" \
        "lenet synthetic 128 $DEVICE_CPU" \
        "vgg16 synthetic 256 $DEVICE_CUDA" \
        "vgg16 synthetic 128 $DEVICE_CUDA" \
        "vgg16 synthetic 64 $DEVICE_CUDA" \
        "vgg16 synthetic 32 $DEVICE_CUDA" \
        "vgg16 synthetic 16 $DEVICE_CUDA" \
        "vgg16 synthetic 128 $DEVICE_CPU" \
        "resnet18 synthetic 128 $DEVICE_CUDA" \
        "resnet18 synthetic 128 $DEVICE_CPU"
    do
        set -- $entry
        if [[ "$2" == "synthetic" ]]; then
            CSV_FILE="./saved_files/pre_csv/${1}_cifar10.csv"
            SAVE_FILE="/torch-deepimportance/models_info/saved_models/${1}_cifar10_whole.pth"
        else
            CSV_FILE="./saved_files/pre_csv/${1}_${2}.csv"
            SAVE_FILE="/torch-deepimportance/models_info/saved_models/${1}_${2^^}_whole.pth"
        fi
        python run_rq_5_demo.py --model $1 --saved-model $SAVE_FILE --dataset $2 --data-path $DATA_PATH --batch-size $3 --device $4 --csv-file $CSV_FILE --attr lrp --top-m-neurons 10 --logging --log-path "./logs/RQ5_b${3}_${4}Log"
    done
}

# Parse arguments
RQ=0
PRETRAIN=0
WISDOM=0

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --rq)
            RQ="$2"
            shift 2
            ;;
        --pretrain)
            PRETRAIN="$2"
            shift 2
            ;;
        --wisdom)
            WISDOM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

case $RQ in
    1) run_rq1 ;;
    2) run_rq2 ;;
    3) run_rq3 ;;
    4) run_rq4 ;;
    5) run_rq5 ;;
    *)
        echo "Please provide a valid --rq [1|2|3|4|5]"
        exit 1
        ;;
esac

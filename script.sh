#!/bin/bash
# Torch-DeepImportance Use Case Runner Script
# This script runs a specific Python program based on the argument provided.

# Default hyperparameter values (can be adjusted if needed)
MODEL_NAME="lenet"
SAVED_MODEL="/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth"
DATASET_NAME="cifar10"
DATASET_DIR="datasets"
DEVICE="cpu"
TOP_M=10                # N: top-m-neurons
TEST_LABEL="plane"      # Label of the test image for demo
NUM_SAMPLES=0           # M: number of samples (0 could indicate default behavior)
ATTR_METHOD="lrp"       # Attribution method name
LAYER_INDEX=1           # Model layer index to analyze

# Function to display usage instructions
usage() {
    echo "Usage: $0 {test|case1|prepare|case3}"
}

# Ensure exactly one argument is provided
if [ $# -ne 1 ]; then
    echo "Error: Exactly one argument expected."
    usage
    exit 1
fi

# Get the first argument (mode)
MODE="$1"

# Check if dataset directory exists; if not, create it
if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset directory '$DATASET_DIR' not found. Creating it..."
    mkdir -p "$DATASET_DIR"
fi

# Determine which use case to run based on the argument
case "$MODE" in
    test)
        echo "Running Test Demo example..."
        python3 run.py --model lenet --saved-model "/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth" --dataset mnist --data-path './datasets/' --importance-file 'logs/impotant.json' --device cpu --n-clusters 2 --top-m-neurons 10 --test-image 1 --idc-test-all --num-samples 0 --attr lrp --end2end --log-path 'logs/DemoLog' --logging
        ;;
    case1)
        echo "Running Use Case 1..."
        python3 run.py \
            --model $MODEL_NAME \
            --saved-model $SAVED_MODEL \
            --dataset $DATASET_NAME \
            --data-path $DATASET_DIR \
            --importance-file './logs/impotant.json' \
            --device $DEVICE \
            --top-m-neurons $TOP_M \
            --use-silhouette \
            --num-samples $NUM_SAMPLES \
            --idc-test-all \
            --test-image $TEST_LABEL \
            --attr $ATTR_METHOD \
            --layer-index $LAYER_INDEX \
            --end2end \
            --n-clusters 2 \
            --log-path 'logs/TestLog' \
            --logging
        ;;
    prepare)
        echo "Running Use Case 2: Prepare the data..."
        python3 prepare_data.py \
            --model $MODEL_NAME \
            --saved_model $SAVED_MODEL \
            --dataset $DATASET_NAME \
            --data_path $DATASET_DIR \
            --batch-size 2 \
            --device $DEVICE \
            --importance-file './logs/impotant.json' \
            --top-m-neurons $TOP_M \
            --num-samples $NUM_SAMPLES \
            --end2end \
            --log-path './logs/PrepareDataLog' \
            --logging
        ;;
    case3)
        echo "Running Use Case 3..."
        python3 run_wisdom.py \
            --model $MODEL_NAME \
            --saved_model $SAVED_MODEL \
            --dataset $DATASET_NAME \
            --data_path $DATASET_DIR \
            --device $DEVICE \
            --top-m-neurons $TOP_M \
            --use-silhouette \
            --num-samples $NUM_SAMPLES
        ;;
    *)
        # Handle invalid arguments
        echo "Invalid argument '$MODE'."
        usage
        exit 1
        ;;
esac

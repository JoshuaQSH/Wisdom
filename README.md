# WISDOM: Torch4DeepImportance and beyond

A pytorch version for Deepimportance (test version). [TensorFlow version for DeepImportance](https://github.com/DeepImportance/deepimportance_code_release/tree/ga_modifications). 

For the paper, please refer [HERE](https://zenodo.org/records/3628024).

## Prerequest

The Captum version demo is tested and should be fine for further developments. The docker is still pending.

Use `conda` or `pyvenv` to build a virtual environment.

```shell
# requriements
$ pip -r install requirements.txt

# If you are using anaconda or miniconda virtual environment, do:
$ conda env create -f requirements_venv.yaml
```

## How to run

```shell
# UC1 - Running IDC test with different attribution methods
python run.py \
      --model <model-name> \
      --saved-model <path-to-the-pretrained-pt-file> \
      --dataset <dataset-name> \
      --data-path <path-to-dataset> \
      --importance-file <path-to-save-importance-file-json> \
      --device cpu \
      # --use-silhouette \
      --n-cluster 2 \ 
      --top-m-neurons 5 \
      # --test-image plane \
      --idc-test-all \
      --num-samples 1000 \
      --attr lrp \
      # --layer-index 1 \
      --end2end \
      --all-class \
      # --class-iters \
      --log-path './logs/TestLog' \
      --logging

# Clustering, choose one
--use-silhouette # dynamic choose the cluster number
--n-cluster N # fixed cluster number

# Sampling test images
--num-sample N # Randomly pick N samples
--idc-test-all # Choose all the image in one batch

# Testing mode combinations
# We currently have four modes:
# 1. End2End with all classes [--end2end, --all-class]
# 2. End2End with single class [--end2end, --test-image N]
# 3. Single layer with all classes [--layer-index, --all-class]
# 4. Single layer with single class [--layer-index, --test-image N]
--end2end       # Activate End2End test
--all-class     # Test all the classes (samples), equal to batch shuffle testing the whole testset
--layer-index N # A specific layer that would like to be tested (works when --end2end is OFF)
--test-image M  # A specific class that would like to be tested (works when --all-class is OFF)
--class-iters   # A class-wise testing following in-ordered class testing (i.e. similar to mode 2 and 4 but will give all the class results)

# Logging
--log-path  '<path>/<name>' # log file path and name: e.g., './logs/TestLog'
--logging                   # Save the log file

# UC-1: IDC results with different attribution methods
## An LeNet Example - CIFAR10 with layer index: 1 - LRP
python3 run.py \
      --model lenet \
      --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' \
      --dataset cifar10 \
      --data-path './dataset/' \
      --use-silhouette \
      --device cpu \
      --n-cluster 2 \
      --top-m-neurons 5 \
      --test-image plane \
      --idc-test-all \
      --num-samples 0  \
      --attr lrp \
      --layer-index 1 \
      --log-path './logs/TestLog' \
      --logging

## An LeNet end2end example - CIFAR10 - LRP
python3 run.py \
      --model lenet \
      --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' \
      --dataset cifar10 \
      --data-path './dataset/' \
      --device cpu \
      --n-cluster 2 \
      --top-m-neurons 10 \
      --end2end \
      --all-class \
      --idc-test-all \
      --num-samples 0  \
      --attr lrp \
      --log-path './logs/TestLog' \
      --logging

# UC-2: WISDOM data for specific model
## Example: LeNet - MNIST - top/6
python3 prepare_data.py \
        --saved-model '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth' \
        --dataset mnist \
        --batch-size 32 \
        --end2end \
        --model lenet \
        --device 'cuda:0' \
        --top-m-neurons 6 \
        --n-clusters 2 \
        --csv-file lenet_mnist_b32 \
        --logging \
        --log-path './logs/PreLeNetMNISTTopNew-6'

# Uc - 3
# Apply the WISDOM 
python3 run_wisdom.py \
      --model lenet \
      --saved-model '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth' \
      --dataset mnist \
      --data-path './dataset/' \
      --device cpu \
      --n-cluster 2 \
      --top-m-neurons 6 \
      --end2end \
      --num-samples 0 \
      --csv-file './saved_files/pre_csv/lenet_mnist_b32.csv' \
      --idc-test-all 

```

### Running script

```shell
# End2end testing the deepimportance with LeNet-5 in MNIST dataset, with fixed cluster number (=2)
$ ./script.sh test

# Run with Use Case N: `./script.sh caseN`, e.g.:
$ ./script.sh case1
```

## Routes

- Activation values for important neuros (v_1, v_2, ...)
- Combining and voting for the best attribution methods to come up with a better important neuron set
- Clustering with Silhoutte score (or with the customized `n_cluster`)
- Combination of clusters from important neuros
- Testset comes in (x_1, y_1)
- Check coverage (See combinations covered by the test set, e.g., 4/6, 1/6 ,....), A.k.a. IDC

## TODO

- [x] [**YOLO**] Implement the [YOLOv8](https://github.com/jahongir7174/YOLOv8-pt/tree/master) (or [YOLOv5](https://github.com/mihir135/yolov5)) in pytorch, with COCO dataset
- [x] [**CI**] Pytest Running with a small demo (MNIST)
- [ ] [**CI**] Docker building
- [ ] [**Lib**] Refine the codes (Now: v0.1 -> v0.2)
- [ ] [**Lib**] pip package ready
- [ ] [**IDC**] runtime version + attention
- [ ] [**YOLO**] YOLO v11

## Research Questions

Note: While running the scripts, ensure to change the `$DATA_PATH` to your own dataset path.

### RQ 1: Critical (or important) neurons

Metrics:
- Top n in (6, 8, 10, 15, 20) neurons
- Accuracy drop based on the neurons pruning

How to run
```shell
# MODELNAME: [lenet, vgg16, resnet18]
# DATASET: [mnist, cifar10]
# Pretrained relevant scores: ./saved_files/pre_csv/<MODELNAME>_<DATASET>.csv
python run_rq_1_demo.py --model MODELNAME --saved-model /path/to/saved/model/pth --dataset DATASET --data-path /path/to/saved/datasets/ --batch-size 32 --device cpu  --csv-file /path/to/wisdom/weights/csv

# Or simply run the script we prepared
./run_rq.sh --rq 1

# Train from scratch
./run_rq.sh --rq 1 --pretrain 1
```

### RQ 2: Diversity

Metrics:
- Generate two testset for evaluations (refer to Deepimportance).
- Top 2% of the inputs perturbations (add Gaussian White Noise). Random ($U_R$) & Important pixels ($U_I$)
- Coverage rate check (expect higher in $U_I$)
- Run with 5 Iterations

Notes: 
- $U_O$: original dataset
- $U_I$: Noise for important pixels
- $U_R$: Noise for random pixels

How to run
```shell
# MODELNAME: [lenet, vgg16, resnet18]
# DATASET: [mnist, cifar10]
# Pretrained relevant scores: ./saved_files/pre_csv/<MODELNAME>_<DATASET>.csv
python run_rq_2_demo.py --model MODELNAME --saved-model /path/to/saved/model/pth --dataset DATASET --data-path /path/to/saved/datasets/ --batch-size 32 --device cpu  --csv-file /path/to/wisdom/weights/csv --idc-test-all --attr wisdom --top-m-neurons 10 --device 'cuda:0'

# With WISDOM-based pertubation
./run_rq.sh --rq 2 --wisdom 1

# With LRP-based pertubation
./run_rq.sh --rq 2 
```

### RQ 3: Effectiveness (or sensitivity)

Metrics:
- Sample 100, 1000, 3000 correct inputs in testset. 
- Replace some of the inputs (1%, 5%, 10%) with adversarial examples (crafted using PGD, FGSM and CW).
- Record the Normalization(delta(Coverage)) (expect stable improvements)

$NCov(s) = \frac{Cov(s) - Cov(s_0)}{max(\Delta) - min(\Delta)}$ <br>
$\Delta = \{Cov(s) - Cov(s_0) | s \in S\}$

How to run
```shell
# MODELNAME: [lenet, vgg16, resnet18]
# DATASET: [mnist, cifar10, imagenet]
# Pretrained relevant scores: ./saved_files/pre_csv/<MODELNAME>_<DATASET>.csv
python run_rq_3_demo.py --model MODELNAME --saved-model /path/to/saved/model/pth --dataset DATASET --data-path /path/to/saved/datasets/ --batch-size 32 --device cpu --csv-file /path/to/wisdom/weights/csv --idc-test-all --attr lrp --top-m-neurons 10

# With WISDOM-based pertubation
./run_rq.sh --rq 3
```

### RQ 4: Correlation

Can the approach reveal the test suite’s diversity (or impartiality)?

Metrics:
- Measure the impartiality of the test suite
- Sample 100, 500 and 1000 test cases from the test set, maintaining the same ratio across classes ($U_{t1}$, $U_{t2}$, $U_{t3}$)
- Generate same size of the test cases with adversarial attacks method CW (same class) ($U_{b1}$, $U_{b2}$, $U_{b3}$)
- Get both Pielou’s evenness score (i.e., output_impartiality) and Coverage score
- Calculate the proportion $p_i$​ of predictions for each class i.
- Compute the Shannon entropy $H$.
- Normalize the entropy by dividing by the maximum possible entropy $log(k)$, k is #class
- Output impartiality: $J = \frac{H}{log(k)}$, $J \in [0, 1]$
- Record Pearson correlation coefficient: $r = \frac{\sum_i(c_i - \bar{c})(p_i - \bar{p})}{\sqrt{(\sum_i(c_i - \bar{c})^2} \sqrt{\sum_i(p_i - \bar{p})^2}}$

How to run
```shell
# MODELNAME: [lenet, vgg16, resnet18]
# DATASET: [mnist, cifar10, imagenet]
# Pretrained relevant scores: ./saved_files/pre_csv/<MODELNAME>_<DATASET>.csv
python run_rq_4_demo.py --model MODELNAME --saved-model /path/to/saved/model/pth --dataset DATASET --data-path /path/to/saved/datasets/ --batch-size 32 --device cpu --csv-file /path/to/wisdom/weights/csv --idc-test-all --top-m-neurons 10

# With WISDOM-based pertubation
./run_rq.sh --rq 4
```

### RQ 5: Efficiency (overhead)

Record the time overhead on different models.


## Metrics reference

Other implmentation for the baseline should include:

- Neuron Coverage (NC) [1]
- K-Multisection Neuron Coverage (KMNC) [2]
- Neuron Boundary Coverage (NBC) [2]
- Strong Neuron Activation Coverage (SNAC) [2]
- Top-K Neuron Coverage (TKNC) [2]
- Top-K Neuron Patterns (TKNP) [2]
- Cluster-based Coverage (CC) [3]
- Likelihood Surprise Coverage (LSC) [4]
- Distance-ratio Surprise Coverage (DSC) [5]
- Mahalanobis Distance Surprise Coverage (MDSC) [5]

[1] DeepXplore: Automated whitebox testing of deep learning systems, SOSP 2017. <br>
[2] DeepGauge: Comprehensive and multi granularity testing criteria for gauging the robustness of deep learning systems, ASE 2018. <br>
[3] Tensorfuzz: Debugging neural networks with coverage-guided fuzzing, ICML 2019. <br>
[4] Guiding deep learning system testing using surprise adequacy, ICSE 2019. <br>
[5] Reducing dnn labelling cost using surprise adequacy: An industrial case study for autonomous driving, FSE Industry Track 2020.

Implementation repo: [NeuraL-Coverage](https://github.com/Yuanyuan-Yuan/NeuraL-Coverage/tree/main)

## Potential improvement and extensions

- [ ] A template-based optimization for all the attribution methods (acceleration)
- [ ] Attribution methods in LLMs and transformer-based models
- [ ] Vectorization for the all the attributions
- [ ] A better KMeans method (torch-based)

## Docker [TODO]

See `Docker` with the `Dockerfile`

```shell

# Run
docker run --gpus all -it --name deepimportance-container torch-deepimportance

docker commit deepimportance-container torch-deepimportance:v1
docker login
docker tag torch-deepimportance:v1 your_dockerhub_username/torch-deepimportance:v1
docker push your_dockerhub_username/torch-deepimportance:v1
docker pull your_dockerhub_username/torch-deepimportance:v1

## testing and debugging
docker exec -it deepimportance-container bash
```

# Torch4DeepImportance

A pytorch version for Deepimportance (test version). [TensorFlow version for DeepImportance](https://github.com/DeepImportance/deepimportance_code_release/tree/ga_modifications). 

For the paper, please refer [HERE](https://zenodo.org/records/3628024).

## How to run (Captum version)

The Captum version demo is tested and should be fine for further developments. The docker is still pending.

Use `conda` or `pyvenv` to build a virtual environment.

```shell
# requriements
$ pip -r install requirements.txt

# If you are using anaconda or miniconda virtual environment, do:
$ conda env create -f requirements_venv.yaml
```

Run with a script:

```shell
# End2end testing the deepimportance with LeNet-5 in MNIST dataset, with fixed cluster number (=2)
$ ./script.sh test

# Run with Use Case N: `./script.sh caseN`, e.g.:
$ ./script.sh case1

# To plot the common neurons in different layer per class
$ cd logs
# python plot_images.py --plot-all --log-file <log_file_name.log>
$ python plot_images.py --plot-all
```

## Routes

- Activation values for important neuros (v_1, v_2, ...)
- Clustering with Silhoutte score (or with the customized `n_cluster`)
- Combination of clusters from important neuros
- Testset comes in (x_1, y_1)
- Check coverage (See combinations covered by the test set, e.g., 4/6, 1/6 ,....), A.k.a. IDC

## TODO

- [x] [**YOLO**] Implement the [YOLOv8](https://github.com/jahongir7174/YOLOv8-pt/tree/master) (or [YOLOv5](https://github.com/mihir135/yolov5)) in pytorch, with COCO dataset
  - [ ] Architecture analysation block by block, YOLOv5s
- [x] [**IDC**] Selectors for different attributors (using the accuracy as the guide)
    - [x] Rankings, plots
    - [x] Transfer learning based training
    - [x] Voting
- [ ] [**Lib**] Refine the codes (Now: v0.1 -> v0.2)
  - [ ] [**Lib**] GitHub CI for Docker building
  - [ ] [**Lib**] GitHub CI for Pytest Running with a small demo (MNIST)
  - [ ] [**Lib**] Extract the small verison libs
- [ ] [**Lib**] pip package and docker conatiner
- [ ] [**IDC**] Customized rules for `torch.nn.Sequencial` in LRP
- [ ] [**IDC**] Add more dataset and models
  - [ ] VGG16 + ImageNet
  - [ ] ConvNext + ImageNet
  - [ ] mobilenet + ImageNet
- [ ] [**IDC**] SOTA methods to compare
- [ ] [**IDC**] Per feature selector (transfer learning style)
  - [ ] A table for the variances and per input predition scores
- [ ] [**IDC**] runtime version + attention
- [ ] [**YOLO**] YOLO v11


## Directory information

- `Docker`: Dockerfile
- `data`: COCO dataset info
- `examples`: Small examples (stand-alone) to test captum
- `images`: All the saved images, including the example images and the heatmap saved by the running demos
- `logs`: The saved the log files
- `models_info`: Pre-trained model files (*.pt) and also the model architecture visualisations
  - `train_from_scratch`: This is a standalone file that allows you to train all the CV models from scratch with CIFAR10 dataset
- `saved_files`: Saved JSON files for the model importances
- `src`: The source files, including the idc implementations and the attribution methods (torch/captum)
- `run.py`: Main entrance for the testing program
- `test_base.sh`: A baseline testing (DeepImportance based with LRP)

## Parameters
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
      --test-image plane \
      # --idc-test-all \
      --num-samples 1000 \
      --attr lrp \
      --layer-index 1 \
      # --layer-by-layer \
      # --end2end \
      # --all-class \
      --log-path './logs/TestLog' \
      --logging

# Clustering, choose one
--use-silhouette # dynamic choose the cluster number
--n-cluster N # fixed cluster number

# Sampling test images
--num-sample N # Randomly pick N samples
--idc-test-all # Choose all the image in one batch

# Choose one or none:
--layer-by-layer # Looping all the layer on a specific class
--end2end # End2End test
--all-class # Test all the class with a given layer

# Logging
--log-path  '<path>/<name>' # log file path and name: e.g., './logs/TestLog'
--logging # Save the log file

# An LeNet Example - CIFAR10 with layer index: 1
python run.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --importance-file './saved_files/lenet_impotant.json' --use-silhouette --device cpu --n-cluster 2 --top-m-neurons 5 --test-image plane --idc-test-all --num-samples 0  --attr lrp --layer-index 1 --log-path './logs/TestLog' --logging

# An LeNet end2end example - CIFAR10
python run.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --importance-file './saved_files/lenet_impotant.json' --device cpu --n-cluster 2 --top-m-neurons 10 --test-image plane --end2end --idc-test-all --num-samples 0  --attr lrp --log-path './logs/TestLog' --logging

# A VGG16 example - ImageNet
python run.py --model vgg16 --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --importance-file './saved_files/vgg16_impotant.json' --device cpu --n-cluster 2 --top-m-neurons 5 --test-image plane --idc-test-all --num-samples 0  --attr lrp --layer-index 1 --log-path './logs/TestLog' --logging

# UC-2: voting data for specific model
## Prepare data - VGG16
python prepare_data.py \
        --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' \
        --dataset cifar10 \
        --batch-size 2 \
        --layer-index 5 \
        --model vgg16 \
        --top-m-neurons 10 \
        --n-clusters 2 \
        --end2end \
        --csv-file ordered \
        --log-path './logs/PrepareDataLog' \
        --logging

## Prepare the data, CIFAR10 and LeNet, in orderd
python prepare_data.py \
        --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' \
        --dataset cifar10 \
        --batch-size 2 \
        --end2end \
        --model lenet \
        --top-m-neurons 10 \
        --n-clusters 2 \
        --csv-file ordered \
        --log-path './logs/PrepareDataLog' \
        --logging

# Uc - 3
# Test with the trained data
python run_pre.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --importance-file './saved_files/lenet_impotant.json' --device cpu --n-cluster 2 --top-m-neurons 10 --test-image plane --end2end --num-samples 0  --attr lrp --layer-index 1
```

## Metrics reference

Other implmentation for the baseline should include:

Neuron Coverage (NC) [1]

K-Multisection Neuron Coverage (KMNC) [2]

Neuron Boundary Coverage (NBC) [2]

Strong Neuron Activation Coverage (SNAC) [2]

Top-K Neuron Coverage (TKNC) [2]

Top-K Neuron Patterns (TKNP) [2]

Cluster-based Coverage (CC) [3]

Likelihood Surprise Coverage (LSC) [4]

Distance-ratio Surprise Coverage (DSC) [5]

Mahalanobis Distance Surprise Coverage (MDSC) [5]

[1] DeepXplore: Automated whitebox testing of deep learning systems, SOSP 2017.

[2] DeepGauge: Comprehensive and multi granularity testing criteria for gauging the robustness of deep learning systems, ASE 2018.

[3] Tensorfuzz: Debugging neural networks with coverage-guided fuzzing, ICML 2019.

[4] Guiding deep learning system testing using surprise adequacy, ICSE 2019.

[5] Reducing dnn labelling cost using surprise adequacy: An industrial case study for autonomous driving, FSE Industry Track 2020.

Implementation repo: [NeuraL-Coverage](https://github.com/Yuanyuan-Yuan/NeuraL-Coverage/tree/main)

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

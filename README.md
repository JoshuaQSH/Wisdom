# Torch4DeepImportance

A pytorch version for Deepimportance (test version). [TensorFlow version for DeepImportance](https://github.com/DeepImportance/deepimportance_code_release/tree/ga_modifications). 

For the paper, please refer [HERE](https://zenodo.org/records/3628024).

A Captum lib added.

## How to run (Captum version)

The Captum version demo is tested and should be fine for further developments. After installing the prerequest libs with `pip`, please go to `captum_demo` for more details.

Use `conda` or `pyvenv` to build a virtual environment.

```shell
# requriements
$ pip -r install requirements.txt
```

Run with a script:

```shell
# The default test will be using a customized LeNet-5 with CIFAR10 dataset
# Usage: ./script.sh <chosen-layer>, <chosen-layer> could be ['fc1', 'fc2', 'conv1', 'conv2']. For the LeNet-5+CIFAR10, we do:
$ ./script.sh lenetfc1

# To plot the common neurons in different layer per class
$ cd logs
# python plot_images.py --plot-all --log-file <log_file_name.log>
$ python plot_images.py --plot-all

# Selector predict
$ python selector_pred_v1.py --dataset cifar10 --batch-size 2 --layer-index 2 --model lenet --top-m-neurons 10 --all-attr
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

# An LeNet Example - CIFAR10
python run.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --importance-file './saved_files/lenet_impotant.json' --use-silhouette --device cpu --n-cluster 2 --top-m-neurons 5 --test-image plane --idc-test-all --num-samples 0  --attr lrp --layer-index 1 --log-path './logs/TestLog' --logging

# An LeNet Example End2End - MNIST
python run.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_MNIST_whole.pth' --dataset mnist --data-path '/data/shenghao/dataset/' --importance-file './saved_files/lenet_impotant.json' --device cpu --n-cluster 2 --top-m-neurons 5 --test-image 1 --end2end --idc-test-all --num-samples 0  --attr lrp --log-path './logs/TestLogMNIST' --logging

# An LeNet end2end example, with logging on
python run.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --importance-file './saved_files/lenet_impotant.json' --device cpu --n-cluster 2 --top-m-neurons 10 --test-image plane --end2end --idc-test-all --num-samples 0  --attr lrp --log-path './logs/TestLog' --logging

# A VGG16 example - cifar10
# attributions = ['lc', 'la', 'ii', 'ldl', 'lgs', 'lig', 'lfa', 'lrp']
python run.py --model vgg16 --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --importance-file './saved_files/vgg16_impotant.json' --device cpu --n-cluster 2 --top-m-neurons 50 --test-image plane --idc-test-all --num-samples 0  --attr lrp --end2end --layer-index 1 --log-path './logs/TestLog' --logging

# A VGG16 example - ImageNet
python run.py --model vgg16 --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --importance-file './saved_files/vgg16_impotant.json' --device cpu --n-cluster 2 --top-m-neurons 5 --test-image plane --idc-test-all --num-samples 0  --attr lrp --layer-index 1 --log-path './logs/TestLog' --logging

# mobilenetv2_CIFAR10_whole
python run.py --model mobilenetv2 --saved-model '/torch-deepimportance/models_info/saved_models/mobilenetv2_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --importance-file './saved_files/densenet_impotant.json' --use-silhouette --device cpu --n-cluster 2 --top-m-neurons 5 --test-image plane --idc-test-all --num-samples 0  --attr lrp --layer-index 3


# UC-2 Selector
## Prepare selector data
python prepare_selector_data.py \
        --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' \
        --dataset cifar10 \
        --batch-size 256 \
        --layer-index 5 \
        --model vgg16 \
        --top-m-neurons 10 \
        --n-clusters 2 \
        --log-path './logs/PrepareDataLog' \
        --logging

## Prepare the data, CIFAR10 and LeNet
python prepare_selector_data.py \
        --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' \
        --dataset cifar10 \
        --batch-size 100 \
        --end2end \
        --model lenet \
        --top-m-neurons 10 \
        --n-clusters 2 \
        --log-path './logs/PrepareDataLog' \
        --logging

## Prediction - lenet
python selector_pred_v1.py --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --batch-size 256 --layer-index 2 --model lenet --top-m-neurons 10 --all-class

## Prediction
python selector_pred_v1.py --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --batch-size 256 --layer-index 2 --model vgg16 --top-m-neurons 10 --all-class


# Uc - 3
python run_pre.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --importance-file './saved_files/lenet_impotant.json' --device cpu --n-cluster 2 --top-m-neurons 10 --test-image plane --end2end --num-samples 0  --attr lrp --layer-index 1
```

## Docker

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

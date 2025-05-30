# Testing the AI4Work Pilot Partner's dataset and models

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
```

## Routes

- Activation values for important neuros (v_1, v_2, ...)
- Clustering with Silhoutte score (or with the customized `n_cluster`)
- Combination of clusters from important neuros
- Testset comes in (x_1, y_1)
- Check coverage (See combinations covered by the test set, e.g., 4/6, 1/6 ,....), A.k.a. IDC

## Parameters and single files running example
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

```

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

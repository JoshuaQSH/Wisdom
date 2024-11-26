# Torch4DeepImportance

A pytorch version for Deepimportance (test version). [TensorFlow version for DeepImportance](https://github.com/DeepImportance/deepimportance_code_release/tree/ga_modifications). [Paper](https://zenodo.org/records/3628024).

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
$ cd captum_demo

# The default test will be using a customized LeNet-5 with CIFAR10 dataset
# Usage: ./script.sh <chosen-layer>, <chosen-layer> could be ['fc1', 'fc2', 'conv1', 'conv2'], example:
$ ./script.sh fc1
```

## Routes

- Activation values for important neuros (v_1, v_2, ...)
- Clustering with Silhoutte score (or with the customized `n_cluster`)
- Combination of clusters from important neuros
- Testset comes in (x_1, y_1)
- Check coverage (See combinations covered by the test set, e.g., 4/6, 1/6 ,....), A.k.a. IDC

## TODO

- [x] [**YOLO**] Implement the [YOLOv8](https://github.com/jahongir7174/YOLOv8-pt/tree/master) (or [YOLOv5](https://github.com/mihir135/yolov5)) in pytorch, with COCO dataset
  - [ ] Layers by layers
  - [ ] Architecture analysation block by block, YOLOv5s
- [ ] [**YOLO**] YOLO v11


## Directory information

- `images`: All the saved images, including the example images and the heatmap saved by the running demos
- `models_info`: Pre-trained model files (*.pt) and also the model architecture visualisations
- `saved_files`: Saved JSON files for the model importances
- `logs`: Saved the log files
- `examples`: Small example testing captum

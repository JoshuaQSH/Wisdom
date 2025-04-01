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

# Torch4DeepImportance

A pytorch version for Deepimportance (test version). [TensorFlow version for DeepImportance](https://github.com/DeepImportance/deepimportance_code_release/tree/ga_modifications). [Paper](https://zenodo.org/records/3628024).

## How to run

Use `conda` or `pyvenv` to build a virtual environment.

```shell
# requriements
$ pip -r install requirements.txt
$ python run.py
```

## Routes

- Activation values for important neuros (v_1, v_2, ...)
- Clustering with Silhoutte score
- Combination of clusters from important neuros
- Testset comes in (x_1, y_1)
- Check coverage (See combinations covered by the test set, e.g., 4/6, 1/6 ,....)

IDC and converage rate
```shell
(x_1, y_1) -> cluster X
min(L2(N,V))
```

# WISDOM Package

A package for WISDOM, a white-box and semantically informed coverage testing tool for Deep Neural Networks with xAI methods.

## Prerequest

Use `conda` or `pyvenv` to build a virtual environment.

```shell
# requriements
$ pip -r install requirements.txt

# If you are using anaconda or miniconda virtual environment, do:
$ conda env create -f requirements_venv.yaml
```

## Install wheel package

```shell
git clone https://github.com/JoshuaQSH/Wisdom.git
cd Wisdom/dist
pip install wisdom-0.1.0-py3-none-any.whl
```

## Uninstall wheel package

```shell
pip uninstall wisdom
```

## Install changes in the library

In case of changes in the source code of the library, then the wheel package needs to be recreated. Please follow the steps below for receating the wheel package.

```shell
cd Wisdom/
python setup.py bdist_wheel
```

Once this is done, a Wisdom/dist directory will be created. Then follow the instractions in the installation section.


## Directories and files

- `attribution`: Main attribution methods definition and a customized template
- `benchmark`: Benchmarking results [WiP]
- `build`: Lib build file
- `clustering`: Clustering methods and WISDOM assignments
- `core`: Core files of WISDOM
- `coverage_methods`: Some basline coverage-based methods
- `dist`: Wheels
- `Docker`: Docker file [WiP]
- `examples`: Example usage [WiP]
- `logs`: Logging file
- `pruning`: Pruning methods (mask and weight pruning)
- `saved_files`: Saved files, including the neuron importance scores in CSV
- `unittest`: Unit testing and sanity check [WiP]
- `utils`: Helper and common files
- `config.py`
- `requirements_venv.yaml`
- `requirements.txt`
- `setup.py`
- `run_wisdom.py`

## Run Wisdom

```shell
python3 run_wisdom.py \
      --model <str:model_name> \
      --saved-model <str:path_to_file> \
      --dataset <str:dataset_name> \
      --data-path <str:path_to_file> \
      --device <cpu/cuda> \
      --n-cluster <int> \
      --top-m-neurons <int> \
      --end2end \
      --num-samples <int> \
      --csv-file <str:path_to_file> \
      --idc-test-all 
```

## Docker

See `Docker` with the `Dockerfile`
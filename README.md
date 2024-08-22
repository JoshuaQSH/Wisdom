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

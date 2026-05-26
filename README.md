# WISDOM: Semantically Informed Coverage Testing for Deep Neural Networks

For the paper, please refer [HERE](https://zenodo.org/records/3628024).

## Prerequest

We recommend using `uv` to create the virtual environment (`.venv`)  and control the Python libraries:

```shell
cd Wisdom
uv sync
source .venv/bin/activate
```

## `run_cases/smoke.py` flag reference

| Flag | Values / type | Default | Usage |
|---|---|---:|---|
| `--mode` | `score`, `rq2`, `dnn-bo`, `yolo-bo` | required | Select which smoke workflow to run. |
| `--weights` | path | `weights/yolo11n.pt` | Detection checkpoint for `score`, `rq2`, and `yolo-bo`. |
| `--img-dir` | path | `standalone/data/coco/images/val2017` | Detection image directory for `score`, `rq2`, and `yolo-bo`. |
| `--device` | device string | `cuda:0` | Execution device. |
| `--num-images` | integer | `4` | Number of YOLO images to use in detection smoke modes. |
| `--batch-size` | integer | `1` | Batch size for the selected smoke run. |
| `--imgsz` | integer | `320` | YOLO image size for detection smoke modes. |
| `--out` | path | required | Output path or prefix. Meaning depends on `--mode`. |
| `--csv-file` | path | empty string | Existing neuron-score CSV. Used by `rq2` and `yolo-bo`; can also point at a DNN score CSV for BO smoke. |
| `--model-path` | path | repo DNN smoke model | Classification checkpoint used by `dnn-bo`. |
| `--dataset` | dataset name | `mnist` | Classification dataset for `dnn-bo`. |
| `--data-path` | path | repo default data root | Classification dataset root for `dnn-bo`. |
| `--top-m` | integer | `4` | Number of monitored neurons for smoke BO/score flows. |
| `--build-samples` | integer | `64` | Build subset size for BO smoke modes. |
| `--eval-samples` | integer | `32` | Evaluation subset size for BO smoke modes. |
| `--bo-backend` | `auto`, `sklearn`, `botorch` | `auto` | BO backend for `dnn-bo` and `yolo-bo`. |
| `--bo-init` | integer | `2` | Initial BO evaluations for `dnn-bo` and `yolo-bo`. |
| `--bo-iter` | integer | `2` | BO optimization iterations for `dnn-bo` and `yolo-bo`. |

## `run_wisdom.py` flag reference

### Core mode and input flags

| Flag | Values / type | Default | Usage |
|---|---|---:|---|
| `--impl` | `wisdom`, `idc` | `wisdom` | Choose the packaged implementation. |
| `--mode` | `wisdom`, `idc` | hidden alias | Backward-compatible alias for `--impl`. |
| `--task` | `auto`, `classification`, `detection` | `auto` | Force the task type or let the script infer it from the inputs. |
| `--model-path` | path | required | Classification `.pth` checkpoint or detection `.pt` weights. |
| `--dataset` | `mnist`, `cifar10`, `cifar100`, `imagenet` | none | Classification dataset name. |
| `--data-path` | path | none | Classification dataset root. |
| `--img-dir` | path | none | Detection image directory. Required for detection runs. |
| `--csv-file` | path | none | Reuse an existing neuron-score CSV instead of pretraining one. |
| `--pretrain` | flag | off | Generate a neuron-score CSV before the coverage run. |
| `--methods` | one or more attribution keys | none | Attribution methods used during pretraining. In `wisdom` mode this can be a consensus set; in `idc` mode it must resolve to a single method. |
| `--attribution-method` | one attribution key | none | Convenience single-method selector, especially for `idc`. |
| `--voting-mode` | `fine-grained`, `coarse` | `fine-grained` | WISDOM consensus voting mode for pretraining. |

### Runtime, sampling, and selection flags

| Flag | Values / type | Default | Usage |
|---|---|---:|---|
| `--device` | device string | auto | Execution device. |
| `--batch-size` | integer | `16` | Batch size for build, eval, and optional pretraining. |
| `--build-samples` | integer | full dataset | Optional build-data subset size. Omit for the full build dataset. |
| `--eval-samples` | integer | full dataset | Optional evaluation subset size. Omit for the full evaluation dataset. |
| `--top-m-neurons` | integer | `10` | Number of monitored neurons used by the run. |
| `--per-group [N]` | integer, optional value | `3` when flag is present without a value | Average coverage across `N` contiguous layer groups. |
| `--per-layer` | flag | off | Average coverage across layers, selecting top-k neurons separately per layer. |
| `--seed` | integer | `42` | Random seed for clustering and subset workflows. |
| `--imgsz` | integer | `640` | Detection image size. |
| `--noise-std` | float | `0.3` | Gaussian noise standard deviation used by `--plot-pixels` perturbation diagnostics. |
| `--pixel-frac` | float | `0.02` | Fraction of important pixels perturbed for `--plot-pixels` diagnostics. |

### Clustering and BO flags

| Flag | Values / type | Default | Usage |
|---|---|---:|---|
| `--cluster-method` | clustering method name | `KMeans` | Final clustering backend when BO is not used. |
| `--n-clusters` | integer | `2` | Final cluster count when BO is not used. |
| `--cache-path` | path | none | Optional cluster-cache directory. |
| `--bo` | flag | off | Tune clustering hyperparameters before the final run. |
| `--bo-backend` | `auto`, `sklearn`, `botorch` | `auto` | BO backend. |
| `--bo-init` | integer | `3` | Initial BO evaluations. |
| `--bo-iter` | integer | `3` | BO optimization iterations. |
| `--bo-cluster-methods` | comma-separated method list | `KMeans,MiniBatchKMeans,Birch` | Discrete BO search space for clustering methods. |
| `--bo-n-clusters` | comma-separated integer list | `2,3,4` | Discrete BO search space for cluster counts. |
| `--corr-points` | integer | `5` | Number of suite sizes used for coverage/F1 correlation. |

### Testing mode, logging, and outputs

| Flag | Values / type | Default | Usage |
|---|---|---:|---|
| `--end2end` | flag | on | End-to-end testing for the whole model. |
| `--all-class` | flag | on | Evaluate the full evaluation set across all classes. |
| `--class-iters` | flag | off | Iterate by class and compute coverage/F1 correlation across classes. |
| `--combo-log` | flag | off | Write per-sample activated combinations plus counts. |
| `--plot-neurons` | flag | off | Save the top-neuron score plot. |
| `--plot-pixels` | flag | off | Save pixel-importance heatmaps and perturbed-image diffs. |
| `--out-dir` | path | `results/run_wisdom` | Output directory. |
| `--run-name` | string | `run_wisdom` | Prefix used for output artifacts. |


### Classification example

```bash
python run_wisdom.py \
  --impl wisdom \
  --task classification \
  --model-path /path/to/lenet_MNIST_whole.pth \
  --dataset mnist \
  --data-path /scratch/staff/lrr550/datasets \
  --csv-file /path/to/lenet_mnist.csv \
  --device cpu \
  --batch-size 8 \
  --build-samples 16 \
  --eval-samples 8 \
  --top-m-neurons 4 \
  --bo \
  --bo-init 1 \
  --bo-iter 1 \
  --voting-mode fine-grained \
  --all-class \
  --bo-cluster-methods KMeans,Birch \
  --bo-n-clusters 2,3 \
  --out-dir results/run_wisdom_doc \
  --run-name mnist_doc_smoke
```

## Routes

- Activation values for important neuros (v_1, v_2, ...)
- Combining and voting for the best attribution methods to come up with a better important neuron set
- Clustering with Silhoutte score (or with the customized `n_cluster`)
- Combination of clusters from important neuros
- Testset comes in (x_1, y_1)
- Check coverage (See combinations covered by the test set, e.g., 4/6, 1/6 ,....), A.k.a. IDC

## TODO

- [x] Per-group / per-layer testing 
      - [x] Support YOLOv11
      - [x] Support YOLOv5
- [ ] Circuits-based path testing
      - [ ] Support BERT
      - [ ] Support LLMs (QWen, LLaMa, DeepSeek, etc.)
- [ ] Docker building

## Research Questions

Note: While running the scripts, ensure to change the `$DATA_PATH` to your own dataset path.

### RQ 1: Critical (or important) neurons

Metrics:
- Top n in (6, 8, 10, 15, 20) neurons
- Accuracy drop based on the neurons pruning


```bash
python run_cases/run_rq1.py --weights /path/to/yolo11n.pt --img-dir /path/to/coco/images/val2017 --csv-file /path/to/wisdom_yolo11n_scores.csv --out-prefix results/rq1/yolo11n
```

Outputs:

- `results/rq1/yolo11n_relevance.csv`
- `results/rq1/yolo11n_acc_drop.csv`
- `results/rq1/yolo11n_<model-tag>_acc_drop.pdf`

Flag reference:

| Flag | Values / type | Default | Usage |
|---|---|---:|---|
| `--weights` | path | `weights/yolo11n.pt` | Detection checkpoint to evaluate. |
| `--img-dir` | path | `standalone/data/coco/images/val2017` | Detection image directory. |
| `--csv-file` | path | `neuron_eval_out/wisdom_yolo11n_scores.csv` | WISDOM neuron-score CSV. |
| `--out-prefix` | path prefix | `results/rq1_yolo11n` | Output prefix used for CSVs and plots. |
| `--device` | device string | `cuda:0` | Execution device. |
| `--num-images` | integer | `50` | Number of labeled images to evaluate. |
| `--batch-size` | integer | `2` | Batch size for inference and pruning checks. |
| `--imgsz` | integer | `320` | Detection image size. |
| `--methods` | one or more attribution keys | empty | Single-method baselines to compare against WISDOM. Requires matching `--single-csv` entries. |
| `--single-csv` | `method=path` entries | empty | Pretrained single-method CSVs paired with `--methods`. |
| `--num-runs` | integer | `1` | Number of repeated RQ1 runs. |
| `--seed` | integer | `42` | Base seed for repeatable subset sampling. |
| `--sample-mode` | `auto`, `first`, `random` | `auto` | Subset selection strategy. |
| `--no-random` | flag | off | Disable the shared random pruning baseline. |
| `--eval-map` | flag | off | Also compute `mAP50` and `mAP50-95` drops with Ultralytics validation. |


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


```bash
python run_cases/run_rq2.py \
  --weights weights/yolo11n.pt \
  --img-dir /path/to/coco/images/val2017 \
  --csv-file path/to/wisdom_yolo11n_scores.csv \
  --out-csv results/rq2/yolo11n_coverage.csv
```

Outputs:

- `results/rq2/yolo11n_coverage.csv`
- a companion log under a sibling `logs/` directory

Flag reference:

| Flag | Values / type | Default | Usage |
|---|---|---:|---|
| `--weights` | path | `weights/yolo11n.pt` | Detection checkpoint to evaluate. |
| `--img-dir` | path | `standalone/data/coco/images/val2017` | Detection image directory. |
| `--csv-file` | path | `neuron_eval_out/wisdom_yolo11n_scores.csv` | WISDOM neuron-score CSV. |
| `--out-csv` | path | `results/rq2_yolo11n_coverage.csv` | Output CSV path. |
| `--device` | device string | `cuda:0` | Execution device. |
| `--num-images` | integer | `20` | Number of images evaluated. |
| `--batch-size` | integer | `2` | Batch size. |
| `--imgsz` | integer | `320` | Detection image size. |
| `--n-iterations` | integer | `3` | Number of repeated perturbation iterations. |

### RQ 3: Effectiveness (or sensitivity)

Metrics:
- Sample 100, 1000, 3000 correct inputs in testset. 
- Replace some of the inputs (1%, 5%, 10%) with adversarial examples (crafted using PGD, FGSM and CW).
- Record the Normalization(delta(Coverage)) (expect stable improvements)

$NCov(s) = \frac{Cov(s) - Cov(s_0)}{max(\Delta) - min(\Delta)}$ <br>
$\Delta = \{Cov(s) - Cov(s_0) | s \in S\}$

```bash
python run_cases/run_rq3.py \
  --weights /path/to/yolo11n.pt \
  --img-dir /path/to/coco/images/val2017 \
  --csv-file /path/to/wisdom_yolo11n_scores.csv \
  --out-csv results/rq3/yolo11n_effectiveness.csv
```

Outputs:

- `results/rq3/yolo11n_effectiveness.csv`
- `results/rq3/yolo11n_effectiveness_plot.pdf`

Flag reference:

| Flag | Values / type | Default | Usage |
|---|---|---:|---|
| `--weights` | path | `weights/yolo11n.pt` | Detection checkpoint to evaluate. |
| `--img-dir` | path | `standalone/data/coco/images/val2017` | Detection image directory. |
| `--csv-file` | path | `neuron_eval_out/wisdom_yolo11n_scores.csv` | WISDOM neuron-score CSV. |
| `--out-csv` | path | `results/rq3_yolo11n_effectiveness.csv` | Output CSV path. |
| `--device` | device string | `cuda:0` | Execution device. |
| `--num-images` | integer | `20` | Number of images evaluated. |
| `--batch-size` | integer | `2` | Batch size. |
| `--imgsz` | integer | `320` | Detection image size. |

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

```bash
python run_cases/run_rq4.py \
  --weights /path/to/yolo11n.pt \
  --img-dir /path/to/coco/images/val2017 \
  --csv-file /path/to/wisdom_yolo11n_scores.csv \
  --out-csv results/rq4/yolo11n_correlation.csv
```

Outputs:

- `results/rq4/yolo11n_correlation.csv`
- a summary log at a sibling `logs/rq4_results.log`

Flag reference:

| Flag | Values / type | Default | Usage |
|---|---|---:|---|
| `--weights` | path | `weights/yolo11n.pt` | Detection checkpoint to evaluate. |
| `--img-dir` | path | `standalone/data/coco/images/val2017` | Detection image directory. |
| `--csv-file` | path | `neuron_eval_out/wisdom_yolo11n_scores.csv` | WISDOM neuron-score CSV. |
| `--out-csv` | path | `results/rq4_yolo11n_correlation.csv` | Output CSV path. |
| `--device` | device string | `cuda:0` | Execution device. |
| `--num-images` | integer | `30` | Number of images evaluated per trial. |
| `--imgsz` | integer | `320` | Detection image size. |
| `--num-trials` | integer | `3` | Number of repeated correlation trials. |

### RQ 5: Efficiency (overhead)

Record the time overhead on different models.

```bash
python run_cases/run_rq5.py \
  --weights /path/to/yolo11n.pt \
  --img-dir /path/to/coco/images/val2017 \
  --csv-file /path/to/wisdom_yolo11n_scores.csv \
  --out-csv results/rq5/yolo11n_efficiency.csv
```

Outputs:

- `results/rq5/yolo11n_efficiency.csv`

Flag reference:

| Flag | Values / type | Default | Usage |
|---|---|---:|---|
| `--weights` | path | `weights/yolo11n.pt` | Detection checkpoint to evaluate. |
| `--img-dir` | path | `standalone/data/coco/images/val2017` | Detection image directory. |
| `--csv-file` | path | `neuron_eval_out/wisdom_yolo11n_scores.csv` | WISDOM neuron-score CSV. |
| `--out-csv` | path | `results/rq5_yolo11n_efficiency.csv` | Output CSV path. |
| `--device` | device string | `cuda:0` | Execution device. |
| `--num-images` | integer | `4` | Number of images used for timing. |
| `--batch-size` | integer | `2` | Batch size. |
| `--imgsz` | integer | `320` | Detection image size. |
| `--wisdom-only` | flag | off | Skip single-method and random baselines and time only the WISDOM path. |
| `--wisdom-methods` | comma-separated attribution keys | `lgxa,lig` | Consensus method list used for WISDOM timing. |

### Notes for the Adversarial (attack) methods

Our adversarial examples are generated using [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch). Example of usages:

```python
import torchattacks

atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)

# Save
atk.save(data_loader, save_path="./AE.pt", verbose=True)
  
# Load
adv_loader = atk.load(load_path="./AE.pt")
```

### Fuzzing Test

How to run

```shell
Usage:
# Random baseline fuzzing for CIFAR10 dataset using VGG16 model.
python ./fuzz_guide/run_fuzz.py \
      --dataset CIFAR10 \
      --model vgg16 \
      --saved-model /torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth \
      --output-dir ./fuzz_guide/fuzz_outputs/ \
      --log-dir ./logs \
      --seed 42 \
      --device 'cuda:0'

# Coverage method guided fuzzing is not used in this script, but can be enabled by setting the --guided flag.
python ./fuzz_guide/run_fuzz.py \
      --dataset CIFAR10 \
      --model vgg16 \
      --saved-model /torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth \
      --criterion NC \
      --output-dir ./fuzz_guide/fuzz_outputs/ \
      --log-dir ./logs \
      --seed 42 \
      --device 'cuda:0' \
      --guided
```

During fuzzing both scripts dump images every `save_every` (=100) epochs:

```shell
<output_dir>/<exp_name>/image/
    ├─ 000_new.jpg   #  freshly accepted mutations
    ├─ 000_old.jpg   #  their parents
    ├─ 000_ae.jpg    #  adversarial subset (if any)
    ├─ 100_new.jpg
    └─ ...
```

- Triggered Faults (Not Valid Count): Valid if 1) the #changed-pixels is less than $\alpha$ * #pixels or 2) the maximum of changed pixel value is less than $\beta$ * 255.
- Naturalness of images: Inception Score (IS) and Frechet Inception Distance (FID)



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
- DeepImportance [6] (**Main reference**) 

[1] DeepXplore: Automated whitebox testing of deep learning systems, SOSP 2017. <br>
[2] DeepGauge: Comprehensive and multi granularity testing criteria for gauging the robustness of deep learning systems, ASE 2018. <br>
[3] Tensorfuzz: Debugging neural networks with coverage-guided fuzzing, ICML 2019. <br>
[4] Guiding deep learning system testing using surprise adequacy, ICSE 2019. <br>
[5] Reducing dnn labelling cost using surprise adequacy: An industrial case study for autonomous driving, FSE Industry Track 2020. <br>
[6] Importance-driven deep learning system testing. ICSE 2020.

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
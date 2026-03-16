# UPDATES – YOLOv11 Integration

This document describes the changes made to integrate YOLOv11 into the WISDOM framework.

## File Organization

All files follow a clean directory structure:

| Directory | Purpose |
|-----------|---------|
| `weights/` | Pretrained model weights (e.g. `yolo11n.pt`) |
| `neuron_eval_out/` | Neuron importance score CSVs from WISDOM |
| `results/` | Output CSVs and PDF plots from RQ experiments |
| `logs/` | Log files |

## New Files

| File | Description |
|------|-------------|
| `yolo_test.py` | Verifies YOLOv11 model loading and inference on COCO images |
| `wisdom_yolo_train.py` | Consensus-based neuron importance scoring for YOLOv11 |
| `run_rq1.py` | RQ1: Critical neuron pruning with confidence/IoU/accuracy drop |
| `run_rq2.py` | RQ2: Diversity – perturb important vs random pixels, compare coverage |
| `run_rq3.py` | RQ3: Effectiveness – adversarial attack detection via coverage change |
| `run_rq4.py` | RQ4: Correlation between WISDOM coverage and Pielou's evenness |
| `run_rq5.py` | RQ5: Runtime and memory overhead measurement |
| `wisdom/utils/yolo_wrapper.py` | Wraps YOLO detection model for Captum attribution |
| `weights/yolo11n.pt` | YOLOv11-nano pretrained weights |
| `tests/test_*.py` | Unit tests for all scripts (11 tests total) |

## Modified Files

| File | Changes |
|------|---------|
| `wisdom/core/wisdom_train.py` | Added YOLO support: `is_yolo` config flag, `_eval_loss_yolo()`, `_compute_yolo_importance()` with `exclude_detect_head` parameter, `_gradient_importance()` with detection head exclusion, `_yolo_prune_eval()`. Updated `_voting_init()` and `_select_top_neurons_all()` with `excluded_prefixes`/`filter_prefixes` for prefix-based layer exclusion. `ConsensusWisdom.fit()` now excludes all `model.23.*` detection head layers from attribution and voting. |
| `wisdom/utils/visulization.py` | Added `viz_rq1_acc_drop()` function for grouped bar chart visualization of RQ1 results (confidence drop, IoU drop, classification accuracy drop). Uses project color palette. |
| `run_rq1.py` | Added IoU and classification accuracy metrics alongside confidence drop. Added `COCOLabeledDataset` for labeled evaluation. Added `eval_iou_and_accuracy()` with NMS-based prediction matching. Added automatic PDF plot generation via `viz_rq1_acc_drop()`. Detection head (`model.23.*`) excluded from pruning. |

## Detection Head Exclusion

To match the classification model behavior where the final classifier layer is excluded from pruning, **all 25 trainable layers in the YOLO detection head (`model.23.*`)** are excluded from:
- Attribution score computation (`_compute_yolo_importance`)
- Gradient-based importance fallback (`_gradient_importance`)
- Neuron selection (`_select_top_neurons_all` via `filter_prefixes`)
- Voting buffer initialization (`_voting_init` via `excluded_prefixes`)
- Neuron pruning in RQ1 (`prune_neurons` and `flatten_importance`)
- WISDOM CSV neuron selection (`wisdom_neurons`)

This ensures only backbone/neck layers (63 of 88 total Conv2d layers) contribute to the importance analysis.

## RQ Experiment Outputs

### RQ1 – Critical Neurons
Reports four metrics per pruning level: Confidence Drop, IoU Drop, Classification Accuracy Drop, Detection Recall Drop. Results → `results/rq1_*.csv` + PDF plot.

### RQ2 – Diversity
Perturbs 2% of pixels (important vs random), measures WISDOM neuron coverage change. Prints summary table to console. Results → `results/rq2_*.csv`, logs → `logs/rq2_results.log`.

### RQ3 – Adversarial Effectiveness
Generates FGSM/PGD adversarial examples at varying error rates, measures normalised coverage change. Saves line plot. Results → `results/rq3_*.csv` + PDF plot.

### RQ4 – Correlation
Computes Pearson correlation between Pielou's evenness (test diversity) and WISDOM coverage vs baseline neuron coverage. Prints summary table. Results → `results/rq4_*.csv`, logs → `logs/rq4_results.log`.

## Dependencies

- `ultralytics` – YOLOv11 model loading and inference
- `captum` – Attribution methods (LayerGradientXActivation, IntegratedGradients, GradientShap)
- `torchvision` – NMS for detection post-processing
- `pytest` – Unit testing

## Captum Compatibility

Not all Captum methods work with YOLOv11's architecture:

| Method | Status | Notes |
|--------|--------|-------|
| `lgxa` (GradientXActivation) | ✅ Works | Fast, recommended |
| `lig` (IntegratedGradients) | ✅ Works | Higher memory usage |
| `lgs` (GradientShap) | ✅ Works | Good accuracy |
| `la` (LayerActivation) | ✅ Works | No target needed |
| `lrp` (LayerLRP) | ❌ Fails | SiLU has no LRP rule |
| `ldl` (DeepLift) | ❌ Fails | MaxPool2d reuse issue |

For unsupported methods, a gradient-magnitude fallback is used automatically.

## Usage

All commands assume you are in the `Wisdom/` directory with the virtual environment activated:

```bash
cd Wisdom
source ../.venv/bin/activate
```

### 1. Generate Neuron Importance Scores (Train Set)

Uses the **COCO2017 train split** (118,287 images) for consensus-based importance scoring.

```bash
python wisdom_yolo_train.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/train2017 \
  --batch-size 2 --num-images 118287 --top-m 20 \
  --methods lgxa lig lgs \
  --out-csv neuron_eval_out/wisdom_yolo11n_scores.csv \
  --device cuda:0 --imgsz 320
```

> **Note:** With `--batch-size 2 --imgsz 320` this is safe from OOM on an 11 GB GPU.
> Estimated runtime: ~280 hours (59 K batches × ~17 s each). For a quick
> sanity check use `--num-images 200`.

### 2. Run Research Questions (Val/Test Set)

All RQ scripts evaluate on the **COCO2017 val split** (5,000 images).

```bash
# RQ1: Critical neuron pruning (IoU + classification accuracy + recall)
python run_rq1.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/val2017 \
  --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv \
  --num-images 5000 --batch-size 2 --imgsz 320 --device cuda:0

# RQ2: Diversity – important vs random pixel perturbation
python run_rq2.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/val2017 \
  --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv \
  --num-images 5000 --batch-size 2 --imgsz 320 --device cuda:0

# RQ3: Adversarial effectiveness (FGSM / PGD)
python run_rq3.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/val2017 \
  --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv \
  --num-images 5000 --batch-size 4 --imgsz 320 --device cuda:0

# RQ4: Correlation (Pielou's evenness vs WISDOM coverage)
python run_rq4.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/val2017 \
  --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv \
  --num-images 5000 --imgsz 320 --device cuda:0

# RQ5: Runtime and memory overhead
python run_rq5.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/val2017 \
  --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv \
  --num-images 50 --batch-size 2 --imgsz 320 --device cuda:0
```

> **OOM avoidance:** All scripts use batched processing internally.
> `--batch-size 2 --imgsz 320` is safe on an 11 GB GPU for all scripts.
> RQ3 can use `--batch-size 4` because adversarial generation is batched
> separately (4 images per attack batch).

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

## Checkpoint / Resume Mechanism

Long-running WISDOM training (~280 h for full COCO2017) saves periodic
checkpoints so you can safely interrupt and resume later.

### How It Works
- Every `--checkpoint-every` batches (default 50), a `.pt` file is saved
  containing the current `layer_scores` dict and the batch index.
- On restart with the same `--checkpoint` path, training resumes from the
  next batch after the last checkpoint.
- After successful completion the checkpoint file is **deleted**
  automatically.

### CLI Usage
```bash
python wisdom_yolo_train.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/train2017 \
  --batch-size 2 --num-images 118287 --top-m 20 \
  --methods lgxa lig lgs \
  --out-csv neuron_eval_out/wisdom_yolo11n_scores.csv \
  --device cuda:0 --imgsz 320 \
  --checkpoint neuron_eval_out/wisdom_yolo11n_ckpt.pt \
  --checkpoint-every 100
```

If the job is killed or OOMs, simply re-run the exact same command —
it picks up where it left off.

## End-to-End Pipeline

The full WISDOM-YOLOv11 pipeline from scratch:

```bash
cd Wisdom && source ../.venv/bin/activate

# Step 1: Generate importance scores (with checkpoint for long runs)
python wisdom_yolo_train.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/train2017 \
  --batch-size 2 --num-images 5000 --top-m 20 \
  --methods lgxa lig lgs --voting-mode coarse \
  --out-csv neuron_eval_out/wisdom_yolo11n_scores.csv \
  --device cuda:0 --imgsz 320 \
  --checkpoint neuron_eval_out/wisdom_yolo11n_ckpt.pt

# Step 2: RQ1 – Critical neuron pruning
python run_rq1.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/val2017 \
  --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv \
  --num-images 200 --batch-size 2 --imgsz 320 --device cuda:0

# Step 3: RQ2 – Diversity
python run_rq2.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/val2017 \
  --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv \
  --num-images 200 --batch-size 2 --imgsz 320 --device cuda:0

# Step 4: RQ3 – Adversarial effectiveness
python run_rq3.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/val2017 \
  --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv \
  --num-images 200 --batch-size 4 --imgsz 320 --device cuda:0

# Step 5: RQ4 – Correlation
python run_rq4.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/val2017 \
  --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv \
  --num-images 200 --imgsz 320 --device cuda:0

# Step 6: RQ5 – Efficiency
python run_rq5.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/val2017 \
  --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv \
  --num-images 50 --batch-size 2 --imgsz 320 --device cuda:0
```

> Increase `--num-images` for production runs (e.g. 5000 for val set,
> 118287 for full train set in Step 1).

## Notes

- **Train set** → `standalone/data/coco/images/train2017` (for `wisdom_yolo_train.py` only)
- **Val/test set** → `standalone/data/coco/images/val2017` (for all `run_rq*.py` scripts)
- Use `--imgsz 320` for testing (640 may cause OOM on smaller GPUs)
- Start with `--num-images` small values (10-20) for validation, increase for full experiments
- The WISDOM consensus scores CSV is shared across RQ1-RQ5 scripts
- Detection head layers (`model.23.*`) are automatically excluded from pruning
- RQ1 generates a PDF plot automatically using `viz_rq1_acc_drop()`

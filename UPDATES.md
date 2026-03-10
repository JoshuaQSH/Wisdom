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

## RQ1 Metrics

RQ1 now reports three metrics for each pruning level:
1. **Confidence Drop**: Sum of class confidence scores (existing)
2. **IoU Drop**: Mean Intersection-over-Union between predicted and ground-truth bounding boxes
3. **Classification Accuracy Drop**: Fraction of matched detections with correct class prediction

Results are saved to CSV and automatically visualized as a grouped bar chart PDF.

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

### 1. Generate Neuron Importance Scores

```bash
cd Wisdom
source ../.venv/bin/activate

python wisdom_yolo_train.py \
  --weights weights/yolo11n.pt \
  --img-dir standalone/data/coco/images/train2017 \
  --batch-size 2 --num-images 100 --top-m 20 \
  --methods lgxa lig lgs \
  --out-csv neuron_eval_out/wisdom_yolo11n_scores.csv \
  --device cuda:0 --imgsz 320
```

### 2. Run Research Questions

```bash
# RQ1: Critical neurons (with IoU + classification accuracy)
python run_rq1.py --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv --num-images 50

# RQ2: Diversity
python run_rq2.py --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv --num-images 20

# RQ3: Adversarial effectiveness
python run_rq3.py --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv --num-images 20

# RQ4: Correlation
python run_rq4.py --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv --num-images 30

# RQ5: Efficiency
python run_rq5.py --csv-file neuron_eval_out/wisdom_yolo11n_scores.csv --num-images 4
```

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

## Notes

- Use `--imgsz 320` for testing (640 may cause OOM on smaller GPUs)
- Start with `--num-images` small values (10-20) for validation, increase for full experiments
- The WISDOM consensus scores CSV is shared across RQ1-RQ5 scripts
- Detection head layers (`model.23.*`) are automatically excluded from pruning
- RQ1 generates a PDF plot automatically using `viz_rq1_acc_drop()`

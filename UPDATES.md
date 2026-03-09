# UPDATES – YOLOv11 Integration

This document describes the changes made to integrate YOLOv11 into the WISDOM framework.

## New Files

| File | Description |
|------|-------------|
| `yolo_test.py` | Verifies YOLOv11 model loading and inference on COCO images |
| `wisdom_yolo_train.py` | Consensus-based neuron importance scoring for YOLOv11 |
| `run_rq1.py` | RQ1: Critical neuron pruning with confidence drop measurement |
| `run_rq2.py` | RQ2: Diversity – perturb important vs random pixels, compare coverage |
| `run_rq3.py` | RQ3: Effectiveness – adversarial attack detection via coverage change |
| `run_rq4.py` | RQ4: Correlation between WISDOM coverage and Pielou's evenness |
| `run_rq5.py` | RQ5: Runtime and memory overhead measurement |
| `wisdom/utils/yolo_wrapper.py` | Wraps YOLO detection model for Captum attribution |
| `standalone/models/yolo11n.pt` | YOLOv11-nano pretrained weights |
| `tests/test_*.py` | Unit tests for all scripts (11 tests total) |

## Modified Files

| File | Changes |
|------|---------|
| `wisdom/core/wisdom_train.py` | Added YOLO support: `is_yolo` config flag, `_eval_loss_yolo()`, `_compute_yolo_importance()`, `_gradient_importance()` fallback, `_yolo_prune_eval()` |

## Dependencies

- `ultralytics` – YOLOv11 model loading and inference
- `captum` – Attribution methods (LayerGradientXActivation, IntegratedGradients, GradientShap)
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
  --weights standalone/models/yolo11n.pt \
  --img-dir standalone/data/coco/images/train2017 \
  --batch-size 2 --num-images 100 --top-m 20 \
  --methods lgxa lig lgs \
  --out-csv wisdom_yolo11n_scores.csv \
  --device cuda:0 --imgsz 320
```

### 2. Run Research Questions

```bash
# RQ1: Critical neurons
python run_rq1.py --csv-file wisdom_yolo11n_scores.csv --num-images 50

# RQ2: Diversity
python run_rq2.py --csv-file wisdom_yolo11n_scores.csv --num-images 20

# RQ3: Adversarial effectiveness
python run_rq3.py --csv-file wisdom_yolo11n_scores.csv --num-images 20

# RQ4: Correlation
python run_rq4.py --csv-file wisdom_yolo11n_scores.csv --num-images 30

# RQ5: Efficiency
python run_rq5.py --csv-file wisdom_yolo11n_scores.csv --num-images 4
```

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

## Notes

- Use `--imgsz 320` for testing (640 may cause OOM on smaller GPUs)
- Start with `--num-images` small values (10-20) for validation, increase for full experiments
- The WISDOM consensus scores CSV is shared across RQ1-RQ5 scripts

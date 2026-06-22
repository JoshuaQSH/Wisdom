# ETerry YOLOv5s WISDOM Use Case

This folder is a standalone ETerry detection package built around YOLOv5s.

## Contents

- `dataset/`: YOLO-format ETerry dataset. `images/train` and `labels/train` are used for retraining; `dataset/detect` contains unlabeled images for detection demos.
- `weights/`: YOLOv5s ETerry weights (`best.pt`, `last.pt`).
- `detect.py`, `train.py`, `val.py`: YOLOv5 detection, retraining, and validation entry points.
- `models/`, `utils/`, `data/hyps/`: local YOLOv5 support files required by detection and retraining.
- `wisdom_pretrain.py`: computes per-group WISDOM neuron rankings for YOLOv5s.
- `wisdom_coverage.py`: runs RQ2-style coverage testing and important-pixel visualizations.
- `video_wisdom_demo.py`: renders `videos/demo.mp4` with YOLOv5 boxes, a red WISDOM top-pixel overlay, and live coverage scores.
- `video_to_gif.py`: converts a short segment of an MP4 video to a GIF; defaults to 5 seconds.
- `results/`: detection, WISDOM, coverage, visualization, and report outputs.

## Common Commands

```bash
source ../.venv/bin/activate

python detect.py \
  --weights weights/best.pt \
  --source dataset/detect/01_harvesting_seq1_cropped/frame_000001.jpg \
  --data dataset/data.yaml \
  --img 640 \
  --device 0 \
  --project results/detect_test \
  --name yolov5s_best

python train.py \
  --data dataset/data.yaml \
  --weights weights/best.pt \
  --hyp data/hyps/hyp.scratch-low.yaml \
  --img 640 \
  --epochs 100 \
  --batch-size 16 \
  --project results/train \
  --name yolov5s_retrain

python wisdom_pretrain.py \
  --batch-size 8 \
  --workers 4 \
  --imgsz 320 \
  --device cuda:0 \
  --selection-mode per-group \
  --top-m 20 \
  --out-csv results/wisdom/wisdom_yolov5s_eterry_train.csv \
  --method-dir results/wisdom/method_scores \
  --report-json results/wisdom/wisdom_yolov5s_eterry_summary.json

python wisdom_coverage.py \
  --stage all \
  --wisdom-csv results/wisdom/wisdom_yolov5s_eterry_train.csv \
  --batch-size 8 \
  --workers 0 \
  --imgsz 320 \
  --device cuda:0 \
  --out-dir results/coverage \
  --per-group-k 5 \
  --pixel-fracs 0.02,0.05 \
  --visual-pixel-fracs 0.02,0.05 \
  --visual-count 2

python video_wisdom_demo.py \
  --workers 2 \
  --batch-size 16 \
  --output results/video_demo/demo_yolov5s_wisdom_heatmap.mp4

python video_to_gif.py results/video_demo/demo_yolov5s_wisdom_heatmap.mp4
```

See `results/ETERRY_YOLOV5S_WISDOM_REPORT.md` for the verified run results and exact CLIs used.

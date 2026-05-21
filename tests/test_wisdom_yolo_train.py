"""Tests for wisdom_yolo_train.py – consensus training on YOLOv11."""
import os
import csv
import tempfile
import pytest

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights", "yolo11n.pt")
COCO_TRAIN = os.path.join(os.path.dirname(__file__), "..", "standalone", "data", "coco", "images", "train2017")

skip_no_weights = pytest.mark.skipif(not os.path.isfile(WEIGHTS), reason="yolo11n.pt not found")
skip_no_data = pytest.mark.skipif(not os.path.isdir(COCO_TRAIN), reason="COCO train images not found")
skip_no_gpu = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="No GPU available"
)


@skip_no_weights
@skip_no_data
def test_train_wisdom_yolo_generates_csv():
    """Run on 2 images with 1 method to verify CSV generation."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from wisdom_yolo_train import train_wisdom_yolo

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "test_scores.csv")
        csv_path = train_wisdom_yolo(
            weights=WEIGHTS,
            img_dir=COCO_TRAIN,
            out_csv=out_csv,
            batch_size=2,
            num_images=2,
            top_m=5,
            methods=["lgxa"],
            voting_mode="coarse",
            device=device,
            imgsz=320,  # small for speed
        )

        assert os.path.isfile(csv_path), f"CSV not generated at {csv_path}"

        # Verify CSV structure
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "CSV is empty"
        assert set(reader.fieldnames) == {"LayerName", "NeuronIndex", "Score"}

        # At least some scores should be non-zero
        non_zero = sum(1 for r in rows if float(r["Score"]) != 0.0)
        assert non_zero > 0, "All scores are zero – voting produced no results"


@skip_no_weights
@skip_no_data
def test_train_wisdom_yolo_fine_grained():
    """Test fine-grained voting mode with 2 methods."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from wisdom_yolo_train import train_wisdom_yolo

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "fg_scores.csv")
        csv_path = train_wisdom_yolo(
            weights=WEIGHTS,
            img_dir=COCO_TRAIN,
            out_csv=out_csv,
            batch_size=2,
            num_images=2,
            top_m=5,
            methods=["lgxa", "lig"],
            voting_mode="fine-grained",
            device=device,
            imgsz=320,
        )
        assert os.path.isfile(csv_path)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0

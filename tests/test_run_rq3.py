"""Tests for run_rq3.py – adversarial effectiveness."""
import os
import tempfile
import pytest

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "standalone", "models", "yolo11n.pt")
COCO_VAL = os.path.join(os.path.dirname(__file__), "..", "standalone", "data", "coco", "images", "val2017")
SCORES_CSV = os.path.join(os.path.dirname(__file__), "..", "wisdom_yolo11n_scores.csv")

skip_missing = pytest.mark.skipif(
    not (os.path.isfile(WEIGHTS) and os.path.isdir(COCO_VAL) and os.path.isfile(SCORES_CSV)),
    reason="Missing weights, data, or scores CSV",
)


@skip_missing
def test_rq3_generates_csv():
    import sys, pathlib, csv
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from run_rq3 import run_rq3

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "rq3.csv")
        result = run_rq3(
            weights=WEIGHTS, img_dir=COCO_VAL, csv_file=SCORES_CSV,
            out_csv=out, device=device, num_images=4, batch_size=2, imgsz=320,
        )
        assert os.path.isfile(result)
        with open(result) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert "Attack" in rows[0]
        assert "Normalised Change" in rows[0]

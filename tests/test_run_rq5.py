"""Tests for run_rq5.py – efficiency evaluation."""
import os
import tempfile
import pytest

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights", "yolo11n.pt")
COCO_VAL = os.path.join(os.path.dirname(__file__), "..", "standalone", "data", "coco", "images", "val2017")
SCORES_CSV = os.path.join(os.path.dirname(__file__), "..", "neuron_eval_out", "wisdom_yolo11n_scores.csv")

skip_missing = pytest.mark.skipif(
    not (os.path.isfile(WEIGHTS) and os.path.isdir(COCO_VAL) and os.path.isfile(SCORES_CSV)),
    reason="Missing weights, data, or scores CSV",
)


@skip_missing
def test_rq5_generates_csv():
    import sys, pathlib, csv
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from run_rq5 import run_rq5

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "rq5.csv")
        result = run_rq5(
            weights=WEIGHTS, img_dir=COCO_VAL, csv_file=SCORES_CSV,
            out_csv=out, device=device, num_images=4, batch_size=2, imgsz=320,
        )
        assert os.path.isfile(result)
        with open(result) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 4, f"Expected at least 4 timing entries, got {len(rows)}"
        assert "Operation" in rows[0]
        assert "Time (s)" in rows[0]


@skip_missing
def test_rq5_timings_reasonable():
    """Check that individual operations complete in under 120s on small data."""
    import sys, pathlib, csv
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from run_rq5 import run_rq5

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "rq5.csv")
        run_rq5(
            weights=WEIGHTS, img_dir=COCO_VAL, csv_file=SCORES_CSV,
            out_csv=out, device=device, num_images=2, batch_size=2, imgsz=320,
        )
        with open(out) as f:
            for row in csv.DictReader(f):
                t = float(row["Time (s)"])
                assert t < 120, f"{row['Operation']} took {t}s (>120s)"

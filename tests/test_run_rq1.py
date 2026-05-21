"""Tests for run_rq1.py – critical neuron evaluation on YOLOv11."""
import os
import csv
import tempfile
import pytest

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights", "yolo11n.pt")
COCO_VAL = os.path.join(os.path.dirname(__file__), "..", "standalone", "data", "coco", "images", "val2017")
SCORES_CSV = os.path.join(os.path.dirname(__file__), "..", "neuron_eval_out", "wisdom_yolo11n_scores.csv")

skip_no_weights = pytest.mark.skipif(not os.path.isfile(WEIGHTS), reason="yolo11n.pt not found")
skip_no_data = pytest.mark.skipif(not os.path.isdir(COCO_VAL), reason="COCO val images not found")
skip_no_csv = pytest.mark.skipif(not os.path.isfile(SCORES_CSV), reason="WISDOM scores CSV not found")


@skip_no_weights
@skip_no_data
@skip_no_csv
def test_rq1_generates_output_csvs():
    """Run RQ1 on 2 images with 1 method and check output files."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from run_cases.run_rq1 import run_rq1

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "test_rq1")
        rel_path, drop_path = run_rq1(
            weights=WEIGHTS,
            img_dir=COCO_VAL,
            csv_file=SCORES_CSV,
            out_prefix=prefix,
            device=device,
            num_images=2,
            batch_size=2,
            imgsz=320,
            methods=["lgxa"],
            single_method_csvs={"lgxa": SCORES_CSV},
        )

        # Check files exist
        assert os.path.isfile(rel_path), f"Relevance CSV not found: {rel_path}"
        assert os.path.isfile(drop_path), f"Accuracy drop CSV not found: {drop_path}"

        # Check relevance CSV structure
        with open(rel_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "Relevance CSV is empty"
        assert "Attribution Method" in reader.fieldnames
        assert "Relevance Score" in reader.fieldnames

        # Check accuracy drop CSV structure
        with open(drop_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "Accuracy drop CSV is empty"
        assert "Attribution Method" in reader.fieldnames
        assert "Top-N" in reader.fieldnames
        assert "Confidence Drop" in reader.fieldnames

        # Should have entries for Wisdom + at least one attribution method + random
        methods = set(r["Attribution Method"] for r in rows)
        assert "Wisdom" in methods, f"Wisdom method missing. Found: {methods}"

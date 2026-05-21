"""Tests for YOLOv11 model loading and inference."""
import os
import pytest

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights", "yolo11n.pt")
COCO_VAL = os.path.join(os.path.dirname(__file__), "..", "standalone", "data", "coco", "images", "val2017")

skip_no_weights = pytest.mark.skipif(
    not os.path.isfile(WEIGHTS), reason="yolo11n.pt weights not found"
)
skip_no_data = pytest.mark.skipif(
    not os.path.isdir(COCO_VAL), reason="COCO val2017 images not found"
)


@skip_no_weights
def test_yolo_model_loads():
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    assert model.model is not None, "Model failed to load"


@skip_no_weights
@skip_no_data
def test_yolo_inference_produces_boxes():
    from ultralytics import YOLO
    import glob

    model = YOLO(WEIGHTS)
    imgs = sorted(glob.glob(os.path.join(COCO_VAL, "*.jpg")))[:3]
    assert len(imgs) > 0, "No images found"
    results = model(imgs, verbose=False)
    assert len(results) == len(imgs)
    # At least one image should have detections
    total_boxes = sum(len(r.boxes) for r in results)
    assert total_boxes > 0, "No detections on any image"


@skip_no_weights
@skip_no_data
def test_yolo_inference_result_fields():
    from ultralytics import YOLO
    import glob

    model = YOLO(WEIGHTS)
    imgs = sorted(glob.glob(os.path.join(COCO_VAL, "*.jpg")))[:1]
    results = model(imgs, verbose=False)
    r = results[0]
    # Check that Result has expected attributes
    assert hasattr(r, "boxes")
    assert hasattr(r.boxes, "cls")
    assert hasattr(r.boxes, "conf")
    assert hasattr(r.boxes, "xyxy")

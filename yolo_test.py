"""
yolo_test.py - Verify YOLOv11 model loading and inference on COCO images.
"""
import glob
from ultralytics import YOLO

WEIGHTS = "weights/yolo11n.pt"
COCO_VAL = "standalone/data/coco/images/val2017"


def load_and_infer(weights: str = WEIGHTS, img_dir: str = COCO_VAL, n_images: int = 3):
    """Load YOLOv11 and run inference on a few images. Returns list of Results."""
    model = YOLO(weights)
    imgs = sorted(glob.glob(f"{img_dir}/*.jpg"))[:n_images]
    if not imgs:
        raise FileNotFoundError(f"No images found in {img_dir}")
    results = model(imgs, verbose=False)
    for r in results:
        fname = r.path.split("/")[-1]
        n_boxes = len(r.boxes)
        print(f"  {fname}: {n_boxes} detections")
    return results


if __name__ == "__main__":
    print("Testing YOLOv11 inference...")
    results = load_and_infer()
    print(f"Done – processed {len(results)} images.")

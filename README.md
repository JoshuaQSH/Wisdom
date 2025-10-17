# WISDOM for YOLO and transformer-based models

A developed branch for YOLOvx and other transformer-based model Testing. 

# TODO

- [ ] [**Dev**] Tool combination with [Introduction: Advanced Explainable AI for computer vision](https://jacobgil.github.io/pytorch-gradcam-book/introduction.html) and [Layer-wise Relevance Propagation for Transformers](https://github.com/rachtibat/LRP-eXplains-Transformers?tab=readme-ov-file#getting-started)
- [x] [**YOLO**] [YOLOv5](https://github.com/mihir135/yolov5) prepared.
  - [ ] Testing with single layer w/ WISDOM
  - [ ] 
- [ ] [**YOLO**] [YOLOv11](https://docs.ultralytics.com/models/yolo11/) prepared
- [ ] [**Compression**] YOLOv11 comression test with attribution methods
- [ ] [**Compression**] LLM with compression

# Tools and Design

## YOLO recommanded dataset form

```shell
/datasets/
└── coco128/  # Dataset root
    ├── images/
    │   ├── train2017/  # Training images
    │   │   ├── 000000000009.jpg
    │   │   └── ...
    │   └── val2017/    # Validation images (optional if using same set for train/val)
    │       └── ...
    └── labels/
        ├── train2017/  # Training labels
        │   ├── 000000000009.txt
        │   └── ...
        └── val2017/    # Validation labels (optional if using same set for train/val)
            └── ...
```
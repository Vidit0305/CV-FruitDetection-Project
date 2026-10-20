# Fruit Detection Dataset Guide

This document provides instructions for acquiring, structuring, and labeling datasets for training and fine-tuning the Real-Time Fruit Detection Model.

---

## 1. Supported Fruit Classes

The standard target classes for this project are:

| Class ID | Fruit Name   | Description / Visual Notes |
|:--------:|:-------------|:---------------------------|
| 0        | Apple        | Red, green, and gala apples |
| 1        | Banana       | Yellow ripe and green unpeeled bananas |
| 2        | Orange       | Whole and sliced citrus oranges / mandarins |
| 3        | Mango        | Ripe yellow, orange, and green mangoes |
| 4        | Grapes       | Bunches and clusters of green, red, or black grapes |
| 5        | Pineapple    | Whole pineapples with crowns |
| 6        | Watermelon   | Whole and sliced red-fleshed watermelons |
| 7        | Strawberry   | Ripe red strawberries |
| 8        | Pear         | Green and yellow European/Asian pears |
| 9        | Lemon        | Whole and sliced yellow lemons / limes |

---

## 2. Directory Structure

Ultralytics YOLO expects datasets organized into separate `images/` and `labels/` folders for each split:

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── apple_001.jpg
│   │   └── banana_002.jpg
│   └── labels/
│       ├── apple_001.txt
│       └── banana_002.txt
├── valid/
│   ├── images/
│   │   └── orange_001.jpg
│   └── labels/
│       └── orange_001.txt
└── test/
    ├── images/
    └── labels/
```

---

## 3. YOLO Bounding-Box Annotation Format

Each image in `images/` must have a corresponding `.txt` file in `labels/` with the exact same base name.

Each line in the `.txt` file represents one bounding box in normalized coordinates:

```text
<class_id> <x_center> <y_center> <width> <height>
```

- `<class_id>`: 0-indexed integer corresponding to the index in `data.yaml`.
- `<x_center>`: Horizontal center coordinate divided by image width (0.0 to 1.0).
- `<y_center>`: Vertical center coordinate divided by image height (0.0 to 1.0).
- `<width>`: Box width divided by image width (0.0 to 1.0).
- `<height>`: Box height divided by image height (0.0 to 1.0).

### Example (`train/labels/apple_001.txt`):
```text
0 0.512000 0.485000 0.350000 0.420000
```

---

## 4. Recommended Public Datasets

You can source pre-annotated multi-class fruit datasets from public repositories:

### A. Roboflow Universe
1. **Fruits Object Detection Dataset**:
   Search Roboflow Universe for `fruits object detection` or visit [Roboflow Universe](https://universe.roboflow.com/).
2. Select **Export Dataset** -> Format: **YOLOv8** -> Download zip.
3. Unzip directly into `dataset/`.

### B. Lightly AI Fruit Detection Dataset
- Public repository: `github.com/lightly-ai/dataset_fruits_detection`
- Contains over 8,000 images covering Apples, Bananas, Grapes, Oranges, Pineapples, and Watermelons in YOLO format.

### C. LVIS Fruits & Vegetables Subset
- Comprehensive 63-class produce dataset derived from LVIS instance annotations.
- Used to train high-capacity produce detectors.

---

## 5. Labeling Custom Images

If collecting custom webcam or phone photos of fruits:

1. **Capture Varied Data**:
   - Different lighting (natural sunlight, warm indoor light, dim shadows).
   - Varied backgrounds (kitchen counters, cutting boards, tables, hands holding the fruit).
   - Single-fruit and multi-fruit arrangements.
   - Rotations, distances, and partial occlusions.
2. **Annotation Tools**:
   - **Roboflow** (browser-based, auto-export in YOLO format).
   - **LabelImg** (local Python tool: `pip install labelImg && labelImg`).
   - **CVAT** (Computer Vision Annotation Tool).

---

## 6. Training with `train.py`

Once your dataset is arranged with `data.yaml`:

```bash
python train.py --data dataset/data.yaml --epochs 40 --batch 16
```

The script will automatically train the model and deploy the best checkpoint to `models/best.pt`.

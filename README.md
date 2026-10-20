# Real-Time Fruit Detection Model Using Python and Computer Vision

A desktop computer vision application built with **Python 3**, **OpenCV**, and **Ultralytics YOLO** for real-time fruit detection and classification from a live webcam feed.

The system performs visual inference on each frame, identifies supported fruits, draws high-contrast bounding boxes, displays live confidence scores, tracks fruit counts, and presents a stabilized status banner.

---

## 1. Supported Fruits

The model is configured and verified to identify common fruit categories:

- 🍎 **Apple**
- 🍌 **Banana**
- 🍊 **Orange**
- 🥭 **Mango**
- 🍇 **Grapes**
- 🍍 **Pineapple**
- 🍉 **Watermelon**
- 🍓 **Strawberry**
- 🍐 **Pear**
- 🍋 **Lemon**

Multiple fruits—including different varieties or multiple instances of the same fruit—can be detected simultaneously in a single frame.

---

## 2. Technology Stack

- **Python 3.10+** (Python 3.11 / 3.12 compatible)
- **OpenCV (`opencv-python`)**: Video capture, frame preprocessing, bounding-box rendering, and live visualization.
- **Ultralytics YOLO (`ultralytics`)**: Deep learning object detection architecture.
- **PyTorch (`torch`, `torchvision`)**: Tensor computation engine (supports CPU inference and CUDA GPU acceleration).
- **NumPy**: Matrix and image array manipulation.

---

## 3. Project Structure

```text
fruit-detection-model/
├── main.py                 # Application entry point & webcam detection loop
├── train.py                # Model training and fine-tuning pipeline
├── requirements.txt        # Python dependency manifest
├── README.md               # Complete project documentation
├── .gitignore              # Git ignore rules
├── models/
│   ├── best.pt             # Active deployed YOLO checkpoint (default)
│   ├── yolov8n_fruit.pt    # Fast 35-class fruit & produce detector
│   └── snapstock_yolov8m.pt# Comprehensive 63-class produce detector
├── dataset/
│   ├── data.yaml           # Dataset configuration template
│   └── README.md           # Dataset acquisition and annotation guide
├── captures/               # Destination for screenshots saved with 'S'
├── tests/
│   └── test_pipeline.py    # Automated test suite (5 verification checks)
├── test_images/            # Sample fruit test images for offline testing
├── src/
│   ├── __init__.py
│   ├── detector.py         # FruitDetector class & inference handling
│   ├── camera.py           # Camera abstraction with error diagnostics
│   └── visualizer.py       # Bounding boxes, labels, and UI overlays
└── config/
    ├── __init__.py
    └── settings.py         # Centralized hyperparameters, colors & paths
```

---

## 4. Installation & Setup

### Step 1: Clone or Navigate to the Project

```bash
cd "/home/vidit/Documents/Work/Projects/Minor Projects/CV-FruitsDetection-Project"
```

### Step 2: Create a Virtual Environment

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

To install all dependencies:

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** If running on a CPU-only machine, you can install the lightweight CPU-optimized PyTorch build for faster download:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

## 5. Usage & Execution

### Basic Execution (Default Webcam)

Launch the real-time detection interface with:

```bash
python main.py
```

Upon launching, the terminal will report:
```text
Loading fruit detection model...
Model loaded. Starting webcam...
[*] Checkpoint: .../models/best.pt
[*] Device:     CPU
[*] Threshold:  40% confidence
[*] Fruits:     Apple, Banana, Orange, Mango, Grapes, Pineapple, Watermelon, Strawberry, Pear
[*] Keyboard:   [Q / ESC] Quit  |  [S] Save Frame
```

An OpenCV desktop window titled **"Real-Time Fruit Detection"** will open showing the camera feed.

---

### Command-Line Options

The application supports flexible parameters:

```bash
python main.py [OPTIONS]
```

| Flag | Default | Description |
|:---|:---|:---|
| `--source` | `0` | Camera index (e.g. `0`, `1`), or path to a video/image file |
| `--model` | `models/best.pt` | Path to a specific YOLO `.pt` checkpoint |
| `--conf` | `0.40` | Minimum confidence score threshold (0.0 to 1.0) |
| `--iou` | `0.45` | IoU threshold for Non-Maximum Suppression (NMS) |
| `--imgsz` | `640` | Image inference resolution |
| `--device` | Auto | Compute device (`cpu` or `cuda`) |
| `--save-dir`| `captures` | Directory where screenshots are saved |
| `--all-classes` | `False` | Display all detected produce classes instead of filtering |

---

### Usage Examples

1. **Test with a static image (without webcam):**
   ```bash
   python main.py --source test_images/apple.jpg
   ```

2. **Use a secondary USB webcam (index 1):**
   ```bash
   python main.py --source 1
   ```

3. **Adjust confidence threshold to 50%:**
   ```bash
   python main.py --conf 0.50
   ```

4. **Switch to the 63-class LVIS produce model:**
   ```bash
   python main.py --model models/snapstock_yolov8m.pt
   ```

---

## 6. Keyboard Controls

While the camera window is active, use the following keyboard shortcuts:

| Key | Action |
|:---|:---|
| **`Q`** | Safely quit the application and release camera resources |
| **`ESC`** | Exit the camera window immediately |
| **`S`** | Capture and save the current annotated frame to `captures/` |

When **`S`** is pressed, the application saves an image named `fruit_detection_YYYYMMDD_HHMMSS.jpg` in the `captures/` folder and flashes a confirmation banner on screen.

---

## 7. User Interface Elements

The OpenCV interface is clean, minimal, and non-distracting:

1. **Bounding Boxes & Labels:**
   - Outlines around each detected fruit in distinctive, class-specific colors.
   - High-contrast tag showing `FruitName | XX.X%` (e.g. `Apple | 96.4%`).
2. **Top Status Banner:**
   - **Status Message:**
     - `No fruit detected — Show a fruit to the camera` (neutral amber indicator)
     - `Detected: Apple` or `Detected: Apple, Banana, Orange` (vibrant green indicator)
     - Stabilized with temporal smoothing so it does not flicker on momentary detection gaps.
   - **Counter:** Compact badge showing total detected fruits (e.g. `Count: 2`).
   - **FPS Display:** Live inference and rendering frames-per-second (`FPS: 28.5`).
3. **Bottom Helper Bar:**
   - Unobtrusive reminder of keyboard shortcuts: `[Q] Quit    [ESC] Exit    [S] Save Frame`.
   - Displays confirmation messages when a frame is saved.

---

## 8. Model Training and Fine-Tuning

A dedicated training script is provided in `train.py` to train or fine-tune models on custom annotated datasets:

```bash
python train.py --data dataset/data.yaml --epochs 30 --batch 16
```

### Training Options:

| Flag | Default | Description |
|:---|:---|:---|
| `--data` | `dataset/data.yaml` | Path to dataset configuration YAML |
| `--base-model` | `yolov8n.pt` | Pretrained base weights for transfer learning |
| `--epochs` | `30` | Number of training iterations |
| `--batch` | `16` (CUDA) / `4` (CPU) | Batch size |
| `--device` | Auto | `cuda` if GPU available, otherwise `cpu` |
| `--output` | `models/best.pt` | Destination for the best trained checkpoint |

See [dataset/README.md](file:///home/vidit/Documents/Work/Projects/Minor%20Projects/CV-FruitsDetection-Project/dataset/README.md) for full instructions on sourcing datasets from Roboflow Universe, setting up YOLO annotations, and preparing custom images.

---

## 9. Verification & Testing

An automated test suite is included to verify all components end-to-end:

```bash
python tests/test_pipeline.py
```

The test suite validates:
1. **Missing Model Error Handling:** Confirms descriptive error output when checkpoint is missing.
2. **Detector Initialization:** Confirms model loading and class discovery.
3. **Visual Inference:** Verifies detection on sample fruit images.
4. **UI Visualizer:** Validates bounding-box coordinates, status banner smoothing, and text styling.
5. **Camera & Capture:** Verifies frame reading and screenshot saving.

---

## 10. Troubleshooting & FAQ

### Problem: `[ERROR] Failed to open video source: '0'`
- **Cause:** Webcam is in use by another application or permissions are missing.
- **Solution:** Close video conferencing apps (Zoom, Teams, browser). On Linux, verify `/dev/video0` permissions (`ls -l /dev/video*`). If you have multiple cameras, try `python main.py --source 1`.

### Problem: `[ERROR] Model weights file not found`
- **Cause:** Checkpoint file missing at specified path.
- **Solution:** Verify `models/best.pt` exists or specify a custom checkpoint using `--model path/to/weights.pt`.

### Problem: Low FPS on CPU
- **Tips for CPU Performance:**
  - Lower the inference size: `python main.py --imgsz 480`
  - Use the lightweight nano model: `models/best.pt` or `models/yolov8n_fruit.pt` (runs 2-3x faster than medium models on CPU).

---

## 11. Limitations & Future Work

- **Lighting & Occlusion:** Extreme low-light conditions or heavy occlusions may reduce detection confidence. Presenting the fruit clearly within the central third of the camera frame ensures optimal recognition.
- **Visual Similarity:** Unripe green apples and green pears share similar colors; good lighting helps the model distinguish subtle contour and stem differences.
- **Model Checkpoints:** Both lightweight nano weights (`yolov8n_fruit.pt`) and high-capacity medium weights (`snapstock_yolov8m.pt`) are included out-of-the-box in `models/`.

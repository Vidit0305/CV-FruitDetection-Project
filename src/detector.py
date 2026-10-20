"""Fruit Detection module using Ultralytics YOLO.

Initializes the model once, validates supported fruit classes, performs visual
inference, and formats detections for clean downstream visualization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set
import numpy as np
from ultralytics import YOLO

from config.settings import (
    DEFAULT_MODEL_PATH,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    AUTO_DEVICE,
    TARGET_FRUIT_MAPPING,
    CANONICAL_FRUITS,
)


@dataclass
class Detection:
    """Represents a single detected fruit in an image frame."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_name: str                 # Standardized display name, e.g. "Apple"
    raw_class: str                  # Original class label from the model
    confidence: float               # Model confidence score (0.0 to 1.0)
    color: Tuple[int, int, int]     # BGR color tuple for rendering


class FruitDetector:
    """Handles object detection model loading, configuration, and inference."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf_threshold: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        imgsz: int = DEFAULT_IMAGE_SIZE,
        device: Optional[str] = None,
        filter_target_fruits: bool = True,
    ):
        """Initialize the Fruit Detector with a specified YOLO model.

        Args:
            model_path: Path to the .pt checkpoint. Defaults to models/best.pt.
            conf_threshold: Minimum confidence score for valid detections.
            iou_threshold: IoU threshold for Non-Maximum Suppression (NMS).
            imgsz: Input image inference size (e.g. 640).
            device: 'cpu' or 'cuda'. If None, auto-detected.
            filter_target_fruits: If True, only returns detections matching target fruits.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device or AUTO_DEVICE
        self.filter_target_fruits = filter_target_fruits

        self._validate_model_path()
        self.model = self._load_model()
        self.supported_targets = self._inspect_model_classes()

    def _validate_model_path(self) -> None:
        """Verify that the model checkpoint exists."""
        if not self.model_path.exists():
            error_msg = (
                f"\n[ERROR] Model weights file not found at: {self.model_path.resolve()}\n"
                f"Please ensure a valid YOLO fruit detection model exists.\n"
                f"Tips:\n"
                f"  1. Place your trained checkpoint at 'models/best.pt'\n"
                f"  2. Or specify a custom model using: python main.py --model path/to/model.pt\n"
                f"  3. Or fine-tune a model locally using: python train.py\n"
            )
            raise FileNotFoundError(error_msg)

    def _load_model(self) -> YOLO:
        """Load the Ultralytics YOLO model onto the selected device."""
        try:
            model = YOLO(str(self.model_path))
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model from '{self.model_path}': {exc}"
            ) from exc

    def _inspect_model_classes(self) -> Set[str]:
        """Inspect model class names and determine supported target fruits."""
        model_classes = self.model.names if hasattr(self.model, "names") else {}
        supported: Set[str] = set()

        for class_name in model_classes.values():
            normalized = class_name.strip().lower()
            if normalized in TARGET_FRUIT_MAPPING:
                display_name = TARGET_FRUIT_MAPPING[normalized]["display"]
                supported.add(display_name)

        return supported

    def get_supported_fruits(self) -> List[str]:
        """Return the canonical fruit classes recognized by this model."""
        return [f for f in CANONICAL_FRUITS if f in self.supported_targets]

    def get_all_model_classes(self) -> Dict[int, str]:
        """Return the complete class mapping embedded in the YOLO model."""
        return dict(self.model.names) if hasattr(self.model, "names") else {}

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """Run object detection on an image frame and return fruit detections.

        Args:
            frame: BGR image frame (NumPy array) from OpenCV.
            conf_threshold: Optional override for the confidence threshold.

        Returns:
            List of Detection objects containing bounding boxes, labels, and confidence.
        """
        if frame is None or frame.size == 0:
            return []

        active_conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        try:
            results = self.model.predict(
                source=frame,
                conf=active_conf,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            # Prevent camera loop crash on transient inference failure
            print(f"[WARNING] Inference failed on current frame: {exc}")
            return []

        detections: List[Detection] = []
        if not results:
            return detections

        first_result = results[0]
        boxes = first_result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            raw_name = self.model.names.get(cls_id, f"class_{cls_id}")
            normalized = raw_name.strip().lower()

            # Check if this class is mapped to one of our target fruits
            if normalized in TARGET_FRUIT_MAPPING:
                mapping = TARGET_FRUIT_MAPPING[normalized]
                display_name = mapping["display"]
                color = mapping["color"]
            else:
                if self.filter_target_fruits:
                    # Skip non-target produce if filtering is enabled
                    continue
                display_name = raw_name.capitalize()
                color = (200, 200, 200)

            # Coordinates in pixels (x1, y1, x2, y2)
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

            # Ensure coordinates are within valid frame boundaries
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    class_name=display_name,
                    raw_class=raw_name,
                    confidence=conf,
                    color=color,
                )
            )

        return detections

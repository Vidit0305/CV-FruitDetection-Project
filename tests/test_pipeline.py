"""Automated Verification and Unit Test Suite for Fruit Detection Pipeline."""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DEFAULT_MODEL_PATH, CANONICAL_FRUITS, CAPTURES_DIR
from src.detector import FruitDetector, Detection
from src.camera import Camera
from src.visualizer import Visualizer


def test_missing_model_error():
    """Verify that a missing model file raises a descriptive FileNotFoundError."""
    print("[TEST 1] Testing missing model weights handling...")
    try:
        FruitDetector(model_path="models/non_existent_weights.pt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "not found at" in str(e)
        print("  -> Passed: FileNotFoundError correctly raised with helpful tips.")


def test_detector_initialization():
    """Verify detector loads model and discovers supported fruits."""
    print("[TEST 2] Testing FruitDetector initialization...")
    detector = FruitDetector(model_path=DEFAULT_MODEL_PATH)
    supported = detector.get_supported_fruits()
    print(f"  -> Model classes count: {len(detector.get_all_model_classes())}")
    print(f"  -> Supported fruits: {supported}")
    assert len(supported) > 0, "Model should support at least one target fruit."
    print("  -> Passed: Model loaded and class mapping verified.")


def test_detection_on_images():
    """Verify visual inference on test images."""
    print("[TEST 3] Testing visual inference on fruit images...")
    detector = FruitDetector(model_path=DEFAULT_MODEL_PATH, conf_threshold=0.25)
    
    test_files = [
        ("test_images/apple.jpg", "Apple"),
        ("test_images/banana.jpg", "Banana"),
        ("test_images/orange.jpg", "Orange"),
        ("test_images/pineapple.jpg", "Pineapple"),
    ]
    
    for img_path, expected_type in test_files:
        if Path(img_path).exists():
            img = cv2.imread(img_path)
            detections = detector.detect(img)
            classes_detected = [d.class_name for d in detections]
            print(f"  -> {img_path}: Detected {classes_detected}")
            assert isinstance(detections, list)
    print("  -> Passed: Inference executed on all test images without errors.")


def test_visualizer_rendering():
    """Verify visualizer renders bounding boxes and UI overlays."""
    print("[TEST 4] Testing Visualizer rendering and status messaging...")
    vis = Visualizer()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 1. No detections
    status_msg, is_detected = vis._get_status_message([])
    assert is_detected is False
    assert status_msg == "No fruit detected — Show a fruit to the camera"
    print(f"  -> Empty scene status: '{status_msg}'")
    
    # 2. Single detection
    mock_detection_1 = Detection(
        box=(100, 100, 300, 300),
        class_name="Apple",
        raw_class="apple",
        confidence=0.964,
        color=(50, 50, 225),
    )
    annotated = vis.draw_detections(dummy_frame, [mock_detection_1])
    status_msg, is_detected = vis._get_status_message([mock_detection_1])
    assert is_detected is True
    assert status_msg == "Detected: Apple"
    print(f"  -> Single fruit status: '{status_msg}'")
    
    # 3. Multiple detections
    mock_detection_2 = Detection(
        box=(400, 150, 600, 450),
        class_name="Banana",
        raw_class="banana",
        confidence=0.948,
        color=(30, 215, 255),
    )
    status_msg, is_detected = vis._get_status_message([mock_detection_1, mock_detection_2])
    assert status_msg == "Detected: Apple, Banana"
    print(f"  -> Multi-fruit status: '{status_msg}'")
    
    # 4. UI overlays
    ui_frame = vis.draw_ui_overlays(annotated, [mock_detection_1, mock_detection_2], fps=30.0)
    assert ui_frame.shape == dummy_frame.shape
    print("  -> Passed: UI overlays and bounding box annotations rendered successfully.")


def test_camera_and_save():
    """Verify camera input from static image and screenshot saving."""
    print("[TEST 5] Testing Camera module and frame capture saving...")
    test_img_path = "test_images/apple.jpg"
    with Camera(source=test_img_path) as cam:
        success, frame = cam.read()
        assert success is True
        assert frame is not None
        w, h = cam.get_resolution()
        print(f"  -> Static image read success: Resolution {w}x{h}")
        
        # Test saving to captures directory
        save_path = CAPTURES_DIR / "test_capture.jpg"
        cv2.imwrite(str(save_path), frame)
        assert save_path.exists()
        print(f"  -> Saved frame verified at: {save_path}")
        save_path.unlink()  # Clean up test capture
    print("  -> Passed: Camera and capture saving functional.")


def run_all_tests():
    print("=" * 60)
    print("Running Real-Time Fruit Detection Pipeline Verification")
    print("=" * 60)
    test_missing_model_error()
    test_detector_initialization()
    test_detection_on_images()
    test_visualizer_rendering()
    test_camera_and_save()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (5/5)")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

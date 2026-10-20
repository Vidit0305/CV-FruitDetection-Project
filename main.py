"""Real-Time Fruit Detection Model Entry Point.

Runs the live webcam detection loop, identifies fruits, renders high-contrast
bounding boxes, confidence scores, and clean status overlays.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time

# Suppress harmless Qt font warnings from OpenCV
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

import cv2

from config.settings import (
    DEFAULT_MODEL_PATH,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_CAMERA_INDEX,
    CAPTURES_DIR,
    FPS_SMOOTHING_FACTOR,
)
from src.detector import FruitDetector
from src.camera import Camera
from src.visualizer import Visualizer


def parse_arguments() -> argparse.Namespace:
    """Parse command-line execution parameters."""
    parser = argparse.ArgumentParser(
        description="Real-Time Fruit Detection Model using OpenCV and Ultralytics YOLO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_CAMERA_INDEX,
        help="Camera index (e.g. 0), video file path, or image file path",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to YOLO detection model checkpoint (.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Detection confidence threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="Non-Maximum Suppression (NMS) IoU threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Inference image resolution size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference compute device ('cpu', 'cuda', or auto-detect)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(CAPTURES_DIR),
        help="Directory to store captured screenshots",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Show all model classes instead of filtering to target fruits",
    )
    return parser.parse_args()


def save_screenshot(frame, save_dir: Path, visualizer: Visualizer) -> None:
    """Save the current frame to the captures directory with timestamp."""
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fruit_detection_{timestamp}.jpg"
    filepath = save_dir / filename

    try:
        cv2.imwrite(str(filepath), frame)
        msg = f"Saved: {filename}"
        print(f"[CAPTURE] Screenshot saved successfully: {filepath.resolve()}")
        visualizer.trigger_notification(msg, duration=2.5)
    except Exception as exc:
        print(f"[ERROR] Failed to save screenshot: {exc}")


def main() -> None:
    """Main application loop."""
    args = parse_arguments()
    save_dir = Path(args.save_dir)

    # 1. Startup: Load model
    print("Loading fruit detection model...")
    try:
        detector = FruitDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            filter_target_fruits=not args.all_classes,
        )
    except FileNotFoundError as fnf_err:
        print(fnf_err)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[FATAL] Model initialization failed: {exc}")
        sys.exit(1)

    # 2. Display model status
    supported = detector.get_supported_fruits()
    print("Model loaded. Starting webcam...")
    print(f"[*] Checkpoint: {detector.model_path.resolve()}")
    print(f"[*] Device:     {detector.device.upper()}")
    print(f"[*] Threshold:  {detector.conf_threshold * 100:.0f}% confidence")
    print(f"[*] Fruits:     {', '.join(supported) if supported else 'All detected classes'}")
    print("[*] Keyboard:   [Q / ESC] Quit  |  [S] Save Frame\n")

    # 3. Initialize Camera
    try:
        camera = Camera(source=args.source)
    except Exception as exc:
        print(f"[FATAL] Camera initialization failed: {exc}")
        sys.exit(1)

    # 4. Initialize Visualizer & Window
    visualizer = Visualizer()
    window_name = "Real-Time Fruit Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # FPS calculation tracking
    fps = 0.0
    prev_time = time.time()

    try:
        while True:
            # Capture frame
            success, frame = camera.read()
            if not success or frame is None:
                if camera.is_image_file:
                    # Allow displaying static image repeatedly until user quits
                    time.sleep(0.03)
                    continue
                print("[INFO] Video stream ended or no frame received.")
                break

            # Calculate FPS
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0:
                current_fps = 1.0 / dt
                fps = (
                    current_fps
                    if fps == 0.0
                    else (FPS_SMOOTHING_FACTOR * fps + (1.0 - FPS_SMOOTHING_FACTOR) * current_fps)
                )

            # Perform detection
            detections = detector.detect(frame)

            # Draw bounding boxes and labels
            annotated = visualizer.draw_detections(frame, detections)

            # Draw clean UI status banner and control bar
            display_frame = visualizer.draw_ui_overlays(annotated, detections, fps=fps)

            # Display in OpenCV window
            cv2.imshow(window_name, display_frame)

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'Q' or ESC
                print("[INFO] Exit signal received. Quitting...")
                break
            elif key in (ord("s"), ord("S")):   # 'S'
                save_screenshot(display_frame, save_dir, visualizer)

            # Check if user closed the window using the window manager 'X'
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed by user.")
                    break
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Exiting safely...")
    except Exception as exc:
        print(f"\n[ERROR] An unexpected error occurred during execution: {exc}")
    finally:
        # Safe cleanup
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released successfully. Goodbye.")


if __name__ == "__main__":
    main()

"""Webcam and video capture module using OpenCV.

Manages camera initialization, resolution configuration, frame capture,
and safe resource cleanup with comprehensive error handling.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from config.settings import DEFAULT_CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class Camera:
    """Manages OpenCV video capture for webcams, video files, or image inputs."""

    def __init__(
        self,
        source: Union[int, str] = DEFAULT_CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
    ):
        """Initialize video capture from a webcam index or media file.

        Args:
            source: Webcam index (int, e.g. 0), string integer ("0"),
                    or path to video / image file.
            width: Desired capture frame width in pixels.
            height: Desired capture frame height in pixels.
        """
        self.source = self._parse_source(source)
        self.requested_width = width
        self.requested_height = height
        self.is_image_file = False
        self._image_cache: Optional[np.ndarray] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self._initialize_capture()

    def _parse_source(self, source: Union[int, str]) -> Union[int, str]:
        """Convert string numbers to integer camera indices."""
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    def _initialize_capture(self) -> None:
        """Open the video stream or load static image."""
        # Check if source is a static image file
        if isinstance(self.source, str) and Path(self.source).is_file():
            suffix = Path(self.source).suffix.lower()
            if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                self.is_image_file = True
                self._image_cache = cv2.imread(self.source)
                if self._image_cache is None:
                    raise IOError(f"Could not decode image from '{self.source}'")
                return

        # Initialize VideoCapture
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self._handle_camera_open_error()

        # Attempt to set frame resolution (driver may pick nearest supported resolution)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)

    def _handle_camera_open_error(self) -> None:
        """Provide detailed diagnostic error when camera fails to open."""
        is_index = isinstance(self.source, int)
        error_lines = [
            f"\n[ERROR] Failed to open video source: '{self.source}'"
        ]

        if is_index:
            error_lines.extend([
                "Troubleshooting steps:",
                f"  1. Ensure your webcam is connected and recognized (camera index {self.source}).",
                "  2. Check if another application (Zoom, browser, etc.) is currently using the camera.",
                "  3. On Linux, ensure user permissions for video devices (e.g. check /dev/video*).",
                f"  4. Try an alternate camera index, e.g.: python main.py --source {1 if self.source == 0 else 0}",
                "  5. To test without a live webcam, use an image: python main.py --source test_images/apple.jpg",
            ])
        else:
            error_lines.extend([
                f"Troubleshooting: Verify the file exists and is a valid video format: '{self.source}'"
            ])

        raise RuntimeError("\n".join(error_lines))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture and return the next frame from the source.

        Returns:
            Tuple (success, frame), where frame is a BGR NumPy array or None.
        """
        if self.is_image_file:
            if self._image_cache is not None:
                return True, self._image_cache.copy()
            return False, None

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def get_resolution(self) -> Tuple[int, int]:
        """Return the actual resolution (width, height) of the video feed."""
        if self.is_image_file and self._image_cache is not None:
            h, w = self._image_cache.shape[:2]
            return w, h
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return self.requested_width, self.requested_height

    def release(self) -> None:
        """Release the camera and free underlying system resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release camera on context exit."""
        self.release()

"""Visualization module for rendering bounding boxes, labels, and UI overlays.

Provides clean, high-contrast, professional visual elements without visual clutter.
Includes temporal smoothing for stable status messages and clear keyboard prompts.
"""

import time
from typing import List, Optional, Tuple, Dict
from collections import Counter
import cv2
import numpy as np

from src.detector import Detection
from config.settings import (
    UI_COLORS,
    STATUS_HOLD_SECONDS,
)


class Visualizer:
    """Renders fruit detections, high-contrast labels, and clean status overlays."""

    def __init__(self):
        # Temporal smoothing state for status message
        self.last_detected_names: List[str] = []
        self.last_detection_timestamp: float = 0.0

        # Temporary banner notifications (e.g., screenshot saved)
        self.notification_text: Optional[str] = None
        self.notification_expiry: float = 0.0

    def trigger_notification(self, text: str, duration: float = 2.0) -> None:
        """Display a temporary confirmation banner in the bottom bar."""
        self.notification_text = text
        self.notification_expiry = time.time() + duration

    def _get_status_message(self, detections: List[Detection]) -> Tuple[str, bool]:
        """Compute a stabilized status message and detection state.

        Returns:
            Tuple of (status_text, is_fruit_detected).
        """
        now = time.time()

        if detections:
            # Unique fruit names detected in current frame, preserving order
            detected_names: List[str] = []
            for d in detections:
                if d.class_name not in detected_names:
                    detected_names.append(d.class_name)

            self.last_detected_names = detected_names
            self.last_detection_timestamp = now
            return f"Detected: {', '.join(detected_names)}", True

        # Check if we should hold the previous detection message temporarily
        if (now - self.last_detection_timestamp) < STATUS_HOLD_SECONDS and self.last_detected_names:
            return f"Detected: {', '.join(self.last_detected_names)}", True

        # No fruit state
        return "No fruit detected — Show a fruit to the camera", False

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Render high-contrast bounding boxes and fruit labels onto the frame.

        Args:
            frame: BGR image frame.
            detections: List of Detection objects.

        Returns:
            Annotated image frame.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.box
            color = det.color
            conf_str = f"{det.confidence * 100:.1f}%"
            label = f"{det.class_name} | {conf_str}"

            # 1. Draw bounding box with crisp line
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # 2. Add corner accents for refined computer vision aesthetic
            corner_len = min(20, max(5, int(min(x2 - x1, y2 - y1) * 0.15)))
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, 3, cv2.LINE_AA)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, 3, cv2.LINE_AA)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, 3, cv2.LINE_AA)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, 3, cv2.LINE_AA)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, 3, cv2.LINE_AA)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, 3, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, 3, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, 3, cv2.LINE_AA)

            # 3. Label tag dimensions
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Determine label position (above box if space permits, otherwise inside top)
            padding_x = 6
            padding_y = 5
            tag_w = text_w + (padding_x * 2)
            tag_h = text_h + (padding_y * 2)

            if y1 - tag_h >= 0:
                tag_y1 = y1 - tag_h
                tag_y2 = y1
            else:
                tag_y1 = y1
                tag_y2 = y1 + tag_h

            tag_x1 = x1
            tag_x2 = min(frame.shape[1], x1 + tag_w)

            # Draw label background tag in fruit color
            cv2.rectangle(
                annotated,
                (tag_x1, tag_y1),
                (tag_x2, tag_y2),
                color,
                thickness=cv2.FILLED,
            )

            # Draw contrasting white text with dark outline for readability
            text_x = tag_x1 + padding_x
            text_y = tag_y2 - padding_y

            # Black outline for crisp text contrast
            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            # White core text
            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return annotated

    def draw_ui_overlays(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float = 0.0,
    ) -> np.ndarray:
        """Render clean, minimal top status bar and bottom shortcut hints.

        Args:
            frame: Annotated or raw image frame.
            detections: List of current detections.
            fps: Live frames-per-second measurement.

        Returns:
            Frame with integrated UI overlays.
        """
        h, w = frame.shape[:2]
        output = frame.copy()

        # ---------------------------------------------------------------------
        # Top Status Bar
        # ---------------------------------------------------------------------
        top_bar_height = 42
        top_overlay = output.copy()
        cv2.rectangle(
            top_overlay,
            (0, 0),
            (w, top_bar_height),
            UI_COLORS["banner_bg"],
            thickness=cv2.FILLED,
        )
        # 85% opacity blend
        cv2.addWeighted(top_overlay, 0.85, output, 0.15, 0, output)
        cv2.line(output, (0, top_bar_height), (w, top_bar_height), UI_COLORS["banner_border"], 1)

        # Status text computation
        status_text, is_detected = self._get_status_message(detections)
        status_color = UI_COLORS["status_detected"] if is_detected else UI_COLORS["status_no_fruit"]

        # Indicator dot
        dot_radius = 5
        dot_center = (18, top_bar_height // 2)
        cv2.circle(output, dot_center, dot_radius, status_color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)

        # Status text rendering
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            output,
            status_text,
            (32, top_bar_height // 2 + 5),
            font,
            0.52,
            status_color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # Fruit count badge
        count = len(detections)
        if count > 0:
            count_text = f"Count: {count}"
            (cw, _), _ = cv2.getTextSize(count_text, font, 0.50, 1)
            # Position to the right, before FPS
            count_x = w - cw - 120
            cv2.putText(
                output,
                count_text,
                (count_x, top_bar_height // 2 + 5),
                font,
                0.50,
                UI_COLORS["text_white"],
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        # Live FPS indicator
        if fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            (fw, _), _ = cv2.getTextSize(fps_text, font, 0.50, 1)
            fps_x = w - fw - 16
            cv2.putText(
                output,
                fps_text,
                (fps_x, top_bar_height // 2 + 5),
                font,
                0.50,
                UI_COLORS["accent_fps"],
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        # ---------------------------------------------------------------------
        # Bottom Control Bar
        # ---------------------------------------------------------------------
        bottom_bar_height = 30
        bottom_y1 = h - bottom_bar_height
        bottom_overlay = output.copy()
        cv2.rectangle(
            bottom_overlay,
            (0, bottom_y1),
            (w, h),
            UI_COLORS["banner_bg"],
            thickness=cv2.FILLED,
        )
        cv2.addWeighted(bottom_overlay, 0.85, output, 0.15, 0, output)
        cv2.line(output, (0, bottom_y1), (w, bottom_y1), UI_COLORS["banner_border"], 1)

        # Check for temporary notification (e.g. screenshot saved)
        now = time.time()
        if self.notification_text and now < self.notification_expiry:
            msg = self.notification_text
            msg_color = UI_COLORS["save_notification"]
        else:
            msg = "[Q] Quit    [ESC] Exit    [S] Save Frame"
            msg_color = UI_COLORS["accent_quit"]

        cv2.putText(
            output,
            msg,
            (16, h - 9),
            font,
            0.45,
            msg_color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        return output

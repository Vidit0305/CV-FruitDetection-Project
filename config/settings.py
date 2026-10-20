"""Configuration settings for the Real-Time Fruit Detection Model."""

from pathlib import Path
import torch

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "best.pt"
CAPTURES_DIR = PROJECT_ROOT / "captures"

# Ensure essential directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# Detection parameters
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMAGE_SIZE = 640

# Hardware execution
AUTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Webcam / Video Stream settings
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_SMOOTHING_FACTOR = 0.9  # Exponential smoothing for FPS counter

# Status message stabilization
STATUS_HOLD_SECONDS = 0.6  # Minimum duration to hold status before switching to "No fruit"

# Target fruit dictionary: Maps various raw class strings (case-insensitive)
# to standardized display names and distinctive, high-contrast BGR colors.
TARGET_FRUIT_MAPPING = {
    # Apple
    "apple": {"display": "Apple", "color": (50, 50, 225)},       # Crisp Red
    # Banana
    "banana": {"display": "Banana", "color": (30, 215, 255)},    # Banana Yellow
    # Orange variants
    "orange": {"display": "Orange", "color": (20, 140, 255)},    # Tangerine Orange
    "orange/orange fruit": {"display": "Orange", "color": (20, 140, 255)},
    "mandarin orange": {"display": "Orange", "color": (20, 140, 255)},
    "green_orange": {"display": "Orange", "color": (30, 180, 220)},
    # Mango
    "mango": {"display": "Mango", "color": (40, 190, 255)},      # Golden Mango
    # Grapes variants
    "grape": {"display": "Grapes", "color": (180, 60, 150)},     # Rich Purple
    "grapes": {"display": "Grapes", "color": (180, 60, 150)},
    # Pineapple
    "pineapple": {"display": "Pineapple", "color": (60, 210, 230)}, # Tropical Amber
    # Watermelon
    "watermelon": {"display": "Watermelon", "color": (70, 180, 50)}, # Lush Green
    # Strawberry variants
    "strawberry": {"display": "Strawberry", "color": (70, 50, 230)}, # Ruby Strawberry
    # Pear
    "pear": {"display": "Pear", "color": (70, 205, 140)},        # Pear / Lime Green
    # Lemon & Lime variants
    "lemon": {"display": "Lemon", "color": (50, 240, 255)},      # Bright Lemon Yellow
    "lime": {"display": "Lemon", "color": (60, 230, 180)},
}

# The 10 canonical fruit types required by the project
CANONICAL_FRUITS = [
    "Apple",
    "Banana",
    "Orange",
    "Mango",
    "Grapes",
    "Pineapple",
    "Watermelon",
    "Strawberry",
    "Pear",
    "Lemon",
]

# UI Visual Style Tokens
UI_COLORS = {
    "background_dark": (20, 20, 24),
    "banner_bg": (15, 15, 20),
    "banner_border": (60, 60, 70),
    "text_white": (255, 255, 255),
    "text_muted": (180, 180, 190),
    "status_no_fruit": (80, 150, 240),   # Warm Amber / Orange
    "status_detected": (90, 220, 110),   # Vibrant Emerald Green
    "accent_fps": (220, 180, 50),        # Cyan / Blue
    "accent_quit": (160, 160, 170),      # Subtle grey
    "save_notification": (50, 215, 255), # Bright Cyan
}

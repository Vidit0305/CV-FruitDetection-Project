"""Model Training and Fine-Tuning Module for Fruit Detection using YOLO.

Provides a reproducible pipeline to train or fine-tune an Ultralytics YOLO
object detector on a custom or public annotated fruit dataset.
"""

import argparse
from pathlib import Path
import shutil
import sys
import torch
from ultralytics import YOLO

from config.settings import DEFAULT_MODEL_PATH, DEFAULT_IMAGE_SIZE


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(
        description="Train or fine-tune a YOLO model on a fruit detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to dataset configuration YAML file",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO checkpoint to start transfer learning from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16 if torch.cuda.is_available() else 4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Input image resolution for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run training on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4 if torch.cuda.is_available() else 2,
        help="Dataloader worker processes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Destination path for best trained weights",
    )
    return parser.parse_args()


def train_fruit_model(args: argparse.Namespace) -> Path:
    """Execute YOLO training and save the resulting checkpoint.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path to the saved best weights.
    """
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\n[ERROR] Dataset YAML file not found: {data_path.resolve()}")
        print("Please prepare your dataset and data.yaml configuration first.")
        print("Refer to 'dataset/README.md' for complete dataset setup instructions.\n")
        sys.exit(1)

    print("=" * 60)
    print("Fruit Detection Model Training / Fine-Tuning")
    print("=" * 60)
    print(f"[*] Dataset Config: {data_path.resolve()}")
    print(f"[*] Base Checkpoint: {args.base_model}")
    print(f"[*] Epochs:         {args.epochs}")
    print(f"[*] Batch Size:     {args.batch}")
    print(f"[*] Image Size:     {args.imgsz}")
    print(f"[*] Compute Device: {args.device.upper()}")
    print("=" * 60)

    # 1. Initialize YOLO model with pretrained weights for transfer learning
    print(f"\n[INFO] Loading base model: {args.base_model}...")
    model = YOLO(args.base_model)

    # 2. Run training
    print("[INFO] Starting training pipeline...")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project="runs/train",
        name="fruit_detector",
        exist_ok=True,
        plots=True,
        save=True,
    )

    # 3. Locate best.pt from the training run
    best_weights_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_weights_path.exists():
        # Fallback search in runs directory
        candidates = list(Path("runs/train").glob("**/weights/best.pt"))
        if candidates:
            best_weights_path = candidates[-1]

    if best_weights_path.exists():
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights_path, destination)
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"[*] Trained Weights:   {best_weights_path}")
        print(f"[*] Deployed Checkpoint: {destination.resolve()}")
        print("\nYou can now launch real-time detection with:")
        print(f"  python main.py --model {destination}")
        print("=" * 60)
        return destination
    else:
        print(f"\n[WARNING] Could not locate 'best.pt' under {best_weights_path}")
        return Path(args.output)


if __name__ == "__main__":
    arguments = parse_arguments()
    train_fruit_model(arguments)

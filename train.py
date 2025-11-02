# LumenTact/train.py
import argparse
import os
import yaml
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--model",
        default="yolov8n.pt",
        help="path to base model (e.g., yolov8n.pt)",
    )
    ap.add_argument("-d", "--data", required=True, help="path to data.yaml file")
    ap.add_argument(
        "-e", "--epochs", type=int, default=50, help="number of epochs to train for"
    )
    ap.add_argument(
        "-b", "--batch", type=int, default=16, help="batch size for training"
    )
    ap.add_argument(
        "-n",
        "--name",
        default="lum tact_run",
        help="name for the training run (results in runs/detect/NAME)",
    )
    args = ap.parse_args()

    # Verify data.yaml file exists
    if not os.path.exists(args.data):
        print(f"[ERROR] Data config file not found at: {args.data}")
        return

    # Load the model
    print(f"[INFO] Loading base model '{args.model}'...")
    model = YOLO(args.model)

    print("[INFO] Starting training...")
    # Start training
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        name=args.name,
        exist_ok=True,  # allow overwriting previous runs
    )

    print(f"[INFO] Training complete. Results saved to: {results.save_dir}")
    print(f"[INFO] Best model weights saved to: {results.best}")


if __name__ == "__main__":
    main()

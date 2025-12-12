"""
Run validation on a trained YOLO checkpoint and emit a small JSON summary.

Example:
  ./venv/bin/python tools/validate_model.py \
    --weights runs/detect/sanpo_yolo/weights/best.pt \
    --data dataset_yolo_train/data.yaml \
    --imgsz 1280 --batch 16
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Validate a YOLO model and write summary JSON.")
    ap.add_argument("--weights", required=True, help="Path to weights (pt/onnx/engine)")
    ap.add_argument("--data", required=True, help="data.yaml path")
    ap.add_argument("--imgsz", type=int, default=1280, help="Image size")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--device", default="0", help="Device id or 'cpu'")
    ap.add_argument("--name", default="val_report", help="Run name under runs/detect/")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"data.yaml not found: {data}")

    model = YOLO(weights)
    print(f"Validating {weights} with {data} ...")
    metrics = model.val(
        data=str(data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        save_json=False,
    )

    out_dir = Path(metrics.save_dir)
    summary = {
        "weights": str(weights),
        "data": str(data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "run_dir": str(out_dir),
        "metrics": {
            "map50": getattr(metrics, "box", metrics).map50 if hasattr(metrics, "box") else None,
            "map50_95": getattr(metrics, "box", metrics).map if hasattr(metrics, "box") else None,
            "precision": getattr(getattr(metrics, "box", metrics), "mp", None),
            "recall": getattr(getattr(metrics, "box", metrics), "mr", None),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Validation complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()

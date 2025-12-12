import argparse
import pathlib
import time
from typing import List

from ultralytics import YOLO


def list_images(source: pathlib.Path, max_items: int) -> List[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in sorted(source.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if max_items > 0:
        files = files[:max_items]
    return files


def main():
    ap = argparse.ArgumentParser(description="Benchmark YOLO inference throughput on a folder of images.")
    ap.add_argument("--weights", required=True, help="Path to YOLO weights (pt/onnx/engine)")
    ap.add_argument("--source", default="dataset_yolo_train/images/val", help="Folder of images to run inference on")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    ap.add_argument("--device", default="0", help="Device id or 'cpu'")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--max", type=int, default=0, help="Limit number of images for benchmarking")
    ap.add_argument("--repeats", type=int, default=1, help="Number of timing repetitions to average")
    args = ap.parse_args()

    src = pathlib.Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    files = list_images(src, args.max)
    if not files:
        raise SystemExit(f"No images found in {src}")

    model = YOLO(args.weights)

    # warmup on a single image
    model.predict(files[0], imgsz=args.imgsz, device=args.device, conf=args.conf, verbose=False)

    total_times = []
    for r in range(args.repeats):
        start = time.perf_counter()
        model.predict(
          source=files,
          imgsz=args.imgsz,
          device=args.device,
          conf=args.conf,
          verbose=False,
          save=False,
        )
        elapsed = time.perf_counter() - start
        total_times.append(elapsed)
        print(f"Repeat {r+1}/{args.repeats}: {len(files)} images in {elapsed:.3f}s ({len(files)/elapsed:.2f} FPS)")

    avg = sum(total_times) / len(total_times)
    fps = len(files) / avg
    per_image_ms = (avg / len(files)) * 1000
    print(f"Avg over {args.repeats} runs: {fps:.2f} FPS | {per_image_ms:.2f} ms/image on device {args.device}")


if __name__ == "__main__":
    main()

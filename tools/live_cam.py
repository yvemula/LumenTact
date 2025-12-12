"""
Live camera tester for YOLO models.

Example:
  ./venv/bin/python tools/live_cam.py \
    --weights runs/detect/sanpo_yolo/weights/best.pt \
    --device 0 --imgsz 1280 --conf 0.25 --camera 0

Press 'q' or ESC to quit.
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser(description="Run live camera inference with YOLO.")
    ap.add_argument("--weights", default="runs/detect/sanpo_yolo/weights/best.pt", help="Path to YOLO weights")
    ap.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--device", default="0", help="Device id (e.g., '0' or 'cpu')")
    ap.add_argument("--show-conf", action="store_true", help="Display confidence scores in labels")
    ap.add_argument("--window", default="YOLO Live", help="Window name")
    return ap.parse_args()


def color_for_class(idx: int):
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
    ]
    return palette[idx % len(palette)]


def main():
    args = parse_args()
    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    print(f"Loading model from {args.weights} on device {args.device} ...")
    model = YOLO(args.weights)
    names = model.names

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    prev_time = time.time()
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; exiting.")
                break

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                verbose=False,
            )
            res = results[0]
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0]) if box.cls is not None else -1
                score = float(box.conf[0]) if box.conf is not None else 0.0
                label = names.get(cls, str(cls)) if isinstance(names, dict) else (names[cls] if cls >= 0 and cls < len(names) else str(cls))
                if args.show_conf:
                    label = f"{label} {score:.2f}"
                color = color_for_class(cls if cls >= 0 else 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1) + 4, int(y1) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(args.window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or q
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

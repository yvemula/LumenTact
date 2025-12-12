"""
Simple Gradio web demo for YOLO inference.

Example:
  ./venv/bin/python tools/web_demo.py --weights runs/detect/sanpo_yolo/weights/best.pt --imgsz 1280 --conf 0.25
"""
import argparse
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser(description="Launch a Gradio web UI for YOLO inference.")
    ap.add_argument("--weights", default="runs/detect/sanpo_yolo/weights/best.pt", help="Path to YOLO weights")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--device", default="0", help="Device id or 'cpu'")
    ap.add_argument("--share", action="store_true", help="Enable Gradio share link")
    ap.add_argument("--title", default="YOLO Demo", help="Gradio title")
    ap.add_argument("--description", default="Upload an image to see YOLO detections.", help="Gradio description")
    return ap.parse_args()


def run():
    args = parse_args()
    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    model = YOLO(args.weights)

    def infer(image: Any):
        if image is None:
            return None
        # image arrives as numpy RGB from Gradio
        res = model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        bgr = res.plot()  # returns numpy BGR
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    demo = gr.Interface(
        fn=infer,
        inputs=gr.Image(type="numpy", label="Input"),
        outputs=gr.Image(type="numpy", label="Detections"),
        title=args.title,
        description=args.description,
        allow_flagging="never",
    )

    demo.launch(share=args.share)


if __name__ == "__main__":
    run()

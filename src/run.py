from __future__ import annotations
import argparse
from pipeline import run

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Image, video, or folder path")
    ap.add_argument("--out_dir", default=None, help="Output folder for frames")
    ap.add_argument("--yolo_weights", default="yolo12n.pt", help="YOLOv12 detect weights")
    ap.add_argument("--yolo_seg_weights", default=None, help="(Optional) YOLOv12 seg weights if you have them")
    ap.add_argument("--show", action="store_true", help="Show live window (ESC to quit)")
    return ap.parse_args()

if __name__ == "__main__":
    a = parse_args()
    run(a.source, a.out_dir, a.yolo_weights, a.yolo_seg_weights, a.show)

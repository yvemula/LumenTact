"""
track.py
Lightweight wrapper for OC-SORT or ByteTrack to smooth YOLO detections.
You can toggle trackers or fall back to raw detections if none installed.
"""

import numpy as np
from typing import List, Dict

# Optional imports (fail-safe if unavailable)
try:
    from ocsort import OCSort
except ImportError:
    OCSort = None

try:
    from bytetracker import BYTETracker
except ImportError:
    BYTETracker = None


class TrackerWrapper:
    def __init__(self, method: str = "ocsort", fps: int = 30):
        self.method = method.lower()
        self.tracker = None

        if self.method == "ocsort" and OCSort:
            self.tracker = OCSort(det_thresh=0.3, max_age=30, min_hits=3)
        elif self.method == "bytetrack" and BYTETracker:
            self.tracker = BYTETracker(frame_rate=fps)
        else:
            print(f"[WARN] Tracker {self.method} unavailable — using passthrough mode")

    def update(self, dets: List[Dict], frame_shape):
        """
        dets: list of {cls, conf, bbox(cx,cy,w,h)}
        frame_shape: (H, W)
        returns: list of same dicts + 'id'
        """
        H, W = frame_shape[:2]
        if not self.tracker:
            # fallback: just enumerate detections
            return [{**d, "id": i} for i, d in enumerate(dets)]

        # Convert to (x1,y1,x2,y2,conf,cls)
        det_array = []
        for d in dets:
            cx, cy, bw, bh = d["bbox"]
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            det_array.append([x1, y1, x2, y2, d["conf"], 0])  # single-class for simplicity

        dets_np = np.array(det_array, dtype=np.float32)
        tracks = self.tracker.update(dets_np, frame_shape)

        out = []
        for tr in tracks:
            x1, y1, x2, y2, track_id, conf, cls_id = tr[:7]
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            out.append({
                "id": int(track_id),
                "cls": "object",
                "conf": float(conf),
                "bbox": (cx, cy, bw, bh),
            })
        return out

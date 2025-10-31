from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional
from models import MonoDepth, SanpoTraversableSeg, overlay_mask
from decision import DecisionEngine

def _iter_frames(src: str) -> Iterator[np.ndarray]:
    p = Path(src)
    if p.is_dir():
        for f in sorted(p.glob("*.*")):
            img = cv2.imread(str(f))
            if img is not None:
                yield img
    elif p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        yield img
    else:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {p}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                yield frame
        finally:
            cap.release()

def run(source: str, out_dir: Optional[str] = None, yolo_weights: str = "yolo12n.pt",
        yolo_seg_weights: Optional[str] = None, show: bool = False):
    md = MonoDepth(model_type="DPT_Large")
    seg = SanpoTraversableSeg(yolo_weights=yolo_weights, yolo_seg_weights=yolo_seg_weights)

    out_path = Path(out_dir) if out_dir else None
    if out_path: out_path.mkdir(parents=True, exist_ok=True)

    for i, frame_bgr in enumerate(_iter_frames(source)):
        depth01 = md(frame_bgr)
        trav01  = seg(frame_bgr, depth01)

        # Decision
        engine = getattr(run, "_engine", None)
        if engine is None:
            engine = DecisionEngine()        # create once and cache on function
            run._engine = engine

        dec = engine.decide(trav01, depth01)  # NOTE: (mask, depth)

        # Debug overlays
        depth_vis = (depth01 * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        trav_color = overlay_mask(frame_bgr, trav01, alpha=0.45)
        dec_vis = engine.draw_debug(trav_color, trav01, dec)

        top = np.hstack([frame_bgr, dec_vis])
        bot = np.hstack([
            depth_vis,
            cv2.cvtColor((trav01 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ])
        panel = np.vstack([top, bot])

        # Print/log the command (and optionally speak it; see TTS below)
        print(f"[{i:06d}] CMD={dec.cmd:>7}  steer={dec.steer:+.2f}  free={dec.traversable_frac:.3f}  reason={dec.reason}")

        if out_path:
            cv2.imwrite(str(out_path / f"frame_{i:06d}.jpg"), panel)

        if show:
            cv2.imshow("LumenTact — [orig | traversable] / [depth | mask]", panel)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    if show:
        cv2.destroyAllWindows()

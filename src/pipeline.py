from __future__ import annotations
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2

from detectors import GuideDogDetector, SanpoTraversableSeg, MonoDepth
from track import TrackerWrapper
from decision import decide_action
from haptics import play
from viz import draw_dets, put_hud, overlay_traversable

@dataclass
class RunConfig:
    video_path: str
    save_out: Optional[str] = None
    show: bool = True
    tracker: str = "ocsort"        # "ocsort" | "bytetrack" | "none"
    guidedog_weights: Optional[str] = "models/guidedog_det.pt"  # optional custom
    conf: float = 0.35
    imgsz: int = 640
    sanpo_onnx: str = "models/sanpo_traversable.onnx"
    use_depth: bool = True
    log_csv: Optional[str] = "runs/last_run.csv"
    haptic_cooldown_s: float = 0.5
    start_paused: bool = False

def _ensure_parent(path: Optional[str]):
    if not path:
        return
    p = Path(path)
    (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)

def run(
    video_path: str,
    save_out: Optional[str] = None,
    show: bool = True,
    tracker_method: str = "ocsort",
    guidedog_weights: Optional[str] = "models/guidedog_det.pt",
    conf: float = 0.35,
    imgsz: int = 640,
    sanpo_onnx: str = "models/sanpo_traversable.onnx",
    use_depth: bool = True,
    log_csv: Optional[str] = "runs/last_run.csv",
    haptic_cooldown_s: float = 0.5,
    start_paused: bool = False,
) -> Tuple[int, float]:

    cfg = RunConfig(
        video_path, save_out, show, tracker_method, guidedog_weights,
        conf, imgsz, sanpo_onnx, use_depth, log_csv, haptic_cooldown_s, start_paused
    )

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {cfg.video_path}")

    writer = None
    if cfg.save_out: _ensure_parent(cfg.save_out)
    _ensure_parent(cfg.log_csv)
    log_fh = open(cfg.log_csv, "w", newline="") if cfg.log_csv else None
    logger = csv.writer(log_fh) if log_fh else None
    if logger:
        logger.writerow(["t","frame_idx","fps","action","strength","num_dets","num_tracked"])

    det = GuideDogDetector(cfg.guidedog_weights, cfg.conf, cfg.imgsz)
    seg = SanpoTraversableSeg(cfg.sanpo_onnx)
    depth = MonoDepth() if cfg.use_depth else None
    tracker = None if cfg.tracker == "none" else TrackerWrapper(method=cfg.tracker, fps=30)

    fps_ema, alpha = None, 0.2
    total_frames = 0
    t0_all = time.monotonic()
    next_buzz_at = 0.0
    paused = cfg.start_paused

    try:
        while True:
            if paused and cfg.show:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord(" "), ord("p")): paused = False
                elif key in (27, ord("q")): break
                continue

            ok, frame = cap.read()
            if not ok: break
            t0 = time.monotonic()

            # Perception
            raw_dets = det(frame)
            tracked = tracker.update(raw_dets, frame.shape) if tracker else raw_dets
            trav_mask = seg(frame)
            depth_map = depth(frame) if depth else None

            # Decision
            action, strength = decide_action(
                tracked, fps_ema or 15.0, frame.shape, trav_mask=trav_mask, depth_map=depth_map
            )

            # Haptics (rate-limited)
            now = time.monotonic()
            if now >= next_buzz_at:
                play(action, strength)
                next_buzz_at = now + cfg.haptic_cooldown_s

            # Viz
            canvas = overlay_traversable(frame.copy(), trav_mask)
            canvas = draw_dets(canvas, tracked)
            canvas = put_hud(canvas, action, fps_ema)

            if cfg.show:
                cv2.imshow("WearableNav (GuideDog + SANPO)", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")): break
                if key in (ord(" "), ord("p")): paused = not paused

            if cfg.save_out:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(cfg.save_out, fourcc, 20.0, (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)

            # Timing / logs
            dt = max(1e-4, time.monotonic() - t0)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema is None else (1 - alpha) * fps_ema + alpha * inst_fps

            total_frames += 1
            if logger:
                logger.writerow([
                    f"{time.time():.3f}", total_frames, f"{fps_ema:.2f}" if fps_ema else "",
                    action, f"{strength:.2f}", len(raw_dets), len(tracked)
                ])

    finally:
        cap.release()
        if writer: writer.release()
        if cfg.show:
            try: cv2.destroyAllWindows()
            except: pass
        if log_fh: log_fh.close()

    avg_fps = total_frames / max(1e-6, (time.monotonic() - t0_all))
    return total_frames, avg_fps

# Back-compat
def run_configured(video_path: str, save_out: Optional[str] = None, show: bool = True):
    return run(video_path, save_out=save_out, show=show)

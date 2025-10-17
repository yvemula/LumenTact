# src/pipeline.py
"""
End-to-end software MVP pipeline:
video -> YOLOv12 det -> tracker -> decision -> haptics(emulated) -> viz/log

Requires:
- detectors.TinyDetector (YOLOv12)
- track.TrackerWrapper (OC-SORT / ByteTrack)
- decision.decide_action
- haptics.play
- viz.draw_dets, viz.put_hud
"""

from __future__ import annotations
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2

from detectors import TinyDetector
from track import TrackerWrapper
from decision import decide_action
from haptics import play
from viz import draw_dets, put_hud


@dataclass
class RunConfig:
    video_path: str
    save_out: Optional[str] = None
    show: bool = True
    tracker: str = "ocsort"        # "ocsort" | "bytetrack" | "none"
    model_name: str = "yolo12n.pt" # YOLOv12 nano; use yolo11n.pt if artifacts lag
    conf: float = 0.35
    imgsz: int = 640
    log_csv: Optional[str] = "runs/last_run.csv"
    target_fps_overlay: float = 30.0
    haptic_cooldown_s: float = 0.5 # limit how often we play a (blocking) pattern
    start_paused: bool = False     # useful when presenting; press SPACE to start


def _ensure_parent(path: Optional[str]):
    if not path:
        return
    p = Path(path)
    if p.suffix:  # file
        p.parent.mkdir(parents=True, exist_ok=True)
    else:         # directory
        p.mkdir(parents=True, exist_ok=True)


def run(
    video_path: str,
    save_out: Optional[str] = None,
    show: bool = True,
    tracker_method: str = "ocsort",
    model_name: str = "yolo12n.pt",
    conf: float = 0.35,
    imgsz: int = 640,
    log_csv: Optional[str] = "runs/last_run.csv",
    haptic_cooldown_s: float = 0.5,
    start_paused: bool = False,
) -> Tuple[int, float]:
    """
    Returns:
        total_frames, avg_fps
    """
    cfg = RunConfig(
        video_path=video_path,
        save_out=save_out,
        show=show,
        tracker=tracker_method,
        model_name=model_name,
        conf=conf,
        imgsz=imgsz,
        log_csv=log_csv,
        haptic_cooldown_s=haptic_cooldown_s,
        start_paused=start_paused,
    )

    # IO & logging setup
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {cfg.video_path}")

    writer = None
    if cfg.save_out:
        _ensure_parent(cfg.save_out)

    _ensure_parent(cfg.log_csv)
    log_fh = open(cfg.log_csv, "w", newline="") if cfg.log_csv else None
    logger = csv.writer(log_fh) if log_fh else None
    if logger:
        logger.writerow([
            "t", "frame_idx", "fps", "action", "strength",
            "num_dets", "num_tracked"
        ])

    # Models
    detector = TinyDetector(model_name=cfg.model_name, conf=cfg.conf, imgsz=cfg.imgsz)
    tracker = None if cfg.tracker == "none" else TrackerWrapper(method=cfg.tracker, fps=int(cfg.target_fps_overlay))

    # Timing & control
    fps_ema, alpha = None, 0.2
    total_frames = 0
    t_start = time.monotonic()
    next_buzz_at = 0.0
    paused = cfg.start_paused

    # Main loop
    try:
        while True:
            if paused and cfg.show:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord(" "), ord("p")):
                    paused = False
                elif key in (27, ord("q")):
                    break
                continue

            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.monotonic()

            # --- Perception ---
            raw_dets = detector(frame)  # list of {cls, conf, bbox(cx,cy,w,h)}
            tracked = raw_dets
            if tracker:
                tracked = tracker.update(raw_dets, frame.shape)

            # --- Decision ---
            action, strength = decide_action(tracked, fps_ema or 15.0)

            # --- Haptics (rate-limited to avoid blocking loop too often) ---
            now = time.monotonic()
            if now >= next_buzz_at:
                play(action, strength)
                next_buzz_at = now + cfg.haptic_cooldown_s

            # --- Viz & Output ---
            out = draw_dets(frame.copy(), tracked)
            out = put_hud(out, action, fps_ema)

            if cfg.show:
                cv2.imshow("WearableNav (software MVP)", out)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # ESC or q
                    break
                if key in (ord(" "), ord("p")):
                    paused = not paused

            if cfg.save_out:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(cfg.save_out, fourcc, 20.0, (out.shape[1], out.shape[0]))
                writer.write(out)

            # --- Timing & logging ---
            dt = max(1e-4, time.monotonic() - t0)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema is None else (1 - alpha) * fps_ema + alpha * inst_fps

            total_frames += 1
            if logger:
                logger.writerow([
                    f"{time.time():.3f}",
                    total_frames,
                    f"{fps_ema:.2f}" if fps_ema else "",
                    action,
                    f"{strength:.2f}",
                    len(raw_dets),
                    len(tracked) if tracked is not None else 0,
                ])

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        if cfg.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if log_fh:
            log_fh.close()

    avg_fps = total_frames / max(1e-6, (time.monotonic() - t_start))
    return total_frames, avg_fps


# Convenience wrapper to keep old call signature (used by src/run.py)
def run_configured(video_path: str, save_out: Optional[str] = None, show: bool = True):
    return run(video_path, save_out=save_out, show=show)

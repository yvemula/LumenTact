from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional
from models import MonoDepth, SanpoTraversableSeg, overlay_mask
from decision import DecisionEngine, Decision

try:
    from haptics import play as play_haptics
except Exception:  # pragma: no cover
    play_haptics = None


class SafetyMonitor:
    """Stops the system after consecutive degraded frames (no depth or no free space)."""

    def __init__(self, max_bad_frames: int = 5, min_trav_frac: float = 0.02):
        self.max_bad_frames = max_bad_frames
        self.min_trav_frac = min_trav_frac
        self.bad_frame_count = 0

    def guard(self, trav_mask01: Optional[np.ndarray], decision: Decision, depth_is_fallback: bool) -> Decision:
        trav_frac = float(trav_mask01.mean()) if trav_mask01 is not None else 0.0
        issues = []
        if trav_mask01 is None or trav_frac < self.min_trav_frac:
            issues.append(f"low traversable ({trav_frac:.3f})")
        if depth_is_fallback:
            issues.append("depth unavailable")

        if not issues:
            self.bad_frame_count = 0
            return decision

        self.bad_frame_count += 1
        if self.bad_frame_count < self.max_bad_frames:
            return decision

        reason = " / ".join(issues)
        return Decision(
            cmd="STOP",
            steer=0.0,
            reason=f"Safety override: {reason}",
            traversable_frac=trav_frac,
            target_x=None,
            confidence=0.0,
        )


class HapticBridge:
    """Translates navigation decisions into belt patterns with confidence-weighted strength."""

    ACTION_MAP = {
        "STOP": "STOP",
        "FORWARD": "CAUTION",
        "LEFT": "VEER_LEFT",
        "RIGHT": "VEER_RIGHT",
    }

    def __init__(self):
        self.last_cmd: Optional[str] = None

    def notify(self, decision: Decision):
        if play_haptics is None or decision is None:
            return
        should_emit = (decision.cmd != self.last_cmd) or (decision.confidence < 0.5)
        if not should_emit:
            return
        action = self.ACTION_MAP.get(decision.cmd, "CAUTION")
        strength = max(0.3, min(1.0, decision.confidence))
        play_haptics(action, strength)
        self.last_cmd = decision.cmd

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
    safety = getattr(run, "_safety", None)
    if safety is None:
        safety = SafetyMonitor()
        run._safety = safety
    haptics_bridge = getattr(run, "_haptics", None)
    if haptics_bridge is None:
        haptics_bridge = HapticBridge()
        run._haptics = haptics_bridge

    out_path = Path(out_dir) if out_dir else None
    if out_path: out_path.mkdir(parents=True, exist_ok=True)

    for i, frame_bgr in enumerate(_iter_frames(source)):
        depth01 = md(frame_bgr)
        depth_is_fallback = depth01 is None
        if depth01 is None:
            depth01 = np.ones(frame_bgr.shape[:2], dtype=np.float32)

        trav_raw = seg(frame_bgr, depth01)
        if trav_raw is None:
            trav01 = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        else:
            trav01 = (trav_raw > 0).astype(np.uint8)

        # Decision
        engine = getattr(run, "_engine", None)
        if engine is None:
            engine = DecisionEngine()        # create once and cache on function
            run._engine = engine

        dec = engine.decide(trav01, depth01)  # NOTE: (mask, depth)
        dec = safety.guard(trav01, dec, depth_is_fallback)
        haptics_bridge.notify(dec)

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
        print(
            f"[{i:06d}] CMD={dec.cmd:>7}  steer={dec.steer:+.2f}  "
            f"free={dec.traversable_frac:.3f}  conf={dec.confidence:.2f}  reason={dec.reason}"
        )

        if out_path:
            cv2.imwrite(str(out_path / f"frame_{i:06d}.jpg"), panel)

        if show:
            cv2.imshow("LumenTact — [orig | traversable] / [depth | mask]", panel)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    if show:
        cv2.destroyAllWindows()

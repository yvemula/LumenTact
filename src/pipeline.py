from __future__ import annotations
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional
from models import MonoDepth, SanpoTraversableSeg, overlay_mask
from decision import DecisionEngine, Decision, DecisionSmoother
from perception import SemanticAnalyzer, OpticalFlowTracker
from telemetry import TelemetryClient

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


class LatencyWatchdog:
    """Monitors per-frame latency and triggers a stop if it blows past the budget."""

    def __init__(self, budget_ms: float = 180.0, max_breaches: int = 3):
        self.budget_ms = budget_ms
        self.max_breaches = max_breaches
        self.breach_count = 0

    def assess(self, latency_ms: float) -> Optional[str]:
        if latency_ms <= self.budget_ms:
            self.breach_count = 0
            return None
        self.breach_count += 1
        if self.breach_count >= self.max_breaches:
            self.breach_count = 0
            return f"Latency watchdog triggered ({latency_ms:.1f}ms > {self.budget_ms:.1f}ms)"
        return None

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

def run(
    source: str,
    out_dir: Optional[str] = None,
    yolo_weights: str = "yolo12n.pt",
    yolo_seg_weights: Optional[str] = None,
    show: bool = False,
    display_scale: float = 1.0,
):
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
    smoother = getattr(run, "_smoother", None)
    if smoother is None:
        smoother = DecisionSmoother()
        run._smoother = smoother
    smoother.reset()
    semantic = getattr(run, "_semantic", None)
    if semantic is None:
        semantic = SemanticAnalyzer()
        run._semantic = semantic
    flow_tracker = getattr(run, "_flow_tracker", None)
    if flow_tracker is None:
        flow_tracker = OpticalFlowTracker()
        run._flow_tracker = flow_tracker
    flow_tracker.reset()
    latency_watchdog = getattr(run, "_watchdog", None)
    if latency_watchdog is None:
        latency_watchdog = LatencyWatchdog()
        run._watchdog = latency_watchdog
    latency_watchdog.breach_count = 0
    telemetry = getattr(run, "_telemetry", None)
    if telemetry is None:
        telemetry = TelemetryClient()
        run._telemetry = telemetry

    out_path = Path(out_dir) if out_dir else None
    if out_path: out_path.mkdir(parents=True, exist_ok=True)

    for i, frame_bgr in enumerate(_iter_frames(source)):
        frame_start = time.perf_counter()
        depth01 = md(frame_bgr)
        depth_is_fallback = depth01 is None
        if depth01 is None:
            depth01 = np.ones(frame_bgr.shape[:2], dtype=np.float32)

        trav_raw = seg(frame_bgr, depth01)
        if trav_raw is None:
            trav01 = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        else:
            trav01 = (trav_raw > 0).astype(np.uint8)

        semantic_info = semantic.analyze(depth01, trav01)
        flow_info = flow_tracker.update(frame_bgr, trav01)
        if flow_info.moving_mask is not None:
            trav01 = np.where(flow_info.moving_mask > 0, 0, trav01)

        # Decision
        engine = getattr(run, "_engine", None)
        if engine is None:
            engine = DecisionEngine()        # create once and cache on function
            run._engine = engine

        dec = engine.decide(trav01, depth01, semantic_info, flow_info.moving_on_path_frac)
        dec = smoother.smooth(dec)
        dec = safety.guard(trav01, dec, depth_is_fallback)

        latency_ms = (time.perf_counter() - frame_start) * 1000.0
        dec.latency_ms = latency_ms
        watchdog_reason = latency_watchdog.assess(latency_ms)
        if telemetry.enabled:
            telemetry.record_latency(
                latency_ms=latency_ms,
                budget_ms=latency_watchdog.budget_ms,
                breached=watchdog_reason is not None,
                frame_index=i,
            )
        if watchdog_reason and dec.cmd != "STOP":
            dec = Decision(
                cmd="STOP",
                steer=0.0,
                reason=watchdog_reason,
                traversable_frac=dec.traversable_frac,
                target_x=dec.target_x,
                confidence=0.0,
                semantics=dec.semantics,
                moving_obstacle_frac=dec.moving_obstacle_frac,
                latency_ms=latency_ms,
            )

        haptics_bridge.notify(dec)

        # Debug overlays
        depth_vis = (depth01 * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        trav_color = overlay_mask(frame_bgr, trav01, alpha=0.45)
        dec_vis = engine.draw_debug(trav_color, trav01, dec)
        if flow_info.moving_mask is not None and np.any(flow_info.moving_mask):
            dec_vis = overlay_mask(dec_vis, flow_info.moving_mask, alpha=0.6, color=(0, 0, 255))

        top = np.hstack([frame_bgr, dec_vis])
        bot = np.hstack([
            depth_vis,
            cv2.cvtColor((trav01 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ])
        panel = np.vstack([top, bot])

        # Print/log the command (and optionally speak it; see TTS below)
        semantic_tags = []
        if semantic_info.has_stairs:
            semantic_tags.append("stairs")
        if semantic_info.has_ramp:
            semantic_tags.append("ramp")
        motion_pct = flow_info.moving_on_path_frac * 100.0
        print(
            f"[{i:06d}] CMD={dec.cmd:>7}  steer={dec.steer:+.2f}  "
            f"free={dec.traversable_frac:.3f}  conf={dec.confidence:.2f}  "
            f"motion={motion_pct:.2f}%  latency={dec.latency_ms:.1f}ms  "
            f"sem={'/'.join(semantic_tags) if semantic_tags else 'none'}  reason={dec.reason}"
        )

        if out_path:
            cv2.imwrite(str(out_path / f"frame_{i:06d}.jpg"), panel)

        if show:
            vis = panel
            scale = 1.0
            if display_scale is not None:
                try:
                    scale = float(display_scale)
                except (TypeError, ValueError):
                    scale = 1.0
                if scale <= 0:
                    scale = 1.0
                if scale != 1.0:
                    vis = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow("LumenTact [orig | traversable] / [depth | mask]", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    if show:
        cv2.destroyAllWindows()

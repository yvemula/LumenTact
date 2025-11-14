from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from decision import Decision, DecisionEngine, DecisionSmoother
from perception import OpticalFlowTracker, SemanticAnalyzer


def make_depth_ramp(h: int = 96, w: int = 96, slope: float = 1.0) -> np.ndarray:
    base = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1)
    depth = np.clip(base * slope + 0.1, 0.0, 1.0)
    return np.tile(depth, (1, w))


def make_depth_stairs(h: int = 96, w: int = 96, steps: int = 6) -> np.ndarray:
    y = np.linspace(0, 1, h, dtype=np.float32)
    stair = np.floor(y * steps) / steps
    return np.tile(stair.reshape(h, 1), (1, w))


def make_traversable_mask(h: int = 96, w: int = 96) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(h * 0.35) :, int(w * 0.15) : int(w * 0.85)] = 1
    return mask


def make_scene(lighting: str, weather: str, size: tuple[int, int] = (96, 96)) -> np.ndarray:
    h, w = size
    base_levels = {
        "night": 30,
        "dusk": 90,
        "day": 160,
        "noon": 210,
    }
    frame = np.full((h, w, 3), base_levels.get(lighting, 120), dtype=np.uint8)
    if weather == "rain":
        for x in range(0, w, 8):
            cv2.line(frame, (x, 0), (x, h - 1), (255, 255, 255), 1)
    elif weather == "fog":
        noise = np.random.default_rng(0).normal(0, 18, size=(h, w, 3)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif weather == "snow":
        rng = np.random.default_rng(1)
        for _ in range(80):
            y = int(rng.integers(0, h))
            x = int(rng.integers(0, w))
            frame[y, x] = 255
    return frame


def test_semantic_analyzer_detects_ramp():
    depth = make_depth_ramp()
    mask = make_traversable_mask()
    analyzer = SemanticAnalyzer(stairs_grad_threshold=0.2, ramp_grad_threshold=0.02)
    info = analyzer.analyze(depth, mask)
    assert info.has_ramp
    assert not info.has_stairs


def test_semantic_analyzer_detects_stairs():
    depth = make_depth_stairs()
    mask = make_traversable_mask()
    analyzer = SemanticAnalyzer(stairs_grad_threshold=0.05, ramp_grad_threshold=0.2)
    info = analyzer.analyze(depth, mask)
    assert info.has_stairs


def test_optical_flow_tracker_detects_motion_on_path():
    tracker = OpticalFlowTracker(mag_threshold=0.2, smooth_kernel=3, decay=0.0)
    frame1 = make_scene("day", "clear")
    frame2 = frame1.copy()
    cv2.rectangle(frame1, (20, 30), (40, 50), (0, 0, 0), -1)
    cv2.rectangle(frame2, (25, 35), (45, 55), (0, 0, 0), -1)
    mask = np.ones(frame1.shape[:2], dtype=np.uint8)
    tracker.update(frame1, mask)  # prime tracker
    info = tracker.update(frame2, mask)
    assert info.moving_mask is not None
    assert info.moving_on_path_frac > 0.0


def test_decision_smoother_reduces_oscillation():
    smoother = DecisionSmoother(steer_alpha=0.5, hold_frames=2)
    first = Decision(cmd="LEFT", steer=-0.8, reason="test", traversable_frac=0.3, target_x=0.2)
    stabilised = smoother.smooth(first)
    assert stabilised.cmd == "LEFT"

    second = Decision(cmd="RIGHT", steer=0.8, reason="sudden flip", traversable_frac=0.3, target_x=0.8)
    smoothed = smoother.smooth(second)
    assert smoothed.cmd == "LEFT"  # held for stability


def test_synthetic_lighting_weather_variants_keep_engine_operational():
    analyzer = SemanticAnalyzer()
    engine = DecisionEngine()
    mask = make_traversable_mask()
    combos = [
        ("night", "fog"),
        ("dusk", "rain"),
        ("day", "clear"),
        ("noon", "snow"),
    ]
    for lighting, weather in combos:
        depth = make_depth_ramp()
        frame = make_scene(lighting, weather)
        info = analyzer.analyze(depth, mask)
        decision = engine.decide(mask, depth, info, moving_on_path_frac=0.0)
        assert decision.cmd in {"FORWARD", "LEFT", "RIGHT", "STOP"}
        assert isinstance(frame, np.ndarray) and frame.shape[0] > 0

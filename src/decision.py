from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

CONF_STOP = 0.65
CONF_STEER = 0.50

def _front_arc_bbox(w: int, h: int) -> Tuple[int,int,int,int]:
    # Center 40% width, top 70% height (tune as needed)
    x1 = int(0.3 * w); x2 = int(0.7 * w)
    y1 = 0;            y2 = int(0.7 * h)
    return x1,y1,x2,y2

def _band_counts(dets: List[Dict]) -> Tuple[int,int]:
    left  = sum(1 for d in dets if d["bbox"][0] < 0.4 and d["conf"] >= 0.4)
    right = sum(1 for d in dets if d["bbox"][0] > 0.6 and d["conf"] >= 0.4)
    return left, right

def _center_threat(dets: List[Dict]) -> bool:
    for d in dets:
        cx,cy,_,_ = d["bbox"]
        if abs(cx-0.5) < 0.2 and cy < 0.7 and d["conf"] >= CONF_STOP:
            return True
    return False

def _estimate_step(depth: Optional[np.ndarray], trav_mask: Optional[np.ndarray]) -> Tuple[bool,bool]:
    """Return (step_up, step_down) using vertical depth gradients along mask boundary."""
    if depth is None or trav_mask is None:
        return False, False
    # Find horizontal edge in front third of image
    H, W = depth.shape[:2]
    roi = depth[int(0.4*H):int(0.85*H), int(0.25*W):int(0.75*W)]
    grad = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)  # vertical gradient
    # Heuristic: strong positive gradient ~ step-up, strong negative ~ step-down
    gy = np.median(grad)
    step_up   = gy > 0.12
    step_down = gy < -0.12
    return step_up, step_down

def _low_overhang(dets: List[Dict]) -> bool:
    # Tall bbox whose bottom is above eye-level band suggests an overhang/low bridge
    for d in dets:
        cx,cy,w,h = d["bbox"]
        if d["cls"] in {"sign", "branch", "overhang"}:
            return True
        # Approx: if bbox height tall and center high; tune with your data
        if h > 0.35 and cy < 0.35:
            return True
    return False

def decide_action(
    dets: List[Dict],
    fps: float,
    frame_shape: Tuple[int,int],
    trav_mask: Optional[np.ndarray] = None,
    depth_map: Optional[np.ndarray] = None
) -> Tuple[str, float]:
    H, W = frame_shape[:2]

    # Safety: FPS floor
    if fps is not None and fps < 8:
        return "STOP", 1.0

    # Overhang check
    if _low_overhang(dets):
        return "DUCK", 1.0

    # Step up/down via depth around traversable edge
    step_up, step_down = _estimate_step(depth_map, trav_mask)
    if step_up:
        return "STEP_UP", 0.9
    if step_down:
        return "STEP_DOWN", 0.9

    # STOP/VEER based on detections + free-space mask
    left_band, right_band = _band_counts(dets)
    center_threat = _center_threat(dets)

    if center_threat:
        # Check if traversable mask gives a side gap
        gap_left = False; gap_right = False
        if trav_mask is not None:
            lh = trav_mask[:, :W//2].mean() / 255.0
            rh = trav_mask[:, W//2:].mean() / 255.0
            gap_left  = lh > 0.20
            gap_right = rh > 0.20

        if gap_left and not gap_right:
            return "VEER_LEFT", 0.8
        if gap_right and not gap_left:
            return "VEER_RIGHT", 0.8

        if left_band < right_band * 0.7:
            return "VEER_LEFT", 0.8
        if right_band < left_band * 0.7:
            return "VEER_RIGHT", 0.8
        return "STOP", 1.0

    # Mild guidance
    if right_band - left_band > 2:
        return "VEER_LEFT", 0.5
    if left_band - right_band > 2:
        return "VEER_RIGHT", 0.5

    # If traversable mask is very thin ahead, warn
    if trav_mask is not None:
        x1,y1,x2,y2 = _front_arc_bbox(W,H)
        roi = trav_mask[y1:y2, x1:x2]
        if roi.size and (roi.mean()/255.0) < 0.08:
            return "STOP", 1.0

    return "CAUTION", 0.3

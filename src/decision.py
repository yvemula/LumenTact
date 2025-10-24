from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import cv2

CONF_STOP = 0.65  # kept for future use if you reintroduce detectors

def _front_arc_bbox(w: int, h: int) -> Tuple[int,int,int,int]:
    # Center 40% width, top 70% height (used as “forward” ROI)
    x1 = int(0.3 * w); x2 = int(0.7 * w)
    y1 = 0;           y2 = int(0.7 * h)
    return x1,y1,x2,y2

def _estimate_step(depth: Optional[np.ndarray], trav_mask: Optional[np.ndarray]) -> Tuple[bool,bool]:
    """
    Return (step_up, step_down) using vertical depth gradients over a mid-lower ROI.
    Positive gradient ~ step-up; negative ~ step-down.
    """
    if depth is None:
        return False, False
    H, W = depth.shape[:2]
    y1, y2 = int(0.45*H), int(0.85*H)
    x1, x2 = int(0.25*W), int(0.75*W)
    roi = depth[y1:y2, x1:x2]
    if roi.size == 0:
        return False, False
    # Vertical gradient
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    med = float(np.median(gy))
    step_up = med > 0.12
    step_down = med < -0.12
    return step_up, step_down

def _low_overhang_from_mask_and_depth(trav_mask: Optional[np.ndarray], depth: Optional[np.ndarray]) -> bool:
    """
    Heuristic: if upper band of image has very low traversability (like a solid ceiling/overhang)
    and depth decreases upward suggesting a close occluder, signal DUCK.
    """
    if trav_mask is None:
        return False
    H, W = trav_mask.shape[:2]
    top = trav_mask[:int(0.25*H), :]
    # “Ceiling” if nearly no free space visible up top
    ceiling_like = (top.mean() / 255.0) < 0.03

    if depth is None:
        return ceiling_like

    # Check if top band is generally “closer” than middle band
    mid = depth[int(0.35*H):int(0.55*H), :]
    topd = depth[:int(0.25*H), :]
    if topd.size and mid.size:
        top_m = float(np.median(topd))
        mid_m = float(np.median(mid))
        # MiDaS is inverse-ish; still, a noticeable difference suggests nearby occluder overhead
        close_overhead = (top_m - mid_m) > 0.08
        return ceiling_like and close_overhead
    return ceiling_like

def _lane_stats(trav_mask: Optional[np.ndarray]) -> Tuple[float,float,float]:
    """
    Returns (left_free, center_free, right_free) as fractions [0..1] in the lower-forward region.
    """
    if trav_mask is None:
        return 0.0, 0.0, 0.0
    H, W = trav_mask.shape[:2]
    # Focus on lower 60% as “walkable field”
    roi = trav_mask[int(0.4*H):, :]
    if roi.size == 0:
        return 0.0, 0.0, 0.0
    thirds = W // 3
    left  = roi[:, :thirds]
    center= roi[:, thirds:2*thirds]
    right = roi[:, 2*thirds:]
    return (left.mean()/255.0, center.mean()/255.0, right.mean()/255.0)

def decide_action(
    fps: float,
    frame_shape: Tuple[int,int],
    trav_mask: Optional[np.ndarray] = None,
    depth_map: Optional[np.ndarray] = None
) -> Tuple[str, float]:

    H, W = frame_shape[:2]

    # Safety: FPS floor
    if fps is not None and fps < 8:
        return "STOP", 1.0

    # Overhang detection
    if _low_overhang_from_mask_and_depth(trav_mask, depth_map):
        return "DUCK", 1.0

    # Step up/down estimation
    step_up, step_down = _estimate_step(depth_map, trav_mask)
    if step_up:
        return "STEP_UP", 0.9
    if step_down:
        return "STEP_DOWN", 0.9

    # Lane free-space
    left_f, ctr_f, right_f = _lane_stats(trav_mask)

    # If center is blocked or very thin, try to veer
    thin = ctr_f < 0.10
    very_thin = ctr_f < 0.06

    if thin:
        # Prefer the side with more free-space margin
        if left_f > right_f + 0.05 and left_f > 0.12:
            return "VEER_LEFT", 0.8 if very_thin else 0.6
        if right_f > left_f + 0.05 and right_f > 0.12:
            return "VEER_RIGHT", 0.8 if very_thin else 0.6
        # If both sides are poor too, STOP
        if max(left_f, right_f) < 0.10:
            return "STOP", 1.0

    # Global “front arc” thinness safety
    if trav_mask is not None:
        x1,y1,x2,y2 = _front_arc_bbox(W,H)
        roi = trav_mask[y1:y2, x1:x2]
        if roi.size and (roi.mean()/255.0) < 0.08:
            return "STOP", 1.0

    # Mild guidance when one side is clearly more open
    if right_f - left_f > 0.12:
        return "VEER_RIGHT", 0.5
    if left_f - right_f > 0.12:
        return "VEER_LEFT", 0.5

    return "CAUTION", 0.3

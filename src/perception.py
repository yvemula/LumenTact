from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class SemanticInfo:
    has_stairs: bool
    has_ramp: bool
    stairs_score: float
    ramp_score: float


class SemanticAnalyzer:
    """
    Lightweight cue extractor that inspects the depth map to find structures
    consistent with stairs (high-frequency vertical gradients) or ramps
    (smooth, monotonic slope). This does not replace a dedicated classifier,
    but it provides useful hints for decision making when no semantic model
    is available.
    """

    def __init__(
        self,
        stairs_grad_threshold: float = 0.12,
        ramp_grad_threshold: float = 0.04,
        min_mask_frac: float = 0.02,
        roi_frac: float = 0.5,
    ):
        self.stairs_grad_threshold = stairs_grad_threshold
        self.ramp_grad_threshold = ramp_grad_threshold
        self.min_mask_frac = min_mask_frac
        self.roi_frac = roi_frac

    def analyze(self, depth01: np.ndarray, trav_mask01: np.ndarray) -> SemanticInfo:
        h, w = depth01.shape[:2]
        y0 = int((1.0 - self.roi_frac) * h)
        roi_depth = depth01[y0:, :]
        roi_mask = trav_mask01[y0:, :]

        if roi_mask.mean() < self.min_mask_frac:
            return SemanticInfo(False, False, 0.0, 0.0)

        safe_depth = cv2.GaussianBlur(roi_depth, (5, 5), 0.0)
        grad_y = cv2.Sobel(safe_depth, cv2.CV_32F, 0, 1, ksize=3)
        masked_grad = grad_y[roi_mask > 0]
        if masked_grad.size == 0:
            return SemanticInfo(False, False, 0.0, 0.0)

        abs_grad = np.abs(masked_grad)
        stairs_score = float(np.clip(np.std(masked_grad), 0.0, 1.0))
        ramp_score = float(np.clip(np.abs(np.mean(masked_grad)), 0.0, 1.0))

        has_stairs = float(np.percentile(abs_grad, 90)) > self.stairs_grad_threshold
        has_ramp = ramp_score > self.ramp_grad_threshold and stairs_score < (self.stairs_grad_threshold * 1.5)

        return SemanticInfo(bool(has_stairs), bool(has_ramp), stairs_score, ramp_score)


@dataclass
class FlowInfo:
    moving_mask: Optional[np.ndarray]
    moving_on_path_frac: float
    global_motion_frac: float


class OpticalFlowTracker:
    """
    Maintains a rolling optical flow estimation and returns masks that highlight
    moving obstacles within or near the traversable corridor.
    """

    def __init__(
        self,
        mag_threshold: float = 0.45,
        smooth_kernel: int = 5,
        decay: float = 0.92,
    ):
        self.mag_threshold = mag_threshold
        self.smooth_kernel = smooth_kernel
        self.decay = decay
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_motion_mask: Optional[np.ndarray] = None

    def reset(self):
        self.prev_gray = None
        self.prev_motion_mask = None

    def update(self, frame_bgr: np.ndarray, trav_mask01: np.ndarray) -> FlowInfo:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return FlowInfo(None, 0.0, 0.0)

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        self.prev_gray = gray

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = (mag > self.mag_threshold).astype(np.uint8)

        if self.smooth_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.smooth_kernel, self.smooth_kernel))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, k)

        if self.prev_motion_mask is not None:
            motion_mask = cv2.addWeighted(motion_mask.astype(np.float32), 1.0,
                                          self.prev_motion_mask.astype(np.float32), self.decay, 0.0)
            motion_mask = (motion_mask > 0.5).astype(np.uint8)
        self.prev_motion_mask = motion_mask

        moving_on_path = (motion_mask > 0) & (trav_mask01 > 0)
        on_path_frac = float(moving_on_path.mean())
        global_frac = float(motion_mask.mean())

        return FlowInfo(moving_on_path.astype(np.uint8), on_path_frac, global_frac)

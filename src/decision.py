# src/decision.py
from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class Decision:
    cmd: str                 # "STOP" | "FORWARD" | "LEFT" | "RIGHT"
    steer: float             # -1.0 (hard left) .. +1.0 (hard right), 0 = straight
    reason: str              # human-readable reason
    traversable_frac: float  # fraction of traversable area used for decision
    target_x: float | None   # 0..1 column target of traversable center in ROI
    confidence: float = 1.0  # 0..1 confidence (used for haptics/failsafes)

class DecisionEngine:
    """
    Converts traversable mask + depth into simple navigation commands.
    - Looks in a bottom ROI (walkable ground).
    - Computes a 'target corridor' (center of mass of traversable).
    - Issues LEFT/RIGHT if target is off-center; FORWARD if centered; STOP if blocked.
    """

    def __init__(
        self,
        bottom_roi_frac: float = 0.40,     # take bottom 40% of the image
        min_trav_frac: float = 0.08,       # require at least 8% traversable in ROI
        deadband: float = 0.10,            # +/- 10% of width counts as "centered"
        near_depth_stop: float = 0.18,     # if depth01 is < this in center-bottom, stop
        center_width_frac: float = 0.25,   # width of center window for near-depth check
    ):
        self.bottom_roi_frac = bottom_roi_frac
        self.min_trav_frac   = min_trav_frac
        self.deadband        = deadband
        self.near_depth_stop = near_depth_stop
        self.center_width_frac = center_width_frac

    def _bottom_roi(self, h: int) -> slice:
        y0 = int((1.0 - self.bottom_roi_frac) * h)
        return slice(y0, h)

    def decide(self, mask01: np.ndarray, depth01: np.ndarray) -> Decision:
        """
        mask01: HxW uint8 {0,1} traversable
        depth01: HxW float32 [0,1], larger = farther
        """
        h, w = mask01.shape
        roi_y = self._bottom_roi(h)
        roi_mask = mask01[roi_y, :]
        roi_depth = depth01[roi_y, :]

        def make_confidence(trav_frac: float, offset: float = 0.0) -> float:
            base = min(1.0, max(0.0, (trav_frac - self.min_trav_frac) / max(1e-6, 0.5 - self.min_trav_frac)))
            center_bonus = 1.0 - min(1.0, abs(offset) / 0.5)
            return float(max(0.0, min(1.0, 0.5 * base + 0.5 * center_bonus)))

        # 1) STOP if center-bottom is too near (step/obstacle)
        c_w = int(self.center_width_frac * w)
        cx0 = (w - c_w) // 2
        cx1 = cx0 + c_w
        center_near = float(np.nanmedian(roi_depth[:, cx0:cx1]))
        if center_near < self.near_depth_stop:
            trav_frac = float(roi_mask.mean())
            return Decision(
                cmd="STOP", steer=0.0, reason=f"near obstacle/step (depth={center_near:.2f})",
                traversable_frac=trav_frac, target_x=None, confidence=1.0
            )

        # 2) Compute traversable coverage & center-of-mass of ROI
        trav_frac = float(roi_mask.mean())  # 0..1
        if trav_frac < self.min_trav_frac:
            return Decision(
                cmd="STOP", steer=0.0, reason=f"insufficient free space ({trav_frac:.3f})",
                traversable_frac=trav_frac, target_x=None, confidence=0.0
            )

        # Column weights: prefer deeper pixels (safer)
        weights = roi_mask.astype(np.float32) * (roi_depth ** 1.0)
        col_scores = weights.sum(axis=0)  # shape: (W,)

        if col_scores.max() <= 0:
            return Decision(
                cmd="STOP", steer=0.0, reason="no valid traversable columns",
                traversable_frac=trav_frac, target_x=None, confidence=0.0
            )

        # Target column = weighted centroid
        xs = np.arange(w, dtype=np.float32)
        target_col = float((col_scores * xs).sum() / max(1e-6, col_scores.sum()))
        target_x01 = target_col / max(1, w - 1)

        # 3) Decide steering
        offset = (target_x01 - 0.5)  # -0.5..+0.5
        if abs(offset) <= self.deadband:
            return Decision(
                cmd="FORWARD",
                steer=offset * 2.0,  # small trim
                reason=f"corridor centered (offset={offset:+.2f})",
                traversable_frac=trav_frac,
                target_x=target_x01,
                confidence=make_confidence(trav_frac, offset)
            )
        elif offset > 0:
            # steer right, normalized to [-1,+1]
            steer = min(1.0, (offset - self.deadband) / (0.5 - self.deadband))
            return Decision(
                cmd="RIGHT", steer=steer,
                reason=f"corridor right (offset={offset:+.2f})",
                traversable_frac=trav_frac, target_x=target_x01,
                confidence=make_confidence(trav_frac, offset)
            )
        else:
            steer = -min(1.0, (-offset - self.deadband) / (0.5 - self.deadband))
            return Decision(
                cmd="LEFT", steer=steer,
                reason=f"corridor left (offset={offset:+.2f})",
                traversable_frac=trav_frac, target_x=target_x01,
                confidence=make_confidence(trav_frac, offset)
            )

    # Optional: draw debug overlay with ROI and target column
    def draw_debug(self, bgr: np.ndarray, mask01: np.ndarray, decision: Decision) -> np.ndarray:
        out = bgr.copy()
        h, w = mask01.shape
        # ROI box
        y0 = int((1 - self.bottom_roi_frac) * h)
        cv2.rectangle(out, (0, y0), (w - 1, h - 1), (0, 255, 255), 2)

        # Target column
        if decision.target_x is not None:
            x = int(decision.target_x * (w - 1))
            cv2.line(out, (x, y0), (x, h - 1), (255, 0, 255), 2)

        # Command text
        txt = (
            f"{decision.cmd} | steer={decision.steer:+.2f} | "
            f"free={decision.traversable_frac:.2f} | conf={decision.confidence:.2f} | {decision.reason}"
        )
        cv2.putText(out, txt, (10, max(25, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out

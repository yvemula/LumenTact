from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import deque
import time
import math

# -----------------------------
# Config data classes
# -----------------------------

@dataclass
class ClassCfg:
    """Per-class thresholds and priority."""
    conf: float = 0.25           # minimum confidence to consider
    near_h: float = 0.35         # bbox height fraction to call "near"
    priority: int = 10           # higher => more urgent

@dataclass
class NavThresholds:
    """Global thresholds for geometry and policy."""
    center_arc: float = 0.20     # +/- fraction around image center considered "center"
    steer_margin: float = 0.15   # extra clearance needed to steer to a side
    min_side_clear: float = 0.35 # if either side has at least this clearance, allow VEER
    ema_alpha: float = 0.35      # EMA smoothing for side clearance [0..1], higher = quicker
    debounce_n: int = 3          # require same action this many frames to emit
    cooldown_s: float = 0.60     # do not resend the SAME action within this window
    sticky_safety_s: float = 0.75 # how long DUCK/STOP stay sticky once triggered
    base_conf: float = 0.20      # floor conf across all classes (safety net)

# Default per-class config (tweak freely)
DEFAULT_PER_CLASS: Dict[str, ClassCfg] = {
    "overhang":    ClassCfg(conf=0.35, near_h=0.30, priority=100),
    "stair":       ClassCfg(conf=0.30, near_h=0.30, priority=90),
    "curb":        ClassCfg(conf=0.30, near_h=0.30, priority=80),
    "person":      ClassCfg(conf=0.25, near_h=0.35, priority=70),
    "car":         ClassCfg(conf=0.25, near_h=0.35, priority=70),
    "bicycle":     ClassCfg(conf=0.25, near_h=0.35, priority=60),
    "stop sign":   ClassCfg(conf=0.25, near_h=0.40, priority=20),
    "bench":       ClassCfg(conf=0.20, near_h=0.40, priority=5),
    # add your custom classes as needed...
}

# -----------------------------
# Helper functions
# -----------------------------

def _cx_norm(box, W):
    """Normalized bbox center x in [0,1]."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0) / float(W)

def _h_frac(box, H):
    """BBox height as fraction of image height."""
    x1, y1, x2, y2 = box
    return (y2 - y1) / float(H + 1e-6)

def _center_overlap(cx, center_arc):
    """How much the bbox center lies inside the center band. 1.0 = dead center, 0 = outside band."""
    # Linear taper from band edge to center (triangle profile)
    delta = abs(cx - 0.5)
    if delta >= center_arc:
        return 0.0
    return 1.0 - (delta / center_arc)

def _ema(prev, new, alpha):
    """Exponentially weighted moving average."""
    return alpha * new + (1 - alpha) * prev

# -----------------------------
# Stateful decision class (handles smoothing, debounce, cooldown)
# -----------------------------

@dataclass
class DecisionState:
    """Keeps short-term memory to stabilize outputs."""
    thr: NavThresholds = field(default_factory=NavThresholds)
    per_class: Dict[str, ClassCfg] = field(default_factory=lambda: DEFAULT_PER_CLASS.copy())

    # Temporal state
    left_clear_ema: float = 1.0
    right_clear_ema: float = 1.0
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=3))
    last_emit: Tuple[str, float] = ("", 0.0)             # (action, timestamp)
    sticky_until: float = 0.0                            # deadline until which safety action is sticky
    sticky_action: Optional[str] = None

    def _is_near(self, name: str, box, H: int) -> bool:
        """Class-aware 'near' using bbox height."""
        hfrac = _h_frac(box, H)
        cfg = self.per_class.get(name, ClassCfg())
        return hfrac >= cfg.near_h

    def _risk_score(self, name: str, conf: float, box, W: int, H: int) -> float:
        """
        Risk = priority * center_overlap * nearness * confidence
        This ranks obstacles for STOP vs VEER decisions.
        """
        pc = self.per_class.get(name, ClassCfg())
        cx = _cx_norm(box, W)
        near = 1.0 if self._is_near(name, box, H) else 0.0
        center = _center_overlap(cx, self.thr.center_arc)
        conf_clip = max(conf, 0.0)
        return pc.priority * center * near * conf_clip

    def _update_clearance(self, dets: List[Tuple[str, float, tuple]], W: int, H: int):
        """
        Update left/right clearance with EMA using obstacle apparent size.
        Larger/nearer obstacles reduce clearance on their side.
        """
        # Start from optimistic clearance each frame
        left, right = 1.0, 1.0
        for name, conf, box in dets:
            pc = self.per_class.get(name)
            if not pc:
                continue
            # enforce base confidence floor
            if conf < max(pc.conf, self.thr.base_conf):
                continue
            # treat these as obstacles that reduce clearance
            if name in ("person", "car", "bicycle", "bench", "stop sign"):
                cx = _cx_norm(box, W)
                hfrac = _h_frac(box, H)
                # Heuristic: larger hfrac => closer => reduce clearance more
                if cx < 0.5:
                    left = min(left, 1.0 - hfrac)
                else:
                    right = min(right, 1.0 - hfrac)

        # Smooth with EMA so single-frame blips don't jerk decisions
        self.left_clear_ema = _ema(self.left_clear_ema, left, self.thr.ema_alpha)
        self.right_clear_ema = _ema(self.right_clear_ema, right, self.thr.ema_alpha)

    def choose_action(self, dets: List[Tuple[str, float, tuple]], W: int, H: int) -> str:
        """
        Stateless API for callers, but internally maintains smoothing/debounce/cooldown.
        dets: list of (class_name, conf, (x1,y1,x2,y2))
        """
        now = time.time()

        # 1) Update side clearance smoothing
        self._update_clearance(dets, W, H)

        # 2) Compute high-priority cues (sticky safety)
        saw_overhang = any((n == "overhang" and self._is_near(n, b, H) and c >= max(self.per_class[n].conf, self.thr.base_conf))
                           for n, c, b in dets)
        saw_stair    = any((n in ("stair", "stairs") and self._is_near(n, b, H) and c >= max(self.per_class[n].conf, self.thr.base_conf))
                           for n, c, b in dets)
        saw_curb     = any((n == "curb" and self._is_near(n, b, H) and c >= max(self.per_class[n].conf, self.thr.base_conf))
                           for n, c, b in dets)

        # 3) If sticky safety active, keep it until timer elapses
        if self.sticky_action and now < self.sticky_until:
            action = self.sticky_action
        else:
            self.sticky_action = None  # expired

            # 4) Safety priority order
            if saw_overhang:
                action = "DUCK"
            elif saw_stair or saw_curb:
                action = "STEP_UP"  # refine to STEP_DOWN when you have that signal
            else:
                # 5) Center risk: decide STOP/VEER
                # Find max risk among center obstacles
                center_risk = 0.0
                for name, conf, box in dets:
                    if name not in self.per_class: 
                        continue
                    if conf < max(self.per_class[name].conf, self.thr.base_conf):
                        continue
                    r = self._risk_score(name, conf, box, W, H)
                    center_risk = max(center_risk, r)

                # Steering based on smoothed clearances
                L, R = self.left_clear_ema, self.right_clear_ema
                if center_risk > 0.0:
                    # Prefer the clearer side if margin is sufficient
                    if L > R + self.thr.steer_margin:
                        action = "VEER_LEFT"
                    elif R > L + self.thr.steer_margin:
                        action = "VEER_RIGHT"
                    elif max(L, R) > self.thr.min_side_clear:
                        action = "VEER_LEFT" if L >= R else "VEER_RIGHT"
                    else:
                        action = "STOP"
                else:
                    # No central risk → gentle guidance by clearance bias
                    if (L - R) > 0.25:
                        action = "VEER_LEFT"
                    elif (R - L) > 0.25:
                        action = "VEER_RIGHT"
                    else:
                        action = "CAUTION"

        # 6) Debounce & cooldown logic
        self.recent_actions.append(action)
        agreed = (len(self.recent_actions) == self.recent_actions.maxlen and
                  len(set(self.recent_actions)) == 1)

        # Cooldown for repeating the same action too frequently
        last_action, last_ts = self.last_emit
        within_cooldown = (action == last_action) and ((now - last_ts) < self.thr.cooldown_s)

        emit = action
        if agreed and not within_cooldown:
            self.last_emit = (action, now)
            # Start sticky timer for strong safety actions
            if action in ("DUCK", "STOP"):
                self.sticky_action = action
                self.sticky_until = now + self.thr.sticky_safety_s
        else:
            # Don’t emit a new action yet; keep last emitted (or say CAUTION upstream)
            emit = last_action if last_action else action

        return emit

# -----------------------------
# Backward-compatible stateless wrapper
# -----------------------------

# Create a module-level singleton for simple uses
_default_state = DecisionState()

def choose_action(
    dets: List[Tuple[str, float, tuple]],
    W: int,
    H: int,
    per_class: Dict[str, ClassCfg] = DEFAULT_PER_CLASS,
    thr: NavThresholds = NavThresholds()
) -> str:
    """
    Backward-compatible function signature.
    Internally forwards to a stateful DecisionState (smoothing/debounce/cooldown).
    """
    # Update defaults if caller passes custom configs
    _default_state.per_class = per_class
    _default_state.thr = thr
    return _default_state.choose_action(dets, W, H)
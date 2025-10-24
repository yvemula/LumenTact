# src/decision.py
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class NavThresholds:
    center_arc: float = 0.20
    near_h_frac: float = 0.35
    steer_margin: float = 0.15

@dataclass
class PerClassCfg:
    conf: float
    priority: int

DEFAULT_PER_CLASS = {
    "overhang": PerClassCfg(0.35, 100),
    "stair":    PerClassCfg(0.30, 90),
    "curb":     PerClassCfg(0.30, 80),
    "person":   PerClassCfg(0.25, 70),
    "car":      PerClassCfg(0.25, 70),
    "bicycle":  PerClassCfg(0.25, 60),
    "bench":    PerClassCfg(0.20, 5),
    "stop sign":PerClassCfg(0.25, 20),
}

def is_near(box, H, near_h_frac):
    x1,y1,x2,y2 = box
    return ((y2-y1)/H) >= near_h_frac

def choose_action(dets: List[Tuple[str,float,tuple]], W:int, H:int,
                  per_class=DEFAULT_PER_CLASS, thr=NavThresholds()) -> str:
    left_cl, right_cl = 1.0, 1.0
    center_danger = False
    stair = curb = overhang = False

    def cx(b): x1,y1,x2,y2 = b; return ((x1+x2)/2)/W
    for name, conf, box in dets:
        if name not in per_class or conf < per_class[name].conf:
            continue
        near = is_near(box, H, thr.near_h_frac)
        if name in ("stair","stairs") and near: stair = True
        if name == "curb" and near:            curb  = True
        if name == "overhang" and near:        overhang = True

        if name in ("person","car","bicycle","bench","stop sign"):
            c = cx(box)
            if abs(c - 0.5) < thr.center_arc and near:
                center_danger = True
            hfrac = (box[3]-box[1])/H
            if c < 0.5:  left_cl  = min(left_cl,  1.0 - hfrac)
            else:        right_cl = min(right_cl, 1.0 - hfrac)

    if overhang: return "DUCK"
    if stair:    return "STEP_UP"
    if curb:     return "STEP_UP"

    if center_danger:
        if left_cl > right_cl + thr.steer_margin:  return "VEER_LEFT"
        if right_cl > left_cl + thr.steer_margin:  return "VEER_RIGHT"
        if max(left_cl, right_cl) > 0.35:          return "VEER_LEFT" if left_cl>=right_cl else "VEER_RIGHT"
        return "STOP"

    if left_cl - right_cl > 0.25:  return "VEER_LEFT"
    if right_cl - left_cl > 0.25:  return "VEER_RIGHT"
    return "CAUTION"

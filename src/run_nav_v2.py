
import os, sys, time, csv, math, argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

DEFAULT_CLASSES = ["person","bicycle","car","traffic light","stop sign","bench"]  # extend with your custom ones

MODE_PRESETS = {
    # (center_arc, near_h_frac, conf_floor, steer_margin, debounce, cooldown)
    "safe":        (0.18, 0.30, 0.20, 0.18, 4, 0.8),
    "normal":      (0.20, 0.35, 0.25, 0.15, 3, 0.6),
    "aggressive":  (0.22, 0.40, 0.30, 0.10, 2, 0.4),
}

@dataclass
class PerClass:
    conf: float
    priority: int

@dataclass
class NavConfig:
    classes: List[str] = field(default_factory=lambda: DEFAULT_CLASSES.copy())
    per_class: Dict[str, PerClass] = field(default_factory=lambda: {
        "overhang": PerClass(0.35, 100),
        "stair":    PerClass(0.30, 90),
        "curb":     PerClass(0.30, 80),
        "person":   PerClass(0.25, 70),
        "car":      PerClass(0.25, 70),
        "bicycle":  PerClass(0.25, 60),
        "traffic light": PerClass(0.25, 20),
        "stop sign":     PerClass(0.25, 20),
        "bench":    PerClass(0.20, 5),
    })
    center_arc: float = 0.20          # +/- fraction around center considered “center”
    near_h_frac: float = 0.35         # bbox height fraction that counts as NEAR
    steer_margin: float = 0.15        # extra clearance needed to steer
    debounce_n: int = 3               # frames of agreement to emit action
    action_cooldown_s: float = 0.6    # do not resend same action within this time
    device: str = "auto"              # "auto" | "mps" | "cpu" | "0" etc.
    imgsz: int = 640
    iou: float = 0.6
    amp: bool = True                  # mixed precision (helps MPS)
    visualize: bool = True

class HapticsBase:
    def send(self, action: str, strength: float = 1.0):
        raise NotImplementedError

class NoOpHaptics(HapticsBase):
    def send(self, action: str, strength: float = 1.0):
        print(f"[HAPTIC] {action} ({strength:.2f})")

def estimate_is_near(box: Tuple[float,float,float,float], H: int, near_h_frac: float) -> bool:
    # Fallback heuristic: bbox height fraction
    x1,y1,x2,y2 = box
    return ((y2 - y1) / H) >= near_h_frac

def choose_action(dets, W, H, cfg: NavConfig) -> str:
    left_clearance = 1.0
    right_clearance = 1.0
    center_danger = False
    stair_ahead = curb_ahead = overhang_ahead = False

    def cx_norm(b):
        x1,y1,x2,y2 = b
        return ((x1+x2)/2)/W

    for name, conf, box in dets:
        pc = cfg.per_class.get(name)
        if not pc: continue
        if conf < max(pc.conf, 0.0):  # per-class threshold
            continue
        near = estimate_is_near(box, H, cfg.near_h_frac)
        if name in ("stair","stairs") and near:   stair_ahead = True
        if name == "curb" and near:               curb_ahead  = True
        if name == "overhang" and near:           overhang_ahead = True

        # treat as obstacle?
        if name in ("person","car","bicycle","bench","stop sign"):
            c = cx_norm(box)
            if abs(c - 0.5) < cfg.center_arc and near:
                center_danger = True
            # reduce clearance on that side based on apparent size
            hfrac = (box[3]-box[1])/H
            if c < 0.5:
                left_clearance  = min(left_clearance, 1.0 - hfrac)
            else:
                right_clearance = min(right_clearance, 1.0 - hfrac)

    # Priority rules
    if overhang_ahead: return "DUCK"
    if stair_ahead:    return "STEP_UP"    # refine later w/ step-down logic
    if curb_ahead:     return "STEP_UP"    # MVP behavior

    if center_danger:
        if left_clearance > right_clearance + cfg.steer_margin:
            return "VEER_LEFT"
        if right_clearance > left_clearance + cfg.steer_margin:
            return "VEER_RIGHT"
        if max(left_clearance, right_clearance) > 0.35:
            return "VEER_LEFT" if left_clearance >= right_clearance else "VEER_RIGHT"
        return "STOP"

    # gentle guidance
    if left_clearance - right_clearance > 0.25:
        return "VEER_LEFT"
    if right_clearance - left_clearance > 0.25:
        return "VEER_RIGHT"

    return "CAUTION"

# ---------- Utilities ----------
def pick_device(flag: str) -> str:
    if flag != "auto": return flag
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def draw_overlay(img, action, cfg: NavConfig):
    H, W = img.shape[:2]
    cx1 = int((0.5 - cfg.center_arc) * W)
    cx2 = int((0.5 + cfg.center_arc) * W)
    cv2.line(img, (cx1,0), (cx1,H), (0,255,255), 1)
    cv2.line(img, (cx2,0), (cx2,H), (0,255,255), 1)
    cv2.putText(img, f"ACTION: {action}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)
    return img

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--source", default="0", help="0 for webcam, or path/to/video_or_folder")
    ap.add_argument("--mode", choices=MODE_PRESETS.keys(), default="normal")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save-video", default="", help="output mp4 to save annotated frames")
    ap.add_argument("--log", default="nav_log.csv")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--batch-delay", type=float, default=0.0, help="sleep per frame (replay throttle)")
    args = ap.parse_args()

    # Build config from mode
    center_arc, near_h, conf_floor, steer_margin, debounce_n, cooldown = MODE_PRESETS[args.mode]
    cfg = NavConfig(
        center_arc=center_arc,
        near_h_frac=near_h,
        steer_margin=steer_margin,
        debounce_n=debounce_n,
        action_cooldown_s=cooldown,
        imgsz=args.imgsz,
        device=pick_device(args.device),
        visualize=not args.no_vis,
    )
    # apply base conf floor
    for k,v in cfg.per_class.items():
        cfg.per_class[k] = PerClass(conf=max(v.conf, conf_floor), priority=v.priority)

    print(f"[INFO] device={cfg.device} mode={args.mode} imgsz={cfg.imgsz} vis={cfg.visualize}")

    model = YOLO(args.weights)

    # Source: webcam or video/file
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERR] Cannot open source: {args.source}")
        sys.exit(1)

    # Video writer (optional)
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (W,H))

    # Logging
    logf = open(args.log, "w", newline="")
    logger = csv.writer(logf)
    logger.writerow(["ts","action","n_dets","names","confs"])

    haptics: HapticsBase = NoOpHaptics()
    act_hist = deque(maxlen=cfg.debounce_n)
    last_sent = ("", 0.0)  # (action, timestamp)

    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]

        res = model.predict(
            frame,
            device=cfg.device,
            imgsz=cfg.imgsz,
            conf=min([p.conf for p in cfg.per_class.values()]),
            iou=cfg.iou,
            half=False,  # MPS ignores half; kept for completeness
            verbose=False
        )[0]

        dets = []
        for b in res.boxes:
            cls_idx = int(b.cls.item())
            name = res.names[cls_idx]
            conf = float(b.conf.item())
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            dets.append((name, conf, (x1,y1,x2,y2)))

        action = choose_action(dets, W, H, cfg)
        act_hist.append(action)

        # Debounce + cooldown
        now = time.time()
        agreed = len(act_hist) == cfg.debounce_n and len(set(act_hist)) == 1
        can_send = (action != last_sent[0]) or (now - last_sent[1] >= cfg.action_cooldown_s)
        if agreed and can_send:
            strength = 1.0 if action in ("STOP","DUCK","STEP_UP","STEP_DOWN") else 0.6
            haptics.send(action, strength)
            last_sent = (action, now)

        # Log
        logger.writerow([now, action, len(dets), "|".join([d[0] for d in dets]), "|".join([f"{d[1]:.2f}" for d in dets])])

        # Visualize / write
        vis = res.plot() if cfg.visualize else frame
        if cfg.visualize:
            vis = draw_overlay(vis, action, cfg)
            cv2.imshow("HapticNav v2", vis)
        if writer:
            writer.write(vis)

        if cfg.visualize and (cv2.waitKey(1) & 0xFF == 27):  # ESC
            break

        if args.batch_delay > 0:
            time.sleep(args.batch_delay)

    cap.release()
    if writer: writer.release()
    logf.close()
    cv2.destroyAllWindows()
    print("[DONE] Closed cleanly")

if __name__ == "__main__":
    main()

# src/main.py
import sys
import time
import csv
import argparse
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO

# --- IMPORT HAPTICS ---
try:
    from haptics import DRV2605LHaptics, NoOpHaptics
except ImportError:
    print("[WARN] haptics.py not found. Using NoOp.")
    class NoOpHaptics:
        def send(self, a, s): print(f"[MOCK] {a}")
    DRV2605LHaptics = NoOpHaptics

# --- CONFIGURATION ---
DEFAULT_CLASSES = ["person","bicycle","car","traffic light","stop sign","bench"]

MODE_PRESETS = {
    # (center_arc, near_h_frac, conf_floor, steer_margin, debounce)
    "safe":        (0.25, 0.25, 0.25, 0.20, 3), # Wide center, sensitive
    "normal":      (0.20, 0.35, 0.30, 0.15, 2), # Balanced
    "aggressive":  (0.15, 0.45, 0.35, 0.10, 1), # Narrow center, requires close obj
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
        "person":   PerClass(0.30, 70),
        "car":      PerClass(0.30, 70),
        "bicycle":  PerClass(0.25, 60),
        "bench":    PerClass(0.25, 20),
    })
    center_arc: float = 0.20
    near_h_frac: float = 0.35
    steer_margin: float = 0.15
    debounce_n: int = 3
    action_cooldown_s: float = 0.8
    device: str = "auto"
    imgsz: int = 640
    visualize: bool = False

def estimate_is_near(box: Tuple[float,float,float,float], H: int, near_h_frac: float) -> bool:
    # Heuristic: If bbox bottom edge (y2) is very low, or box height is huge
    x1, y1, x2, y2 = box
    box_h = y2 - y1
    # 1. Box covers huge portion of screen height
    if (box_h / H) > near_h_frac: return True
    # 2. Box bottom is near bottom of frame
    if (y2 / H) > 0.90: return True
    return False

def choose_action(dets, W, H, cfg: NavConfig) -> str:
    left_clearance = 1.0
    right_clearance = 1.0
    center_danger = False
    stair_ahead = overhang_ahead = False

    def cx_norm(b):
        x1,y1,x2,y2 = b
        return ((x1+x2)/2)/W

    # 1. Process Detections
    for name, conf, box in dets:
        pc = cfg.per_class.get(name)
        if not pc or conf < pc.conf: continue

        near = estimate_is_near(box, H, cfg.near_h_frac)
        
        # Immediate Hazards
        if name in ("stair", "stairs") and near: stair_ahead = True
        if name == "overhang" and near: overhang_ahead = True

        # Obstacles
        if name in ("person", "car", "bicycle", "bench", "stop sign", "tree"):
            c = cx_norm(box)
            # Check if object is in the "Danger Center"
            if abs(c - 0.5) < cfg.center_arc and near:
                center_danger = True
            
            # Update clearances (0.0 = blocked, 1.0 = clear)
            # Use box height as proxy for how "blocked" that side is
            hfrac = min((box[3]-box[1])/H, 1.0)
            if c < 0.5:
                left_clearance = min(left_clearance, 1.0 - hfrac)
            else:
                right_clearance = min(right_clearance, 1.0 - hfrac)

    # 2. Priority Logic
    if overhang_ahead: return "DUCK"
    if stair_ahead:    return "STEP_UP"

    if center_danger:
        # If both sides are tight, STOP
        if left_clearance < 0.3 and right_clearance < 0.3:
            return "STOP"
        
        # Deadband Logic: Only veer if one side is significantly better
        if left_clearance > right_clearance + cfg.steer_margin:
            return "VEER_LEFT"
        elif right_clearance > left_clearance + cfg.steer_margin:
            return "VEER_RIGHT"
        else:
            # Equal danger? Just stop.
            return "STOP"

    # 3. Gentle Guidance (No center danger, but maybe drift avoidance)
    guidance_thresh = 0.4
    if left_clearance - right_clearance > guidance_thresh:
        return "VEER_LEFT"
    if right_clearance - left_clearance > guidance_thresh:
        return "VEER_RIGHT"

    return "CAUTION" # Pulse to show system active? Or "NONE"

def draw_overlay(img, action, cfg: NavConfig):
    H, W = img.shape[:2]
    # Draw Center Danger Zone
    cx1 = int((0.5 - cfg.center_arc) * W)
    cx2 = int((0.5 + cfg.center_arc) * W)
    
    # Overlay lines
    color = (0, 0, 255) if action == "STOP" else (0, 255, 255)
    cv2.line(img, (cx1, 0), (cx1, H), (100,100,100), 1)
    cv2.line(img, (cx2, 0), (cx2, H), (100,100,100), 1)
    
    # Text
    cv2.putText(img, f"ACT: {action}", (20, 50), cv2.FONT_HERSHEY_BOLD, 1.2, color, 2)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt") # Uses standard nano model
    ap.add_argument("--source", default="0") # Webcam
    ap.add_argument("--haptics", choices=["real", "mock"], default="mock")
    ap.add_argument("--mode", choices=MODE_PRESETS.keys(), default="normal")
    ap.add_argument("--vis", action="store_true", help="Show video window")
    ap.add_argument("--skip", type=int, default=1, help="Process 1 out of N frames")
    args = ap.parse_args()

    # --- SETUP ---
    p = MODE_PRESETS[args.mode]
    cfg = NavConfig(
        center_arc=p[0], near_h_frac=p[1], steer_margin=p[3], 
        debounce_n=p[4], visualize=args.vis
    )
    # Apply conf floor
    for k in cfg.per_class: cfg.per_class[k].conf = max(cfg.per_class[k].conf, p[2])

    # Init Haptics
    if args.haptics == "real":
        try:
            # NOTE: Update addresses if you have multiple motors!
            haptics = DRV2605LHaptics(addresses=[0x5A], motor_type="ERM")
        except Exception as e:
            print(f"[ERR] Haptics failed: {e}. Falling back to NoOp.")
            haptics = NoOpHaptics()
    else:
        haptics = NoOpHaptics()

    # Load Model
    print(f"[INFO] Loading {args.weights} on {cfg.device}...")
    model = YOLO(args.weights)
    
    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    
    act_hist = deque(maxlen=cfg.debounce_n)
    last_action_time = 0.0
    last_sent_action = ""
    frame_idx = 0

    print("[INFO] Starting loop. Press ESC to quit.")
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        frame_idx += 1
        
        # --- FRAME SKIPPING FOR PERFORMANCE ---
        if frame_idx % args.skip != 0:
            if cfg.visualize:
                cv2.imshow("Nav", frame)
                if cv2.waitKey(1) == 27: break
            continue

        H, W = frame.shape[:2]

        # --- INFERENCE ---
        res = model.predict(frame, conf=0.25, verbose=False)[0]
        
        dets = []
        for b in res.boxes:
            cls_idx = int(b.cls.item())
            name = res.names[cls_idx]
            conf = float(b.conf.item())
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            dets.append((name, conf, (x1,y1,x2,y2)))

        # --- LOGIC ---
        action = choose_action(dets, W, H, cfg)
        act_hist.append(action)

        # Debounce: Do we have consensus over N frames?
        consensus = len(set(act_hist)) == 1 and len(act_hist) == cfg.debounce_n
        
        if consensus:
            now = time.time()
            # Cooldown check
            if (action != last_sent_action) or (now - last_action_time > cfg.action_cooldown_s):
                if action not in ["CAUTION", "NONE"]:
                    # Fire Haptics!
                    haptics.send(action)
                    last_action_time = now
                    last_sent_action = action
                    print(f"[DECISION] {action}")

        # --- VISUALIZATION ---
        if cfg.visualize:
            vis = res.plot()
            vis = draw_overlay(vis, action if consensus else "...", cfg)
            cv2.imshow("Nav", vis)
            if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
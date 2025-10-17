

import time
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ---- Config ----
CLASSES = {
    "person": 0, "bicycle": 1, "car": 2, "traffic light": 3, "stop sign": 4, "bench": 5,
}
CENTER_ARC = 0.20       # +/- 20% of image width around center
NEAR_H_FRAC = 0.35      # bbox height > 35% of image height => "near" (tune!)
CONF_THRESH = 0.25      # model conf threshold (tune per class later)
DEBOUNCE_N = 3          # need same action for N frames to send haptics

# Class priorities (higher = more urgent)
PRIORITY = {
    "overhang": 100, "stair": 90, "curb": 80,
    "person": 70, "car": 70, "bicycle": 60,
    "traffic light": 20, "stop sign": 20, "bench": 5
}

def send_haptics(action: str, strength: float = 1.0):
    print(f"[HAPTIC] {action} (strength={strength:.2f})")


def choose_action(dets, W, H):
    """
    dets: list of (cls_name, conf, (x1,y1,x2,y2))
    Returns action:str or None
    """
    # Partition space
    left_clearance = 1.0
    right_clearance = 1.0
    center_dangers = []

    # Helper: center fraction [0..1] of bbox
    def bbox_center_x(b):
        x1,y1,x2,y2 = b
        return (x1+x2)/2.0 / W

    def bbox_h_frac(b):
        x1,y1,x2,y2 = b
        return (y2 - y1) / H

    # Track special cues
    stair_ahead = False
    curb_ahead  = False
    overhang_ahead = False

    for name, conf, box in dets:
        if conf < CONF_THRESH:
            continue
        cx = bbox_center_x(box)
        hfrac = bbox_h_frac(box)
        near = hfrac >= NEAR_H_FRAC
        # Special cues
        if name in ("stair", "stairs"):
            if near: stair_ahead = True
        if name == "curb":
            if near: curb_ahead = True
        if name == "overhang":
            if near: overhang_ahead = True

        # Consider people/cars/bikes as obstacles
        is_obstacle = name in ("person","car","bicycle","bench","stop sign")  # expand as needed
        if is_obstacle:
            # center band => danger
            if abs(cx - 0.5) < CENTER_ARC and near:
                center_dangers.append((PRIORITY.get(name,10), name))
            # lower clearance estimate per side (bigger, nearer -> less clearance)
            if cx < 0.5:
                left_clearance  = min(left_clearance, 1.0 - hfrac)
            else:
                right_clearance = min(right_clearance, 1.0 - hfrac)

    # Hard safety first
    if overhang_ahead:
        return "DUCK"
    if stair_ahead:
        return "STEP_UP"   # or STEP_DOWN if you later detect down-stairs
    if curb_ahead:
        return "STEP_UP"   # MVP: treat curb as step up

    if center_dangers:
        # If both sides somewhat free, steer to better side, else STOP
        if left_clearance > right_clearance + 0.15:
            return "VEER_LEFT"
        elif right_clearance > left_clearance + 0.15:
            return "VEER_RIGHT"
        elif max(left_clearance, right_clearance) > 0.35:
            # pick the best side if at least one is decent
            return "VEER_LEFT" if left_clearance >= right_clearance else "VEER_RIGHT"
        else:
            return "STOP"

    # Gentle guidance: bias toward more clearance
    if left_clearance - right_clearance > 0.25:
        return "VEER_LEFT"
    if right_clearance - left_clearance > 0.25:
        return "VEER_RIGHT"

    return "CAUTION"  # default soft pulse

def main():
    # Load your best weights; start with yolov8n.pt if you only sanity-checked
    model = YOLO("runs/detect/train/weights/best.pt")  # or "yolov8n.pt"
    cap = cv2.VideoCapture(0)  # webcam; or path to a test video
    if not cap.isOpened():
        print("[ERR] Cannot open camera. Try a video file path.")
        return

    action_hist = deque(maxlen=DEBOUNCE_N)
    last_sent = None
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        # Inference (M1 GPU)
        results = model.predict(
            frame,
            device="mps",
            imgsz=640,
            conf=CONF_THRESH,
            iou=0.6,
            verbose=False
        )[0]

        # Collect detections (cls_name, conf, (x1,y1,x2,y2))
        dets = []
        for b in results.boxes:
            cls_idx = int(b.cls.item())
            name = results.names[cls_idx]
            conf = float(b.conf.item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            dets.append((name, conf, (x1,y1,x2,y2)))

        action = choose_action(dets, W, H)
        action_hist.append(action)

        # Debounce: only send if last N agree
        if len(action_hist) == action_hist.maxlen and len(set(action_hist)) == 1:
            if action != last_sent:
                send_haptics(action, strength=1.0 if action in ("STOP","DUCK","STEP_UP","STEP_DOWN") else 0.6)
                last_sent = action

        # (Optional) visualize sectors and action for debugging
        cx1 = int((0.5 - CENTER_ARC) * W); cx2 = int((0.5 + CENTER_ARC) * W)
        vis = results.plot()
        cv2.line(vis, (cx1, 0), (cx1, H), (0,255,255), 1)
        cv2.line(vis, (cx2, 0), (cx2, H), (0,255,255), 1)
        cv2.putText(vis, f"ACTION: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)
        cv2.imshow("HapticNav Debug", vis)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


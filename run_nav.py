

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

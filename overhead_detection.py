"""
LumenTact – Overhead Hazard Tracking
Adds persistent tracking and motion analysis.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import warnings
warnings.filterwarnings("ignore")

YOLO_MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45

OVERHEAD_CLASSES = ["person", "umbrella", "backpack", "traffic light"]

CRITICAL_RATIO = 0.35
WARNING_RATIO = 0.55
SAFE_RATIO = 0.70

TRACK_BUFFER = 15
VELOCITY_THRESHOLD = 5.0

# -------------------------------
# Hazard Object
# -------------------------------
class Hazard:
    def __init__(self, bbox, name):
        self.bbox = bbox
        self.name = name
        self.positions = deque(maxlen=TRACK_BUFFER)
        self.positions.append(self.center())
        self.approaching = False

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, bbox):
        self.bbox = bbox
        self.positions.append(self.center())

        if len(self.positions) >= 2:
            dy = self.positions[-1][1] - self.positions[-2][1]
            self.approaching = dy > VELOCITY_THRESHOLD

# -------------------------------
# Simple Tracker
# -------------------------------
class Tracker:
    def __init__(self):
        self.hazards = []

    def update(self, detections):
        self.hazards.clear()
        for bbox, name in detections:
            self.hazards.append(Hazard(bbox, name))

# -------------------------------
# Visualization
# -------------------------------
def draw_zones(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, int(h * CRITICAL_RATIO)), (0, 0, 255), -1)
    cv2.rectangle(overlay, (0, int(h * CRITICAL_RATIO)),
                  (w, int(h * WARNING_RATIO)), (0, 165, 255), -1)
    cv2.rectangle(overlay, (0, int(h * WARNING_RATIO)),
                  (w, int(h * SAFE_RATIO)), (0, 255, 255), -1)

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

# -------------------------------
# Main
# -------------------------------
def main():
    model = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(0)
    tracker = Tracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h = frame.shape[0]
        roi_y = int(h * SAFE_RATIO)

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []

        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name not in OVERHEAD_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if y2 < roi_y:
                detections.append(((x1, y1, x2, y2), cls_name))

        tracker.update(detections)

        output = frame.copy()
        draw_zones(output)

        for hazard in tracker.hazards:
            x1, y1, x2, y2 = map(int, hazard.bbox)
            color = (0, 0, 255) if hazard.approaching else (255, 255, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("Overhead Hazards Tracking", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

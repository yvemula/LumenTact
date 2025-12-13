"""
Collision Risk Scoring Module for LumenTact
Detection, tracking, and per-object risk scoring.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Dict
import math
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Configuration
# -------------------------------
YOLO_MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
TRACKING_HISTORY = 20

# -------------------------------
# Data Structures
# -------------------------------
@dataclass
class RiskScore:
    total: float
    proximity: float
    velocity: float
    size: float
    alignment: float
    ttc: float

@dataclass
class TrackedObject:
    obj_id: int
    cls_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    frame_id: int
    first_seen: int
    positions: deque = field(default_factory=lambda: deque(maxlen=TRACKING_HISTORY))
    velocities: deque = field(default_factory=lambda: deque(maxlen=TRACKING_HISTORY))

    def get_center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, bbox, confidence, frame_id):
        prev = self.get_center()
        self.bbox = bbox
        self.confidence = confidence
        self.frame_id = frame_id
        curr = self.get_center()
        self.positions.append(curr)
        self.velocities.append((curr[0] - prev[0], curr[1] - prev[1]))

    def speed(self):
        if not self.velocities:
            return 0.0
        vx = np.mean([v[0] for v in self.velocities])
        vy = np.mean([v[1] for v in self.velocities])
        return math.sqrt(vx*vx + vy*vy)

# -------------------------------
# Object Tracker
# -------------------------------
class ObjectTracker:
    def __init__(self):
        self.objects: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.frame_id = 0

    def update(self, detections):
        self.frame_id += 1

        for bbox, cls_name, conf in detections:
            match_id = self._match(bbox)
            if match_id is not None:
                self.objects[match_id].update(bbox, conf, self.frame_id)
            else:
                self.objects[self.next_id] = TrackedObject(
                    obj_id=self.next_id,
                    cls_name=cls_name,
                    bbox=bbox,
                    confidence=conf,
                    frame_id=self.frame_id,
                    first_seen=self.frame_id
                )
                self.next_id += 1

        self.objects = {
            oid: obj for oid, obj in self.objects.items()
            if self.frame_id - obj.frame_id <= 30
        }

    def risk_score(self, obj, frame_shape):
        h, w = frame_shape[:2]
        cx, _ = obj.get_center()
        x1, y1, x2, y2 = obj.bbox

        proximity = 1.0 - ((h - y2) / h)
        velocity = min(obj.speed() / 15.0, 1.0)
        size = min(((x2-x1)*(y2-y1)) / (w*h*0.3), 1.0)
        alignment = 1.0 - abs(cx - w/2) / (w/2)

        ttc = float('inf') if velocity < 0.1 else proximity / velocity

        total = (
            0.35 * proximity +
            0.25 * velocity +
            0.15 * size +
            0.25 * alignment
        )

        return RiskScore(total, proximity, velocity, size, alignment, ttc)

    def _match(self, bbox):
        best_iou, best_id = IOU_THRESHOLD, None
        for oid, obj in self.objects.items():
            iou = self._iou(bbox, obj.bbox)
            if iou > best_iou:
                best_iou, best_id = iou, oid
        return best_id

    @staticmethod
    def _iou(b1, b2):
        x1, y1, x2, y2 = b1
        x3, y3, x4, y4 = b2
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2-xi1)*(yi2-yi1)
        a1 = (x2-x1)*(y2-y1)
        a2 = (x4-x3)*(y4-y3)
        return inter / (a1 + a2 - inter)

# -------------------------------
# Main Loop
# -------------------------------
def main():
    model = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(0)
    tracker = ObjectTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detections.append(((x1, y1, x2, y2), cls, conf))

        tracker.update(detections)

        for obj in tracker.objects.values():
            risk = tracker.risk_score(obj, frame.shape)
            x1, y1, x2, y2 = map(int, obj.bbox)
            color = (0, int(255 * (1-risk.total)), int(255 * risk.total))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{obj.cls_name} {risk.total:.2f}",
                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("LumenTact", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

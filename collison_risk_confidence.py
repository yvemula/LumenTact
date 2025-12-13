"""
Collision Risk Scoring Module for LumenTact
Advanced risk assessment system that calculates time-to-collision,
priority scoring, and multi-hazard decision making for navigation.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Configuration
# -------------------------------
YOLO_MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5

# Risk Factors and Weights
PRIORITY_WEIGHTS = {
    'person': 10.0,
    'bicycle': 9.0,
    'car': 9.5,
    'motorcycle': 9.0,
    'bus': 8.0,
    'truck': 8.5,
    'traffic light': 3.0,
    'stop sign': 7.0,
    'bench': 4.0,
    'chair': 5.0,
    'couch': 5.0,
    'potted plant': 4.0,
    'dining table': 6.0,
    'tv': 3.0,
    'laptop': 2.0,
    'cell phone': 1.0,
    'bottle': 2.0,
    'cup': 2.0,
    'knife': 8.0,
}

# Risk calculation parameters
PROXIMITY_WEIGHT = 0.35
VELOCITY_WEIGHT = 0.25
SIZE_WEIGHT = 0.15
CENTER_ALIGNMENT_WEIGHT = 0.15
PRIORITY_WEIGHT = 0.10

# Distance thresholds (normalized 0-1)
CRITICAL_DISTANCE = 0.15
WARNING_DISTANCE = 0.30
CAUTION_DISTANCE = 0.50

# Velocity thresholds (pixels per frame)
HIGH_VELOCITY = 15.0
MEDIUM_VELOCITY = 7.0

# Time-to-collision estimation
ASSUMED_USER_SPEED = 1.2  # meters per second
PIXEL_TO_METER_RATIO = 100  # approximate

# Tracking parameters
TRACKING_HISTORY = 20
IOU_THRESHOLD = 0.3

# Decision thresholds
STOP_THRESHOLD = 0.75
VEER_THRESHOLD = 0.50
CAUTION_THRESHOLD = 0.30

# -------------------------------
# Data Structures
# -------------------------------
@dataclass
class RiskScore:
    """Container for risk assessment components"""
    total: float
    proximity: float
    velocity: float
    size: float
    alignment: float
    priority: float
    time_to_collision: float
    risk_level: str
    recommended_action: str

@dataclass
class TrackedObject:
    """Represents a tracked object with risk assessment"""
    obj_id: int
    cls_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    frame_id: int
    first_seen: int
    positions: deque = field(default_factory=lambda: deque(maxlen=TRACKING_HISTORY))
    velocities: deque = field(default_factory=lambda: deque(maxlen=TRACKING_HISTORY))
    risk_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def __post_init__(self):
        self.positions.append(self.get_center())
        self.velocities.append((0.0, 0.0))
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_size(self) -> float:
        """Get normalized size (area) of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def update(self, bbox: Tuple, confidence: float, frame_id: int):
        """Update object with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.frame_id = frame_id
        
        # Update position history
        curr_center = self.get_center()
        self.positions.append(curr_center)
        
        # Calculate velocity
        if len(self.positions) >= 2:
            prev_center = self.positions[-2]
            velocity = (curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
            self.velocities.append(velocity)
    
    def get_average_velocity(self) -> Tuple[float, float]:
        """Get smoothed velocity vector"""
        if not self.velocities:
            return (0.0, 0.0)
        vx = np.mean([v[0] for v in self.velocities])
        vy = np.mean([v[1] for v in self.velocities])
        return (vx, vy)
    
    def get_velocity_magnitude(self) -> float:
        """Get speed (magnitude of velocity)"""
        vx, vy = self.get_average_velocity()
        return np.sqrt(vx**2 + vy**2)
    
    def is_approaching(self) -> bool:
        """Check if object is moving towards user (downward in frame)"""
        _, vy = self.get_average_velocity()
        return vy > 2.0  # Moving down towards bottom of frame

# -------------------------------
# Object Tracker
# -------------------------------
class RiskTracker:
    """Manages tracked objects and risk assessment"""
    def __init__(self):
        self.objects: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.frame_id = 0
        self.frame_width = 640
        self.frame_height = 480
    
    def update(self, detections: List[Tuple], frame_shape: Tuple[int, int]):
        """Update tracker with new detections"""
        self.frame_id += 1
        self.frame_height, self.frame_width = frame_shape[:2]
        
        current_ids = set()
        
        for det in detections:
            bbox, cls_name, conf = det
            
            # Try to match with existing object
            matched_id = self._match_detection(bbox)
            
            if matched_id is not None:
                self.objects[matched_id].update(bbox, conf, self.frame_id)
                current_ids.add(matched_id)
            else:
                # Create new tracked object
                new_obj = TrackedObject(
                    obj_id=self.next_id,
                    cls_name=cls_name,
                    bbox=bbox,
                    confidence=conf,
                    frame_id=self.frame_id,
                    first_seen=self.frame_id
                )
                self.objects[self.next_id] = new_obj
                current_ids.add(self.next_id)
                self.next_id += 1
        
        # Remove stale objects
        stale_ids = [oid for oid, obj in self.objects.items() 
                     if self.frame_id - obj.frame_id > 30]
        for oid in stale_ids:
            del self.objects[oid]
    
    def _match_detection(self, bbox: Tuple) -> int:
        """Match detection to existing object using IoU"""
        best_iou = IOU_THRESHOLD
        best_id = None
        
        for oid, obj in self.objects.items():
            iou = self._calculate_iou(bbox, obj.bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = oid
        
        return best_id
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_risk_score(self, obj: TrackedObject) -> RiskScore:
        """Calculate comprehensive risk score for an object"""
        # 1. Proximity Score (normalized distance from bottom center)
        center_x, center_y = obj.get_center()
        bottom_y = obj.bbox[3]
        
        # Distance from bottom center of frame
        frame_center_x = self.frame_width / 2
        frame_bottom_y = self.frame_height
        
        horizontal_dist = abs(center_x - frame_center_x) / (self.frame_width / 2)
        vertical_dist = (frame_bottom_y - bottom_y) / frame_bottom_y
        
        proximity_score = 1.0 - vertical_dist  # Closer = higher score
        
        # 2. Velocity Score
        velocity_mag = obj.get_velocity_magnitude()
        if velocity_mag > HIGH_VELOCITY:
            velocity_score = 1.0
        elif velocity_mag > MEDIUM_VELOCITY:
            velocity_score = 0.6
        else:
            velocity_score = 0.3
        
        # Boost if approaching
        if obj.is_approaching():
            velocity_score *= 1.5
        
        velocity_score = min(velocity_score, 1.0)
        
        # 3. Size Score (larger objects = higher risk)
        obj_size = obj.get_size()
        max_size = self.frame_width * self.frame_height
        size_score = min(obj_size / (max_size * 0.3), 1.0)
        
        # 4. Center Alignment Score (objects in center path = higher risk)
        alignment_score = 1.0 - horizontal_dist
        
        # 5. Priority Score (class-based)
        priority_value = PRIORITY_WEIGHTS.get(obj.cls_name, 5.0)
        priority_score = priority_value / 10.0
        
        # Calculate weighted total
        total_score = (
            proximity_score * PROXIMITY_WEIGHT +
            velocity_score * VELOCITY_WEIGHT +
            size_score * SIZE_WEIGHT +
            alignment_score * CENTER_ALIGNMENT_WEIGHT +
            priority_score * PRIORITY_WEIGHT
        )
        
        # Estimate time to collision
        ttc = self._estimate_time_to_collision(obj, vertical_dist, velocity_mag)
        
        # Determine risk level
        if total_score > STOP_THRESHOLD:
            risk_level = "CRITICAL"
            action = "STOP"
        elif total_score > VEER_THRESHOLD:
            risk_level = "HIGH"
            action = "VEER_LEFT" if center_x < frame_center_x else "VEER_RIGHT"
        elif total_score > CAUTION_THRESHOLD:
            risk_level = "MEDIUM"
            action = "SLOW_DOWN"
        else:
            risk_level = "LOW"
            action = "PROCEED"
        
        return RiskScore(
            total=total_score,
            proximity=proximity_score,
            velocity=velocity_score,
            size=size_score,
            alignment=alignment_score,
            priority=priority_score,
            time_to_collision=ttc,
            risk_level=risk_level,
            recommended_action=action
        )
    
    def _estimate_time_to_collision(self, obj: TrackedObject, 
                                    vertical_dist: float, velocity_mag: float) -> float:
        """Estimate time until collision in seconds"""
        if velocity_mag < 1.0:
            return float('inf')
        
        # Convert pixel distance to meters (approximate)
        distance_meters = vertical_dist * self.frame_height / PIXEL_TO_METER_RATIO
        
        # Relative velocity (user + object)
        object_velocity = velocity_mag / 30.0  # Assume 30 FPS
        relative_velocity = ASSUMED_USER_SPEED + object_velocity
        
        if relative_velocity <= 0:
            return float('inf')
        
        ttc = distance_meters / relative_velocity
        return max(ttc, 0.1)
    
    def get_all_objects(self) -> List[TrackedObject]:
        """Get all active tracked objects"""
        return list(self.objects.values())
    
    def get_highest_risk_objects(self, n: int = 3) -> List[Tuple[TrackedObject, RiskScore]]:
        """Get top N highest risk objects"""
        objects_with_risk = [(obj, self.calculate_risk_score(obj)) 
                            for obj in self.objects.values()]
        objects_with_risk.sort(key=lambda x: x[1].total, reverse=True)
        return objects_with_risk[:n]
    
    def get_recommended_action(self) -> Tuple[str, float, List[str]]:
        """Get overall recommended action based on all risks"""
        if not self.objects:
            return "PROCEED", 0.0, []
        
        # Get all risk scores
        risk_scores = [self.calculate_risk_score(obj) for obj in self.objects.values()]
        
        # Find maximum risk
        max_risk = max(risk_scores, key=lambda x: x.total)
        
        # Collect all high-priority warnings
        warnings = []
        for obj, score in zip(self.objects.values(), risk_scores):
            if score.total > CAUTION_THRESHOLD:
                warnings.append(f"{obj.cls_name} ({score.risk_level})")
        
        return max_risk.recommended_action, max_risk.total, warnings

# -------------------------------
# Visualization Functions
# -------------------------------
def get_risk_color(risk_score: float) -> Tuple[int, int, int]:
    """Get color based on risk score"""
    if risk_score > STOP_THRESHOLD:
        return (0, 0, 255)  # Red
    elif risk_score > VEER_THRESHOLD:
        return (0, 165, 255)  # Orange
    elif risk_score > CAUTION_THRESHOLD:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 255, 0)  # Green

def draw_risk_assessment(frame, obj: TrackedObject, risk: RiskScore):
    """Draw comprehensive risk visualization"""
    x1, y1, x2, y2 = map(int, obj.bbox)
    color = get_risk_color(risk.total)
    
    # Draw bounding box with thickness based on risk
    thickness = int(2 + risk.total * 4)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw velocity vector
    if obj.get_velocity_magnitude() > 1.0:
        vx, vy = obj.get_average_velocity()
        center = obj.get_center()
        end_point = (int(center[0] + vx * 3), int(center[1] + vy * 3))
        cv2.arrowedLine(frame, tuple(map(int, center)), end_point, color, 2, tipLength=0.3)
    
    # Risk bar
    bar_width = x2 - x1
    bar_height = 8
    bar_fill = int(bar_width * risk.total)
    cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 5 + bar_height), (100, 100, 100), -1)
    cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_fill, y2 + 5 + bar_height), color, -1)
    
    # Labels
    label1 = f"{obj.cls_name} [{risk.risk_level}]"
    label2 = f"Risk: {risk.total:.2f} | TTC: {risk.time_to_collision:.1f}s"
    
    cv2.rectangle(frame, (x1, y1 - 45), (x2, y1), (0, 0, 0), -1)
    cv2.putText(frame, label1, (x1 + 5, y1 - 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, label2, (x1 + 5, y1 - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def draw_risk_panel(frame, tracker: RiskTracker):
    """Draw comprehensive risk dashboard"""
    h, w = frame.shape[:2]
    
    # Get recommended action
    action, risk_level, warnings = tracker.get_recommended_action()
    action_color = get_risk_color(risk_level)
    
    # Main action panel
    panel_h = 120
    cv2.rectangle(frame, (10, h - panel_h - 10), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, h - panel_h - 10), (w - 10, h - 10), action_color, 3)
    
    # Action text
    cv2.putText(frame, f"RECOMMENDED ACTION: {action}", (20, h - panel_h + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2)
    cv2.putText(frame, f"Overall Risk Level: {risk_level:.2f}", (20, h - panel_h + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Warnings
    if warnings:
        warning_text = "Active Hazards: " + ", ".join(warnings[:3])
        cv2.putText(frame, warning_text, (20, h - panel_h + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Top risks sidebar
    top_risks = tracker.get_highest_risk_objects(3)
    sidebar_x = w - 250
    sidebar_y = 50
    
    cv2.rectangle(frame, (sidebar_x - 10, 10), (w - 10, 200), (0, 0, 0), -1)
    cv2.putText(frame, "TOP RISKS:", (sidebar_x, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    for i, (obj, risk) in enumerate(top_risks):
        y_pos = sidebar_y + i * 45
        risk_color = get_risk_color(risk.total)
        
        cv2.putText(frame, f"{i+1}. {obj.cls_name}", (sidebar_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 1)
        cv2.putText(frame, f"   Risk: {risk.total:.2f}", (sidebar_x, y_pos + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"   TTC: {risk.time_to_collision:.1f}s", (sidebar_x, y_pos + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

# -------------------------------
# Main Processing Loop
# -------------------------------
def main():
    print("[INFO] Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    tracker = RiskTracker()
    prev_time = time.time()
    frame_count = 0
    
    print("[INFO] Starting collision risk scoring system...")
    print("[INFO] Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Run YOLO detection
        results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append(((x1, y1, x2, y2), cls_name, conf))
        
        # Update tracker
        tracker.update(detections, frame.shape)
        
        # Draw all tracked objects with risk
        output = frame.copy()
        for obj in tracker.get_all_objects():
            risk = tracker.calculate_risk_score(obj)
            draw_risk_assessment(output, obj, risk)
        
        # Draw risk panel
        draw_risk_panel(output, tracker)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time
        
        # FPS counter
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("LumenTact - Collision Risk Scorer", output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Risk scoring system terminated")
    print(f"[STATS] Total frames: {frame_count}")
    print(f"[STATS] Total objects tracked: {tracker.next_id}")

if __name__ == "__main__":
    main()
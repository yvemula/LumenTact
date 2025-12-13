"""
Overhead Hazard Detection Module for LumenTact
Focuses on upper frame region to detect hanging obstacles like branches,
signs, awnings, and other overhead hazards that require ducking.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Configuration
# -------------------------------
YOLO_MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.45

# Overhead Hazard Classes
OVERHEAD_CLASSES = [
    "person",           # People's heads/arms
    "umbrella",         # Umbrellas
    "handbag",          # Hanging bags
    "backpack",         # Backpacks
    "kite",             # Flying objects
    "sports ball",      # Balls
    "traffic light",    # Traffic signals
    "stop sign",        # Signs
    "potted plant",     # Hanging plants
    "clock",            # Wall clocks
    "vase",             # Hanging decorations
]

# Frame Division (focus on upper regions)
CRITICAL_OVERHEAD_RATIO = 0.35  # Top 35% is critical
WARNING_OVERHEAD_RATIO = 0.55   # 35-55% is warning
SAFE_OVERHEAD_RATIO = 0.70      # 55-70% is caution

# Clearance Zones (in pixels from top)
CLEARANCE_CRITICAL = 100   # Must duck immediately
CLEARANCE_WARNING = 200    # Prepare to duck
CLEARANCE_CAUTION = 300    # Be aware

# Tracking Parameters
TRACKING_BUFFER_SIZE = 15
VELOCITY_THRESHOLD = 5.0  # pixels per frame for "approaching" detection
ALERT_COOLDOWN = 3.0      # seconds between alerts

# Edge Detection for Unknown Objects
EDGE_DENSITY_THRESHOLD = 0.15  # Ratio of edges in overhead region
CONTOUR_AREA_THRESHOLD = 500   # Minimum area for unknown obstacles

# Visualization
SHOW_ZONES = True
SHOW_TRAJECTORIES = True
SHOW_EDGE_DETECTION = False
ALPHA_OVERLAY = 0.3

# -------------------------------
# Hazard Tracking Classes
# -------------------------------
class OverheadHazard:
    """Represents a tracked overhead hazard"""
    def __init__(self, bbox, cls_name, confidence, frame_id):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.cls_name = cls_name
        self.confidence = confidence
        self.first_seen = frame_id
        self.last_seen = frame_id
        self.positions = deque(maxlen=TRACKING_BUFFER_SIZE)
        self.positions.append(self.get_center())
        self.velocity = (0, 0)
        self.is_approaching = False
    
    def get_center(self):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_bottom_y(self):
        """Get bottom edge of object (closest to user's head)"""
        return self.bbox[3]
    
    def update(self, bbox, confidence, frame_id):
        """Update hazard with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = frame_id
        self.positions.append(self.get_center())
        
        # Calculate velocity
        if len(self.positions) >= 2:
            curr_pos = self.positions[-1]
            prev_pos = self.positions[-2]
            self.velocity = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            
            # Check if approaching (moving down towards user)
            self.is_approaching = self.velocity[1] > VELOCITY_THRESHOLD
    
    def get_clearance_zone(self, frame_height):
        """Determine which clearance zone this hazard is in"""
        bottom_y = self.get_bottom_y()
        
        if bottom_y < frame_height * CRITICAL_OVERHEAD_RATIO:
            return "CRITICAL", (0, 0, 255), "DUCK NOW!"
        elif bottom_y < frame_height * WARNING_OVERHEAD_RATIO:
            return "WARNING", (0, 165, 255), "Prepare to Duck"
        elif bottom_y < frame_height * SAFE_OVERHEAD_RATIO:
            return "CAUTION", (0, 255, 255), "Overhead Hazard"
        else:
            return "SAFE", (0, 255, 0), "Clear Above"
    
    def get_trajectory_points(self):
        """Get list of position points for drawing trajectory"""
        return list(self.positions)

class HazardTracker:
    """Manages multiple overhead hazards"""
    def __init__(self):
        self.hazards = {}  # id -> OverheadHazard
        self.next_id = 0
        self.frame_id = 0
        self.last_alert_time = {}
    
    def update(self, detections):
        """Update tracked hazards with new detections"""
        self.frame_id += 1
        current_ids = set()
        
        for det in detections:
            bbox, cls_name, conf = det
            
            # Try to match with existing hazard
            matched_id = self._match_detection(bbox)
            
            if matched_id is not None:
                self.hazards[matched_id].update(bbox, conf, self.frame_id)
                current_ids.add(matched_id)
            else:
                # Create new hazard
                new_hazard = OverheadHazard(bbox, cls_name, conf, self.frame_id)
                self.hazards[self.next_id] = new_hazard
                current_ids.add(self.next_id)
                self.next_id += 1
        
        # Remove stale hazards (not seen for 30 frames)
        stale_ids = [hid for hid, hazard in self.hazards.items() 
                     if self.frame_id - hazard.last_seen > 30]
        for hid in stale_ids:
            del self.hazards[hid]
    
    def _match_detection(self, bbox):
        """Match detection to existing hazard using IoU"""
        best_iou = 0.3  # Minimum IoU threshold
        best_id = None
        
        for hid, hazard in self.hazards.items():
            iou = self._calculate_iou(bbox, hazard.bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = hid
        
        return best_id
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_all_hazards(self):
        """Get list of all active hazards"""
        return list(self.hazards.values())
    
    def get_critical_hazards(self, frame_height):
        """Get hazards in critical zone"""
        critical = []
        for hazard in self.hazards.values():
            zone, _, _ = hazard.get_clearance_zone(frame_height)
            if zone == "CRITICAL":
                critical.append(hazard)
        return critical
    
    def should_alert(self, hazard_id):
        """Check if enough time has passed since last alert"""
        current_time = time.time()
        if hazard_id not in self.last_alert_time:
            self.last_alert_time[hazard_id] = current_time
            return True
        
        if current_time - self.last_alert_time[hazard_id] > ALERT_COOLDOWN:
            self.last_alert_time[hazard_id] = current_time
            return True
        
        return False

# -------------------------------
# Edge-Based Unknown Object Detection
# -------------------------------
def detect_unknown_overheads(frame, roi_y1, roi_y2):
    """Detect overhead objects using edge density and contours"""
    roi = frame[roi_y1:roi_y2, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Calculate edge density
    edge_density = np.count_nonzero(edges) / edges.size
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    unknown_objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > CONTOUR_AREA_THRESHOLD:
            x, y, w, h = cv2.boundingRect(contour)
            # Adjust coordinates to full frame
            unknown_objects.append({
                'bbox': (x, y + roi_y1, x + w, y + h + roi_y1),
                'area': area,
                'confidence': min(area / 5000.0, 1.0)
            })
    
    return unknown_objects, edge_density

# -------------------------------
# Visualization Functions
# -------------------------------
def draw_clearance_zones(frame):
    """Draw overhead clearance zones"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Critical zone (red)
    critical_y = int(h * CRITICAL_OVERHEAD_RATIO)
    cv2.rectangle(overlay, (0, 0), (w, critical_y), (0, 0, 255), -1)
    
    # Warning zone (orange)
    warning_y = int(h * WARNING_OVERHEAD_RATIO)
    cv2.rectangle(overlay, (0, critical_y), (w, warning_y), (0, 165, 255), -1)
    
    # Caution zone (yellow)
    caution_y = int(h * SAFE_OVERHEAD_RATIO)
    cv2.rectangle(overlay, (0, warning_y), (w, caution_y), (0, 255, 255), -1)
    
    # Blend
    cv2.addWeighted(overlay, ALPHA_OVERLAY, frame, 1 - ALPHA_OVERLAY, 0, frame)
    
    # Zone labels
    cv2.putText(frame, "CRITICAL ZONE - DUCK!", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "WARNING ZONE", (10, critical_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "CAUTION ZONE", (10, warning_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_hazard(frame, hazard, show_trajectory=True):
    """Draw a single hazard with all annotations"""
    x1, y1, x2, y2 = map(int, hazard.bbox)
    
    # Get zone info
    zone, color, message = hazard.get_clearance_zone(frame.shape[0])
    
    # Draw bounding box (thicker for critical)
    thickness = 4 if zone == "CRITICAL" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw trajectory
    if show_trajectory and len(hazard.positions) > 1:
        points = hazard.get_trajectory_points()
        for i in range(len(points) - 1):
            pt1 = tuple(map(int, points[i]))
            pt2 = tuple(map(int, points[i + 1]))
            cv2.line(frame, pt1, pt2, color, 2)
    
    # Draw velocity arrow if approaching
    if hazard.is_approaching:
        center = tuple(map(int, hazard.get_center()))
        arrow_end = (center[0], center[1] + 30)
        cv2.arrowedLine(frame, center, arrow_end, (0, 0, 255), 3, tipLength=0.3)
        cv2.putText(frame, "APPROACHING!", (center[0] - 50, center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Label
    label = f"{hazard.cls_name} [{zone}]"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # Background for label
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                 (x1 + label_size[0], y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Clearance distance (bottom of object to bottom of frame)
    clearance = frame.shape[0] - y2
    cv2.putText(frame, f"{clearance}px clearance", (x1, y2 + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_unknown_objects(frame, unknown_objects):
    """Draw detected unknown overhead objects"""
    for obj in unknown_objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        conf = obj['confidence']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, f"Unknown {conf:.2f}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

# -------------------------------
# Alert System
# -------------------------------
def generate_alerts(tracker, frame_height):
    """Generate alert messages for critical hazards"""
    alerts = []
    critical_hazards = tracker.get_critical_hazards(frame_height)
    
    for hazard in critical_hazards:
        zone, color, message = hazard.get_clearance_zone(frame_height)
        if zone == "CRITICAL":
            alerts.append({
                'message': f"{message} - {hazard.cls_name}",
                'color': color,
                'urgency': 'HIGH'
            })
    
    return alerts

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
    
    tracker = HazardTracker()
    prev_time = time.time()
    frame_count = 0
    
    print("[INFO] Starting overhead hazard detection system...")
    print("[INFO] Press 'q' to quit, 'z' to toggle zones, 't' to toggle trajectories, 'e' to toggle edges")
    
    show_zones = SHOW_ZONES
    show_trajectories = SHOW_TRAJECTORIES
    show_edges = SHOW_EDGE_DETECTION
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Define overhead ROI (upper 70% of frame)
        roi_y1 = 0
        roi_y2 = int(h * SAFE_OVERHEAD_RATIO)
        
        # Run YOLO on full frame
        results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]
            
            if cls_name in OVERHEAD_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                # Only track if in overhead region
                if y2 < roi_y2:
                    detections.append(((x1, y1, x2, y2), cls_name, conf))
        
        # Update tracker
        tracker.update(detections)
        
        # Detect unknown overhead objects
        unknown_objects, edge_density = detect_unknown_overheads(frame, roi_y1, roi_y2)
        
        # Create output frame
        output = frame.copy()
        
        # Draw clearance zones
        if show_zones:
            draw_clearance_zones(output)
        
        # Draw all tracked hazards
        for hazard in tracker.get_all_hazards():
            draw_hazard(output, hazard, show_trajectories)
        
        # Draw unknown objects
        if show_edges:
            draw_unknown_objects(output, unknown_objects)
        
        # Generate alerts
        alerts = generate_alerts(tracker, h)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time
        
        # Draw main info panel
        info_y = h - 150
        cv2.rectangle(output, (10, info_y - 30), (400, h - 10), (0, 0, 0), -1)
        info_y_start = info_y
        cv2.putText(output, f"FPS: {fps:.1f}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(output, f"Tracked Hazards: {len(tracker.get_all_hazards())}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(output, f"Critical Hazards: {len(tracker.get_critical_hazards(h))}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        info_y += 30
        cv2.putText(output, f"Edge Density: {edge_density:.3f}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(output, f"Unknown Objects: {len(unknown_objects)}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw alerts
        if alerts:
            alert_y = 100
            for alert in alerts:
                cv2.rectangle(output, (w - 310, alert_y - 25), (w - 10, alert_y + 10), 
                            alert['color'], -1)
                cv2.putText(output, alert['message'], (w - 300, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                alert_y += 40
        
        # Display
        cv2.imshow("LumenTact - Overhead Hazard Detection", output)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            show_zones = not show_zones
            print(f"[INFO] Zone overlay: {'ON' if show_zones else 'OFF'}")
        elif key == ord('t'):
            show_trajectories = not show_trajectories
            print(f"[INFO] Trajectories: {'ON' if show_trajectories else 'OFF'}")
        elif key == ord('e'):
            show_edges = not show_edges
            print(f"[INFO] Edge detection: {'ON' if show_edges else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Overhead hazard detection system terminated")
    print(f"[STATS] Total frames processed: {frame_count}")
    print(f"[STATS] Total hazards tracked: {tracker.next_id}")

if __name__ == "__main__":
    main()
"""
Curb and Stair Detection Module for LumenTact
Combines YOLO object detection with edge detection and line analysis
to identify ground-level hazards like curbs, stairs, and elevation changes.
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
CONFIDENCE_THRESHOLD = 0.4

# Edge Detection Parameters
CANNY_LOW = 50
CANNY_HIGH = 150
BLUR_KERNEL = 5

# Hough Transform Parameters
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 50
HOUGH_MAX_LINE_GAP = 10

# Line Classification Thresholds
HORIZONTAL_ANGLE_THRESHOLD = 15  # degrees from horizontal
VERTICAL_ANGLE_THRESHOLD = 75    # degrees from vertical
STAIR_SPACING_THRESHOLD = 30     # pixels between parallel lines
CURB_HEIGHT_THRESHOLD = 20       # pixels minimum height

# Region of Interest (focus on lower portion of frame)
ROI_TOP_RATIO = 0.4      # Start ROI at 40% from top
ROI_BOTTOM_RATIO = 1.0   # End at bottom

# Smoothing and Stability
DETECTION_BUFFER_SIZE = 10
CONFIDENCE_SMOOTHING = 0.7

# Alert Zones
IMMEDIATE_ZONE = 0.7  # Bottom 30% of frame
WARNING_ZONE = 0.5    # Middle 20% of frame

# Visualization
SHOW_EDGES = True
SHOW_LINES = True
SHOW_ROI = True

# -------------------------------
# Line Analysis Functions
# -------------------------------
def calculate_line_angle(x1, y1, x2, y2):
    """Calculate angle of line in degrees (0-180)"""
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)

def is_horizontal_line(x1, y1, x2, y2):
    """Check if line is approximately horizontal"""
    angle = calculate_line_angle(x1, y1, x2, y2)
    return angle < HORIZONTAL_ANGLE_THRESHOLD or angle > (180 - HORIZONTAL_ANGLE_THRESHOLD)

def is_vertical_line(x1, y1, x2, y2):
    """Check if line is approximately vertical"""
    angle = calculate_line_angle(x1, y1, x2, y2)
    return VERTICAL_ANGLE_THRESHOLD < angle < (180 - VERTICAL_ANGLE_THRESHOLD)

def line_length(x1, y1, x2, y2):
    """Calculate Euclidean length of line"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def merge_nearby_lines(lines, distance_threshold=20):
    """Merge lines that are close to each other"""
    if lines is None or len(lines) == 0:
        return []
    
    merged = []
    used = set()
    
    for i, line1 in enumerate(lines):
        if i in used:
            continue
        
        x1, y1, x2, y2 = line1
        merged_line = [x1, y1, x2, y2]
        used.add(i)
        
        for j, line2 in enumerate(lines[i+1:], start=i+1):
            if j in used:
                continue
            
            x3, y3, x4, y4 = line2
            
            # Check if lines are close
            dist = min(
                np.sqrt((x1-x3)**2 + (y1-y3)**2),
                np.sqrt((x1-x4)**2 + (y1-y4)**2),
                np.sqrt((x2-x3)**2 + (y2-y3)**2),
                np.sqrt((x2-x4)**2 + (y2-y4)**2)
            )
            
            if dist < distance_threshold:
                merged_line = [
                    min(merged_line[0], x3),
                    min(merged_line[1], y3),
                    max(merged_line[2], x4),
                    max(merged_line[3], y4)
                ]
                used.add(j)
        
        merged.append(merged_line)
    
    return merged

def find_parallel_horizontal_lines(lines, spacing_threshold=STAIR_SPACING_THRESHOLD):
    """Find groups of parallel horizontal lines (potential stairs)"""
    if not lines:
        return []
    
    horizontal_lines = [l for l in lines if is_horizontal_line(*l)]
    horizontal_lines.sort(key=lambda l: (l[1] + l[3]) / 2)  # Sort by y-coordinate
    
    stair_groups = []
    current_group = []
    
    for i, line in enumerate(horizontal_lines):
        if not current_group:
            current_group.append(line)
        else:
            prev_y = (current_group[-1][1] + current_group[-1][3]) / 2
            curr_y = (line[1] + line[3]) / 2
            spacing = curr_y - prev_y
            
            if spacing_threshold * 0.5 < spacing < spacing_threshold * 2:
                current_group.append(line)
            else:
                if len(current_group) >= 3:  # At least 3 lines for stairs
                    stair_groups.append(current_group)
                current_group = [line]
        
        if i == len(horizontal_lines) - 1 and len(current_group) >= 3:
            stair_groups.append(current_group)
    
    return stair_groups

def detect_curb_edge(edges, lines):
    """Detect sharp elevation changes (curbs)"""
    curbs = []
    
    for line in lines:
        x1, y1, x2, y2 = line
        
        if is_horizontal_line(x1, y1, x2, y2):
            # Check for strong vertical edge below horizontal line
            search_height = min(CURB_HEIGHT_THRESHOLD * 2, edges.shape[0] - int(y1))
            roi_y1, roi_y2 = int(y1), int(y1) + search_height
            roi_x1, roi_x2 = int(min(x1, x2)), int(max(x1, x2))
            
            if roi_y2 < edges.shape[0] and roi_x2 < edges.shape[1]:
                roi = edges[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # Sum vertical edges in ROI
                vertical_intensity = np.sum(roi, axis=0)
                if np.max(vertical_intensity) > CURB_HEIGHT_THRESHOLD * 10:
                    curbs.append({
                        'line': line,
                        'confidence': min(np.max(vertical_intensity) / 1000.0, 1.0),
                        'position': (x1 + x2) / 2
                    })
    
    return curbs

# -------------------------------
# Detection History Management
# -------------------------------
class DetectionHistory:
    def __init__(self, buffer_size=DETECTION_BUFFER_SIZE):
        self.stair_detections = deque(maxlen=buffer_size)
        self.curb_detections = deque(maxlen=buffer_size)
        self.edge_density = deque(maxlen=buffer_size)
    
    def update(self, has_stairs, has_curb, edge_count):
        self.stair_detections.append(1 if has_stairs else 0)
        self.curb_detections.append(1 if has_curb else 0)
        self.edge_density.append(edge_count)
    
    def get_stair_confidence(self):
        if not self.stair_detections:
            return 0.0
        return sum(self.stair_detections) / len(self.stair_detections)
    
    def get_curb_confidence(self):
        if not self.curb_detections:
            return 0.0
        return sum(self.curb_detections) / len(self.curb_detections)
    
    def get_avg_edge_density(self):
        if not self.edge_density:
            return 0
        return np.mean(self.edge_density)

# -------------------------------
# Hazard Assessment
# -------------------------------
def assess_hazard_level(detection_y, frame_height):
    """Determine hazard urgency based on vertical position"""
    normalized_y = detection_y / frame_height
    
    if normalized_y > IMMEDIATE_ZONE:
        return "IMMEDIATE", (0, 0, 255)  # Red
    elif normalized_y > WARNING_ZONE:
        return "WARNING", (0, 165, 255)  # Orange
    else:
        return "ADVANCE_NOTICE", (0, 255, 255)  # Yellow

# -------------------------------
# Visualization Functions
# -------------------------------
def draw_roi(frame, roi_coords):
    """Draw region of interest overlay"""
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                 (roi_coords[0], roi_coords[1]), 
                 (roi_coords[2], roi_coords[3]), 
                 (255, 255, 0), 2)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

def draw_stair_detection(frame, stair_groups):
    """Draw detected stairs with step counting"""
    for i, group in enumerate(stair_groups):
        # Calculate bounding box for stair group
        all_x = [x for line in group for x in [line[0], line[2]]]
        all_y = [y for line in group for y in [line[1], line[3]]]
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x_min) - 10, int(y_min) - 10),
                     (int(x_max) + 10, int(y_max) + 10), (255, 0, 255), 3)
        
        # Draw individual step lines
        for line in group:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        # Label with step count
        num_steps = len(group)
        label = f"STAIRS: {num_steps} steps"
        y_pos = int(y_min) - 20
        
        # Assess hazard level
        hazard, color = assess_hazard_level(y_max, frame.shape[0])
        
        cv2.putText(frame, label, (int(x_min), y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"[{hazard}]", (int(x_min), y_pos - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_curb_detection(frame, curbs):
    """Draw detected curbs with confidence"""
    for curb in curbs:
        x1, y1, x2, y2 = map(int, curb['line'])
        conf = curb['confidence']
        
        # Draw curb line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw confidence indicator
        center_x = int((x1 + x2) / 2)
        cv2.circle(frame, (center_x, y1), 8, (0, 255, 0), -1)
        
        # Assess hazard level
        hazard, color = assess_hazard_level(y1, frame.shape[0])
        
        # Label
        label = f"CURB {conf:.2f}"
        cv2.putText(frame, label, (center_x - 50, y1 - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"[{hazard}]", (center_x - 50, y1 - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# -------------------------------
# Main Processing
# -------------------------------
def main():
    print("[INFO] Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    history = DetectionHistory()
    prev_time = time.time()
    frame_count = 0
    
    print("[INFO] Starting curb and stair detection system...")
    print("[INFO] Press 'q' to quit, 'e' to toggle edges, 'l' to toggle lines, 'r' to toggle ROI")
    
    show_edges_flag = SHOW_EDGES
    show_lines_flag = SHOW_LINES
    show_roi_flag = SHOW_ROI
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Define ROI (lower portion of frame)
        roi_y1 = int(h * ROI_TOP_RATIO)
        roi_y2 = int(h * ROI_BOTTOM_RATIO)
        roi_x1 = 0
        roi_x2 = w
        
        # Extract ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESHOLD,
                                minLineLength=HOUGH_MIN_LINE_LENGTH,
                                maxLineGap=HOUGH_MAX_LINE_GAP)
        
        # Process lines
        processed_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Adjust coordinates to full frame
                processed_lines.append([x1, y1 + roi_y1, x2, y2 + roi_y1])
        
        # Merge nearby lines
        merged_lines = merge_nearby_lines(processed_lines)
        
        # Detect stairs (parallel horizontal lines)
        stair_groups = find_parallel_horizontal_lines(merged_lines)
        
        # Detect curbs
        curbs = detect_curb_edge(edges, merged_lines)
        
        # Update detection history
        has_stairs = len(stair_groups) > 0
        has_curbs = len(curbs) > 0
        edge_count = np.count_nonzero(edges)
        history.update(has_stairs, has_curbs, edge_count)
        
        # Create output frame
        output = frame.copy()
        
        # Draw ROI
        if show_roi_flag:
            draw_roi(output, (roi_x1, roi_y1, roi_x2, roi_y2))
        
        # Draw edges (optional)
        if show_edges_flag:
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_full = np.zeros_like(frame)
            edges_full[roi_y1:roi_y2, roi_x1:roi_x2] = edges_colored
            output = cv2.addWeighted(output, 0.7, edges_full, 0.3, 0)
        
        # Draw detected lines (optional)
        if show_lines_flag and merged_lines:
            for line in merged_lines:
                x1, y1, x2, y2 = map(int, line)
                if is_horizontal_line(x1, y1, x2, y2):
                    color = (255, 255, 0)  # Cyan for horizontal
                elif is_vertical_line(x1, y1, x2, y2):
                    color = (255, 0, 255)  # Magenta for vertical
                else:
                    color = (128, 128, 128)  # Gray for diagonal
                cv2.line(output, (x1, y1), (x2, y2), color, 1)
        
        # Draw stair detections
        if stair_groups:
            draw_stair_detection(output, stair_groups)
        
        # Draw curb detections
        if curbs:
            draw_curb_detection(output, curbs)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time
        
        # Draw info panel
        info_y = 30
        cv2.rectangle(output, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.putText(output, f"FPS: {fps:.1f}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(output, f"Stairs Detected: {len(stair_groups)}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        info_y += 30
        cv2.putText(output, f"Stair Confidence: {history.get_stair_confidence():.2f}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(output, f"Curbs Detected: {len(curbs)}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += 30
        cv2.putText(output, f"Curb Confidence: {history.get_curb_confidence():.2f}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(output, f"Edges: {edge_count}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("LumenTact - Curb & Stair Detection", output)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            show_edges_flag = not show_edges_flag
            print(f"[INFO] Edge overlay: {'ON' if show_edges_flag else 'OFF'}")
        elif key == ord('l'):
            show_lines_flag = not show_lines_flag
            print(f"[INFO] Line detection: {'ON' if show_lines_flag else 'OFF'}")
        elif key == ord('r'):
            show_roi_flag = not show_roi_flag
            print(f"[INFO] ROI display: {'ON' if show_roi_flag else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Curb and stair detection system terminated")
    print(f"[STATS] Total frames processed: {frame_count}")
    print(f"[STATS] Average stair confidence: {history.get_stair_confidence():.2f}")
    print(f"[STATS] Average curb confidence: {history.get_curb_confidence():.2f}")

if __name__ == "__main__":
    main()
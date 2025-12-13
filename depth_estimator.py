"""
Monocular Depth Estimation Module for LumenTact
Integrates MiDaS depth estimation with YOLO object detection
to provide distance measurements for obstacle avoidance.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Configuration
# -------------------------------
YOLO_MODEL_PATH = 'yolov8n.pt'
MIDAS_MODEL_TYPE = "DPT_Large"  # Options: DPT_Large, DPT_Hybrid, MiDaS_small
CONFIDENCE_THRESHOLD = 0.5
DEPTH_COLORMAP = cv2.COLORMAP_MAGMA

# Distance estimation calibration
FOCAL_LENGTH = 615.0  # Approximate focal length (pixels)
KNOWN_OBJECT_SIZES = {
    'person': 1.7,      # meters (average height)
    'car': 4.5,         # meters (length)
    'chair': 0.5,       # meters (width)
    'bicycle': 1.8,     # meters (length)
    'motorcycle': 2.2,  # meters (length)
    'bus': 12.0,        # meters (length)
    'truck': 8.0,       # meters (length)
}

# Distance zones (meters)
CRITICAL_ZONE = 1.5
WARNING_ZONE = 3.0
SAFE_ZONE = 5.0

# Visualization settings
SHOW_DEPTH_MAP = True
SHOW_3D_BOXES = True
ALPHA_BLEND = 0.6

# -------------------------------
# Initialize Models
# -------------------------------
print("[INFO] Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("[INFO] Loading MiDaS depth estimation model...")
midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if MIDAS_MODEL_TYPE == "DPT_Large" or MIDAS_MODEL_TYPE == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

print(f"[INFO] Using device: {device}")

# -------------------------------
# Depth Processing Functions
# -------------------------------
def estimate_depth(frame):
    """Generate depth map from RGB frame using MiDaS"""
    input_batch = transform(frame).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    return depth_map

def normalize_depth_map(depth_map):
    """Normalize depth map for visualization"""
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized = (depth_map - depth_min) / (depth_max - depth_min)
    return normalized

def depth_to_distance(depth_value, depth_map, bbox_height, object_class):
    """
    Convert relative depth to approximate real-world distance
    Uses multiple estimation methods and averages them
    """
    distances = []
    
    # Method 1: Using known object size
    if object_class in KNOWN_OBJECT_SIZES:
        known_height = KNOWN_OBJECT_SIZES[object_class]
        estimated_distance = (FOCAL_LENGTH * known_height) / bbox_height
        distances.append(estimated_distance)
    
    # Method 2: Inverse depth mapping (normalized)
    depth_normalized = 1.0 - (depth_value / depth_map.max())
    max_depth_range = 10.0  # meters
    estimated_distance = depth_normalized * max_depth_range
    distances.append(estimated_distance)
    
    # Method 3: Relative depth scaling
    avg_depth = depth_map.mean()
    relative_depth = depth_value / avg_depth
    base_distance = 3.0  # meters
    estimated_distance = base_distance / relative_depth if relative_depth > 0 else 10.0
    distances.append(estimated_distance)
    
    # Return weighted average
    if len(distances) > 1:
        return np.mean(distances)
    return distances[0] if distances else 5.0

def get_zone_color(distance):
    """Return color based on distance zone"""
    if distance < CRITICAL_ZONE:
        return (0, 0, 255)  # Red - Critical
    elif distance < WARNING_ZONE:
        return (0, 165, 255)  # Orange - Warning
    elif distance < SAFE_ZONE:
        return (0, 255, 255)  # Yellow - Caution
    else:
        return (0, 255, 0)  # Green - Safe

def draw_3d_box(frame, bbox, distance, color):
    """Draw pseudo-3D bounding box based on distance"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Scale offset based on distance (closer = larger offset)
    offset = int(20 / max(distance, 0.5))
    offset = min(offset, 30)
    
    # Draw back face
    cv2.rectangle(frame, (x1 + offset, y1 - offset), 
                  (x2 + offset, y2 - offset), color, 1)
    
    # Draw front face
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Connect corners
    cv2.line(frame, (x1, y1), (x1 + offset, y1 - offset), color, 1)
    cv2.line(frame, (x2, y1), (x2 + offset, y1 - offset), color, 1)
    cv2.line(frame, (x1, y2), (x1 + offset, y2 - offset), color, 1)
    cv2.line(frame, (x2, y2), (x2 + offset, y2 - offset), color, 1)

# -------------------------------
# Statistics Tracking
# -------------------------------
class DepthStats:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.fps_history = []
        self.detection_counts = []
        self.closest_distances = []
    
    def update(self, fps, num_detections, closest_dist):
        self.fps_history.append(fps)
        self.detection_counts.append(num_detections)
        self.closest_distances.append(closest_dist)
        
        if len(self.fps_history) > self.window_size:
            self.fps_history.pop(0)
            self.detection_counts.pop(0)
            self.closest_distances.pop(0)
    
    def get_avg_fps(self):
        return np.mean(self.fps_history) if self.fps_history else 0
    
    def get_avg_detections(self):
        return np.mean(self.detection_counts) if self.detection_counts else 0
    
    def get_min_distance(self):
        return min(self.closest_distances) if self.closest_distances else float('inf')

# -------------------------------
# Main Processing Loop
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    stats = DepthStats()
    frame_count = 0
    prev_time = time.time()
    
    print("[INFO] Starting depth estimation system...")
    print("[INFO] Press 'q' to quit, 'd' to toggle depth map, '3' to toggle 3D boxes")
    
    show_depth = SHOW_DEPTH_MAP
    show_3d = SHOW_3D_BOXES
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Generate depth map
        depth_map = estimate_depth(frame)
        depth_normalized = normalize_depth_map(depth_map)
        
        # Run YOLO detection
        results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = results[0].boxes
        
        # Create depth visualization
        depth_colored = cv2.applyColorMap(
            (depth_normalized * 255).astype(np.uint8), 
            DEPTH_COLORMAP
        )
        
        # Blend depth map with original frame
        if show_depth:
            blended = cv2.addWeighted(frame, 1 - ALPHA_BLEND, depth_colored, ALPHA_BLEND, 0)
        else:
            blended = frame.copy()
        
        closest_distance = float('inf')
        detection_info = []
        
        # Process each detection
        for box in detections:
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Calculate center point and depth
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            bbox_height = y2 - y1
            
            # Sample depth in center region of bounding box
            roi_y1, roi_y2 = int(y1 + bbox_height * 0.3), int(y1 + bbox_height * 0.7)
            roi_x1, roi_x2 = int(x1 + (x2 - x1) * 0.3), int(x1 + (x2 - x1) * 0.7)
            roi_y1, roi_y2 = max(0, roi_y1), min(h, roi_y2)
            roi_x1, roi_x2 = max(0, roi_x1), min(w, roi_x2)
            
            depth_roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
            avg_depth = np.median(depth_roi) if depth_roi.size > 0 else depth_map[cy, cx]
            
            # Estimate real-world distance
            distance = depth_to_distance(avg_depth, depth_map, bbox_height, cls_name)
            distance = max(0.1, min(distance, 20.0))  # Clamp to reasonable range
            
            closest_distance = min(closest_distance, distance)
            
            # Determine zone color
            color = get_zone_color(distance)
            
            # Draw 3D box or regular box
            if show_3d:
                draw_3d_box(blended, (x1, y1, x2, y2), distance, color)
            else:
                cv2.rectangle(blended, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Add labels
            label = f"{cls_name} {distance:.1f}m ({conf:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(blended, (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)), color, -1)
            cv2.putText(blended, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw distance line
            cv2.line(blended, (cx, cy), (cx, int(y2)), color, 2)
            cv2.circle(blended, (cx, cy), 5, color, -1)
            
            detection_info.append((cls_name, distance, conf))
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time
        
        # Update statistics
        stats.update(fps, len(detections), closest_distance)
        
        # Draw info panel
        info_y = 30
        cv2.rectangle(blended, (10, 10), (300, 140), (0, 0, 0), -1)
        cv2.putText(blended, f"FPS: {fps:.1f} (avg: {stats.get_avg_fps():.1f})", 
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(blended, f"Detections: {len(detections)} (avg: {stats.get_avg_detections():.1f})", 
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(blended, f"Closest: {closest_distance:.1f}m", 
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_zone_color(closest_distance), 2)
        info_y += 30
        cv2.putText(blended, f"Frame: {frame_count}", 
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw distance zones legend
        legend_x = w - 150
        legend_y = 30
        cv2.rectangle(blended, (legend_x - 10, 10), (w - 10, 170), (0, 0, 0), -1)
        cv2.putText(blended, "Distance Zones:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
        cv2.rectangle(blended, (legend_x, legend_y - 10), (legend_x + 20, legend_y + 5), (0, 0, 255), -1)
        cv2.putText(blended, f"< {CRITICAL_ZONE}m", (legend_x + 30, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 25
        cv2.rectangle(blended, (legend_x, legend_y - 10), (legend_x + 20, legend_y + 5), (0, 165, 255), -1)
        cv2.putText(blended, f"< {WARNING_ZONE}m", (legend_x + 30, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 25
        cv2.rectangle(blended, (legend_x, legend_y - 10), (legend_x + 20, legend_y + 5), (0, 255, 255), -1)
        cv2.putText(blended, f"< {SAFE_ZONE}m", (legend_x + 30, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 25
        cv2.rectangle(blended, (legend_x, legend_y - 10), (legend_x + 20, legend_y + 5), (0, 255, 0), -1)
        cv2.putText(blended, f"> {SAFE_ZONE}m", (legend_x + 30, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display
        cv2.imshow("LumenTact - Depth Estimation", blended)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_depth = not show_depth
            print(f"[INFO] Depth map overlay: {'ON' if show_depth else 'OFF'}")
        elif key == ord('3'):
            show_3d = not show_3d
            print(f"[INFO] 3D boxes: {'ON' if show_3d else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Depth estimation system terminated")
    print(f"[STATS] Total frames processed: {frame_count}")
    print(f"[STATS] Average FPS: {stats.get_avg_fps():.2f}")
    print(f"[STATS] Minimum recorded distance: {stats.get_min_distance():.2f}m")

if __name__ == "__main__":
    main()
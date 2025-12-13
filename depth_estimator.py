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

def main():

    pass

if __name__ == "__main__":
    main()
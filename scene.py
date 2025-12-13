"""
Scene Segmentation Module for LumenTact
Semantic segmentation to classify regions as ground, obstacles, sky, etc.
Helps distinguish walkable surfaces from hazards for navigation.
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
CONFIDENCE_THRESHOLD = 0.5

# Segmentation classes mapping
GROUND_CLASSES = ['road', 'sidewalk', 'terrain', 'ground']
OBSTACLE_CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'chair', 'table', 'bench']
SKY_CLASSES = ['sky']
BUILDING_CLASSES = ['building', 'wall', 'fence']
VEGETATION_CLASSES = ['tree', 'grass', 'plant']

# Color coding for segmentation
COLOR_GROUND = (0, 255, 0)      # Green
COLOR_OBSTACLE = (0, 0, 255)    # Red
COLOR_SKY = (255, 200, 100)     # Light blue
COLOR_BUILDING = (128, 128, 128) # Gray
COLOR_VEGETATION = (0, 128, 0)  # Dark green
COLOR_UNKNOWN = (50, 50, 50)    # Dark gray

# Grid-based segmentation parameters
GRID_SIZE = 20  # pixels
GROUND_DETECTION_RATIO = 0.6  # Threshold for ground classification

# Edge-based ground detection
HORIZON_ESTIMATION_RATIO = 0.4  # Assume horizon at 40% from top
VANISHING_POINT_REGION = 0.3    # Region around center for VP detection

# Walkability scoring
WALKABILITY_HISTORY = 10
MIN_WALKABLE_WIDTH = 100  # Minimum pixels for safe path

# Visualization
SHOW_GRID = True
SHOW_OVERLAY = True
ALPHA_BLEND = 0.5
SHOW_WALKABLE_PATH = True

# -------------------------------
# Ground Detection using Color and Texture
# -------------------------------
class GroundDetector:
    """Detect ground/walkable surfaces using color and texture analysis"""
    def __init__(self):
        self.ground_color_samples = []
        self.texture_threshold = 30
    
    def detect_ground_region(self, frame):
        """Detect ground region using multiple cues"""
        h, w = frame.shape[:2]
        
        # Initialize mask
        ground_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Lower region assumption (ground is typically in lower part)
        lower_region = frame[int(h * 0.6):, :]
        
        # 2. Color-based segmentation (ground often has consistent color)
        hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)
        
        # Calculate dominant color in lower region
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist_h)
        
        # Create mask for similar colors
        lower_bound = np.array([max(0, dominant_hue - 20), 30, 30])
        upper_bound = np.array([min(180, dominant_hue + 20), 255, 255])
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # 3. Texture-based segmentation (ground has consistent texture)
        gray_lower = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)
        texture_var = cv2.Laplacian(gray_lower, cv2.CV_64F).var()
        
        # 4. Edge density (ground has fewer edges than obstacles)
        edges = cv2.Canny(gray_lower, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        # Combine cues
        if edge_density < 0.1 and texture_var < self.texture_threshold * 100:
            ground_mask[int(h * 0.6):, :] = color_mask
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)
        ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_OPEN, kernel)
        
        return ground_mask
    
    def estimate_horizon(self, frame):
        """Estimate horizon line using vanishing point detection"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough line transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None:
            return int(h * HORIZON_ESTIMATION_RATIO)
        
        # Find approximately horizontal lines
        horizontal_lines = []
        for line in lines:
            rho, theta = line[0]
            if abs(theta - np.pi/2) < 0.3:  # Nearly horizontal
                horizontal_lines.append(rho)
        
        if horizontal_lines:
            # Median of horizontal lines
            horizon_y = int(np.median(horizontal_lines))
            horizon_y = max(int(h * 0.2), min(int(h * 0.6), horizon_y))
            return horizon_y
        
        return int(h * HORIZON_ESTIMATION_RATIO)

# -------------------------------
# Grid-based Scene Segmentation
# -------------------------------
class GridSegmenter:
    """Divide scene into grid and classify each cell"""
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.grid_history = deque(maxlen=WALKABILITY_HISTORY)
    
    def create_grid_mask(self, frame_shape):
        """Create grid structure"""
        h, w = frame_shape[:2]
        rows = h // self.grid_size
        cols = w // self.grid_size
        return rows, cols
    
    def classify_grid_cell(self, frame, obstacles_mask, ground_mask, row, col):
        """Classify a single grid cell"""
        y1 = row * self.grid_size
        y2 = min((row + 1) * self.grid_size, frame.shape[0])
        x1 = col * self.grid_size
        x2 = min((col + 1) * self.grid_size, frame.shape[1])
        
        cell_obstacles = obstacles_mask[y1:y2, x1:x2]
        cell_ground = ground_mask[y1:y2, x1:x2]
        
        total_pixels = cell_obstacles.size
        if total_pixels == 0:
            return 'unknown'
        
        obstacle_ratio = np.sum(cell_obstacles) / (total_pixels * 255)
        ground_ratio = np.sum(cell_ground) / (total_pixels * 255)
        
        if obstacle_ratio > 0.3:
            return 'obstacle'
        elif ground_ratio > GROUND_DETECTION_RATIO:
            return 'ground'
        elif y1 < frame.shape[0] * HORIZON_ESTIMATION_RATIO:
            return 'sky'
        else:
            return 'unknown'
    
    def segment_grid(self, frame, obstacles_mask, ground_mask):
        """Segment entire frame into grid"""
        rows, cols = self.create_grid_mask(frame.shape)
        grid_classification = np.empty((rows, cols), dtype=object)
        
        for r in range(rows):
            for c in range(cols):
                grid_classification[r, c] = self.classify_grid_cell(
                    frame, obstacles_mask, ground_mask, r, c
                )
        
        return grid_classification
    
    def calculate_walkability_score(self, grid_classification):
        """Calculate walkability score for bottom rows"""
        rows, cols = grid_classification.shape
        
        # Focus on bottom 3 rows (immediate path)
        bottom_rows = min(3, rows)
        immediate_region = grid_classification[-bottom_rows:, :]
        
        # Count ground cells in immediate path
        ground_cells = np.sum(immediate_region == 'ground')
        total_cells = immediate_region.size
        
        walkability = ground_cells / total_cells if total_cells > 0 else 0.0
        
        # Check for continuous path (left-center-right connectivity)
        center_col = cols // 2
        left_col = center_col - 2
        right_col = center_col + 2
        
        left_clear = np.all(immediate_region[:, max(0, left_col):center_col] != 'obstacle')
        center_clear = np.all(immediate_region[:, center_col] != 'obstacle')
        right_clear = np.all(immediate_region[:, center_col:min(cols, right_col)] != 'obstacle')
        
        # Bonus for continuous center path
        if center_clear:
            walkability += 0.2
        
        # Penalty for blocked sides
        if not left_clear and not right_clear:
            walkability *= 0.5
        
        return min(walkability, 1.0)
    
    def find_walkable_corridors(self, grid_classification):
        """Find continuous walkable corridors"""
        rows, cols = grid_classification.shape
        corridors = []
        
        # Scan from bottom to top
        for col in range(cols):
            corridor_start = None
            corridor_length = 0
            
            for row in range(rows - 1, -1, -1):
                if grid_classification[row, col] == 'ground':
                    if corridor_start is None:
                        corridor_start = row
                    corridor_length += 1
                else:
                    if corridor_start is not None and corridor_length > 3:
                        corridors.append({
                            'col': col,
                            'start_row': corridor_start,
                            'length': corridor_length
                        })
                    corridor_start = None
                    corridor_length = 0
            
            # Handle corridor reaching top
            if corridor_start is not None and corridor_length > 3:
                corridors.append({
                    'col': col,
                    'start_row': corridor_start,
                    'length': corridor_length
                })
        
        return corridors

# -------------------------------
# Obstacle Mask Generation
# -------------------------------
def create_obstacle_mask(frame, detections):
    """Create binary mask of obstacle locations"""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for det in detections:
        bbox, cls_name, conf = det
        if cls_name in OBSTACLE_CLASSES:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = 255
    
    # Dilate to add safety margin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

# -------------------------------
# Visualization Functions
# -------------------------------
def visualize_segmentation(frame, grid_classification, grid_size):
    """Create color-coded segmentation overlay"""
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)
    
    rows, cols = grid_classification.shape
    
    for r in range(rows):
        for c in range(cols):
            y1 = r * grid_size
            y2 = min((r + 1) * grid_size, h)
            x1 = c * grid_size
            x2 = min((c + 1) * grid_size, w)
            
            cls = grid_classification[r, c]
            
            if cls == 'ground':
                color = COLOR_GROUND
            elif cls == 'obstacle':
                color = COLOR_OBSTACLE
            elif cls == 'sky':
                color = COLOR_SKY
            elif cls == 'building':
                color = COLOR_BUILDING
            elif cls == 'vegetation':
                color = COLOR_VEGETATION
            else:
                color = COLOR_UNKNOWN
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    return overlay

def draw_grid_lines(frame, grid_size):
    """Draw grid overlay on frame"""
    h, w = frame.shape[:2]
    
    # Vertical lines
    for x in range(0, w, grid_size):
        cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)
    
    # Horizontal lines
    for y in range(0, h, grid_size):
        cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)

def draw_walkable_corridors(frame, corridors, grid_size):
    """Draw detected walkable corridors"""
    for corridor in corridors:
        col = corridor['col']
        start_row = corridor['start_row']
        length = corridor['length']
        
        x = col * grid_size + grid_size // 2
        y_start = start_row * grid_size
        y_end = (start_row - length) * grid_size
        
        # Color based on length
        if length > 15:
            color = (0, 255, 0)  # Long corridor - green
        elif length > 8:
            color = (0, 255, 255)  # Medium - yellow
        else:
            color = (0, 165, 255)  # Short - orange
        
        cv2.line(frame, (x, y_start), (x, y_end), color, 3)
        cv2.circle(frame, (x, y_end), 5, color, -1)

def draw_walkability_bar(frame, walkability_score):
    """Draw walkability score indicator"""
    h, w = frame.shape[:2]
    
    # Background
    bar_x, bar_y = w - 60, h - 200
    bar_w, bar_h = 40, 150
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
    
    # Fill based on score
    fill_h = int(bar_h * walkability_score)
    fill_y = bar_y + bar_h - fill_h
    
    if walkability_score > 0.7:
        color = (0, 255, 0)
    elif walkability_score > 0.4:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)
    
    cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_w, bar_y + bar_h), color, -1)
    
    # Label
    cv2.putText(frame, "Walk", (bar_x - 10, bar_y - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"{walkability_score:.2f}", (bar_x - 5, bar_y + bar_h + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_horizon_line(frame, horizon_y):
    """Draw estimated horizon line"""
    w = frame.shape[1]
    cv2.line(frame, (0, horizon_y), (w, horizon_y), (255, 255, 0), 2)
    cv2.putText(frame, "HORIZON", (10, horizon_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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
    
    ground_detector = GroundDetector()
    grid_segmenter = GridSegmenter(GRID_SIZE)
    
    prev_time = time.time()
    frame_count = 0
    walkability_history = deque(maxlen=WALKABILITY_HISTORY)
    
    print("[INFO] Starting scene segmentation system...")
    print("[INFO] Press 'q' to quit, 'g' to toggle grid, 'o' to toggle overlay")
    
    show_grid = SHOW_GRID
    show_overlay = SHOW_OVERLAY
    
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
        
        # Create obstacle mask
        obstacle_mask = create_obstacle_mask(frame, detections)
        
        # Detect ground
        ground_mask = ground_detector.detect_ground_region(frame)
        
        # Estimate horizon
        horizon_y = ground_detector.estimate_horizon(frame)
        
        # Grid segmentation
        grid_classification = grid_segmenter.segment_grid(frame, obstacle_mask, ground_mask)
        
        # Calculate walkability
        walkability_score = grid_segmenter.calculate_walkability_score(grid_classification)
        walkability_history.append(walkability_score)
        avg_walkability = np.mean(walkability_history)
        
        # Find walkable corridors
        corridors = grid_segmenter.find_walkable_corridors(grid_classification)
        
        # Create visualization
        output = frame.copy()
        
        # Draw segmentation overlay
        if show_overlay:
            seg_overlay = visualize_segmentation(frame, grid_classification, GRID_SIZE)
            output = cv2.addWeighted(output, 1 - ALPHA_BLEND, seg_overlay, ALPHA_BLEND, 0)
        
        # Draw grid
        if show_grid:
            draw_grid_lines(output, GRID_SIZE)
        
        # Draw horizon
        draw_horizon_line(output, horizon_y)
        
        # Draw corridors
        draw_walkable_corridors(output, corridors, GRID_SIZE)
        
        # Draw walkability bar
        draw_walkability_bar(output, avg_walkability)
        
        # Draw detections
        for det in detections:
            bbox, cls_name, conf = det
            if cls_name in OBSTACLE_CLASSES:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(output, cls_name, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time
        
        # Info panel
        info_y = 30
        cv2.rectangle(output, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.putText(output, f"FPS: {fps:.1f}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(output, f"Walkability: {avg_walkability:.2f}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += 30
        cv2.putText(output, f"Corridors: {len(corridors)}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(output, f"Obstacles: {len([d for d in detections if d[1] in OBSTACLE_CLASSES])}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("LumenTact - Scene Segmentation", output)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            show_grid = not show_grid
            print(f"[INFO] Grid overlay: {'ON' if show_grid else 'OFF'}")
        elif key == ord('o'):
            show_overlay = not show_overlay
            print(f"[INFO] Segmentation overlay: {'ON' if show_overlay else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Scene segmentation system terminated")
    print(f"[STATS] Total frames: {frame_count}")
    print(f"[STATS] Average walkability: {np.mean(walkability_history):.2f}")

if __name__ == "__main__":
    main()
"""
Scene Segmentation Module for LumenTact
Core scene understanding with ground detection, obstacle masking,
and grid-based walkability analysis.
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

GRID_SIZE = 20
GROUND_DETECTION_RATIO = 0.6
HORIZON_ESTIMATION_RATIO = 0.4
WALKABILITY_HISTORY = 10

OBSTACLE_CLASSES = [
    'person', 'car', 'bicycle', 'motorcycle',
    'chair', 'table', 'bench'
]

# -------------------------------
# Ground Detection
# -------------------------------
class GroundDetector:
    """Detect walkable ground using color, texture, and edge cues"""
    def __init__(self):
        self.texture_threshold = 30
    
    def detect_ground_region(self, frame):
        h, w = frame.shape[:2]
        ground_mask = np.zeros((h, w), dtype=np.uint8)

        lower_region = frame[int(h * 0.6):, :]
        hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist)

        lower = np.array([max(0, dominant_hue - 20), 30, 30])
        upper = np.array([min(180, dominant_hue + 20), 255, 255])
        color_mask = cv2.inRange(hsv, lower, upper)

        gray = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)
        texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size

        if edge_density < 0.1 and texture_var < self.texture_threshold * 100:
            ground_mask[int(h * 0.6):, :] = color_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)
        ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_OPEN, kernel)

        return ground_mask
    
    def estimate_horizon(self, frame):
        h = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is None:
            return int(h * HORIZON_ESTIMATION_RATIO)

        horizontal = [
            rho for line in lines
            for rho, theta in [line[0]]
            if abs(theta - np.pi / 2) < 0.3
        ]

        if horizontal:
            return int(np.clip(
                np.median(horizontal),
                int(h * 0.2),
                int(h * 0.6)
            ))

        return int(h * HORIZON_ESTIMATION_RATIO)

# -------------------------------
# Grid-Based Segmentation
# -------------------------------
class GridSegmenter:
    """Classify scene into grid cells"""
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.history = deque(maxlen=WALKABILITY_HISTORY)

    def segment(self, frame, obstacle_mask, ground_mask):
        h, w = frame.shape[:2]
        rows, cols = h // self.grid_size, w // self.grid_size
        grid = np.empty((rows, cols), dtype=object)

        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * self.grid_size, (r + 1) * self.grid_size
                x1, x2 = c * self.grid_size, (c + 1) * self.grid_size

                obs = obstacle_mask[y1:y2, x1:x2]
                grd = ground_mask[y1:y2, x1:x2]
                total = obs.size

                if total == 0:
                    grid[r, c] = 'unknown'
                    continue

                if np.sum(obs) / (total * 255) > 0.3:
                    grid[r, c] = 'obstacle'
                elif np.sum(grd) / (total * 255) > GROUND_DETECTION_RATIO:
                    grid[r, c] = 'ground'
                elif y1 < frame.shape[0] * HORIZON_ESTIMATION_RATIO:
                    grid[r, c] = 'sky'
                else:
                    grid[r, c] = 'unknown'

        return grid

    def walkability(self, grid):
        bottom = grid[-min(3, grid.shape[0]):, :]
        score = np.sum(bottom == 'ground') / bottom.size
        self.history.append(score)
        return np.mean(self.history)

# -------------------------------
# Obstacle Mask
# -------------------------------
def create_obstacle_mask(frame, detections):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for (x1, y1, x2, y2), cls, _ in detections:
        if cls in OBSTACLE_CLASSES:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            mask[y1:y2, x1:x2] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.dilate(mask, kernel, 1)

# -------------------------------
# Main Loop (Minimal Output)
# -------------------------------
def main():
    model = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(0)

    ground = GroundDetector()
    grid = GridSegmenter(GRID_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []

        for box in results[0].boxes:
            cls = model.names[int(box.cls[0])]
            detections.append((box.xyxy[0].tolist(), cls, float(box.conf[0])))

        obstacle_mask = create_obstacle_mask(frame, detections)
        ground_mask = ground.detect_ground_region(frame)
        grid_map = grid.segment(frame, obstacle_mask, ground_mask)
        score = grid.walkability(grid_map)

        cv2.putText(frame, f"Walkability: {score:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("LumenTact Scene Segmentation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

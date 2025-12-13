"""
Scene Segmentation Module for LumenTact
Ground detection and grid-based semantic segmentation.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Configuration
# -------------------------------
YOLO_SEG_MODEL_PATH = 'yolov8n-seg.pt'
CONFIDENCE_THRESHOLD = 0.5

GRID_SIZE = 20
GROUND_DETECTION_RATIO = 0.6
HORIZON_ESTIMATION_RATIO = 0.4

COLOR_GROUND = (0, 255, 0)
COLOR_OBSTACLE = (0, 0, 255)
COLOR_SKY = (255, 200, 100)
COLOR_UNKNOWN = (50, 50, 50)

ALPHA_BLEND = 0.5
SHOW_GRID = True

# -------------------------------
# Ground Detection
# -------------------------------
class GroundDetector:
    def __init__(self):
        self.texture_threshold = 30

    def detect_ground_region(self, frame):
        h, w = frame.shape[:2]
        ground_mask = np.zeros((h, w), dtype=np.uint8)

        lower = frame[int(h * 0.6):, :]
        hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant = np.argmax(hist)

        lb = np.array([max(0, dominant - 20), 30, 30])
        ub = np.array([min(180, dominant + 20), 255, 255])
        color_mask = cv2.inRange(hsv, lb, ub)

        gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size

        texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if edge_density < 0.1 and texture_var < self.texture_threshold * 100:
            ground_mask[int(h * 0.6):, :] = color_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)
        ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_OPEN, kernel)

        return ground_mask

# -------------------------------
# Grid Segmenter
# -------------------------------
class GridSegmenter:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size

    def segment(self, frame, obstacle_mask, ground_mask):
        h, w = frame.shape[:2]
        rows = h // self.grid_size
        cols = w // self.grid_size
        grid = np.empty((rows, cols), dtype=object)

        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * self.grid_size, (r + 1) * self.grid_size
                x1, x2 = c * self.grid_size, (c + 1) * self.grid_size

                obs = obstacle_mask[y1:y2, x1:x2]
                grd = ground_mask[y1:y2, x1:x2]

                if obs.size == 0:
                    grid[r, c] = 'unknown'
                    continue

                if np.sum(obs) / (obs.size * 255) > 0.3:
                    grid[r, c] = 'obstacle'
                elif np.sum(grd) / (grd.size * 255) > GROUND_DETECTION_RATIO:
                    grid[r, c] = 'ground'
                elif y1 < h * HORIZON_ESTIMATION_RATIO:
                    grid[r, c] = 'sky'
                else:
                    grid[r, c] = 'unknown'

        return grid

# -------------------------------
# Main Loop
# -------------------------------
def main():
    model = YOLO(YOLO_SEG_MODEL_PATH)
    cap = cv2.VideoCapture(0)

    ground_detector = GroundDetector()
    grid_segmenter = GridSegmenter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = np.zeros_like(frame)
        obstacle_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        if results[0].masks is not None:
            for mask in results[0].masks.data:
                obstacle_mask[mask.cpu().numpy() == 1] = 255

        ground_mask = ground_detector.detect_ground_region(frame)
        grid = grid_segmenter.segment(frame, obstacle_mask, ground_mask)

        overlay[ground_mask > 0] = COLOR_GROUND
        overlay[obstacle_mask > 0] = COLOR_OBSTACLE

        if SHOW_GRID:
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    y = r * GRID_SIZE
                    x = c * GRID_SIZE
                    cv2.rectangle(overlay, (x, y),
                                  (x + GRID_SIZE, y + GRID_SIZE),
                                  (80, 80, 80), 1)

        output = cv2.addWeighted(frame, 1 - ALPHA_BLEND, overlay, ALPHA_BLEND, 0)
        cv2.imshow("LumenTact - Grid Segmentation", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

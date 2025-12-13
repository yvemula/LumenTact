"""
Optical Flow Analysis Module for LumenTact
Dense optical flow with region-based motion analysis and clustering.
"""

import cv2
import numpy as np
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Configuration
# -------------------------------
FLOW_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2

MOTION_THRESHOLD = 2.0
STATIC_THRESHOLD = 0.5
DYNAMIC_OBJECT_THRESHOLD = 3.0

REGION_GRID_SIZE = 40
MOTION_HISTORY_SIZE = 30

# -------------------------------
# Optical Flow Computer
# -------------------------------
class OpticalFlowComputer:
    def __init__(self):
        self.prev_gray = None
    
    def compute_flow(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None

        prev = cv2.resize(self.prev_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)
        curr = cv2.resize(gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)

        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, FLOW_LEVELS,
            FLOW_WINSIZE, FLOW_ITERATIONS,
            FLOW_POLY_N, FLOW_POLY_SIGMA, 0
        )

        flow = cv2.resize(flow, (gray.shape[1], gray.shape[0]))
        flow /= FLOW_SCALE
        self.prev_gray = gray.copy()
        return flow

    def magnitude(self, flow):
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return mag

# -------------------------------
# Motion Region Analyzer
# -------------------------------
class MotionRegionAnalyzer:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.history = deque(maxlen=MOTION_HISTORY_SIZE)

    def analyze(self, mag, shape):
        h, w = shape[:2]
        rows, cols = h // self.grid_size, w // self.grid_size
        grid = np.zeros((rows, cols))
        types = np.empty((rows, cols), dtype=object)

        for r in range(rows):
            for c in range(cols):
                region = mag[
                    r*self.grid_size:(r+1)*self.grid_size,
                    c*self.grid_size:(c+1)*self.grid_size
                ]
                avg = np.mean(region)
                grid[r, c] = avg

                if avg > DYNAMIC_OBJECT_THRESHOLD:
                    types[r, c] = 'dynamic'
                elif avg > MOTION_THRESHOLD:
                    types[r, c] = 'moving'
                elif avg > STATIC_THRESHOLD:
                    types[r, c] = 'slow'
                else:
                    types[r, c] = 'static'

        self.history.append(grid)
        return grid, types

    def detect_clusters(self, grid, types):
        clusters = []
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)

        def dfs(r, c, cluster):
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return
            if visited[r, c] or types[r, c] == 'static':
                return
            visited[r, c] = True
            cluster.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                dfs(r+dr, c+dc, cluster)

        for r in range(rows):
            for c in range(cols):
                if not visited[r, c] and types[r, c] != 'static':
                    cluster = []
                    dfs(r, c, cluster)
                    if len(cluster) >= 2:
                        clusters.append(cluster)

        return clusters

# -------------------------------
# Main Loop
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    flow_comp = OpticalFlowComputer()
    analyzer = MotionRegionAnalyzer(REGION_GRID_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = flow_comp.compute_flow(gray)
        output = frame.copy()

        if flow is not None:
            mag = flow_comp.magnitude(flow)
            grid, types = analyzer.analyze(mag, frame.shape)
            clusters = analyzer.detect_clusters(grid, types)

            cv2.putText(output, f"Clusters: {len(clusters)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

        cv2.imshow("Optical Flow - Regions", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

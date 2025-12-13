"""
Optical Flow Analysis Module for LumenTact
Dense optical flow analysis for motion detection and regional motion analysis.
"""

import cv2
import numpy as np
import time
from collections import deque
import warnings
from collections import deque
warnings.filterwarnings('ignore')

# -------------------------------
# Configuration
# -------------------------------
FLOW_METHOD = 'farneback'
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
FLOW_HISTORY_SIZE = 15
MOTION_HISTORY_SIZE = 30

SHOW_FLOW_VECTORS = True
VECTOR_STEP = 10
VECTOR_SCALE = 3

# -------------------------------
# Optical Flow Computer
# -------------------------------
class OpticalFlowComputer:
    def __init__(self, method='farneback'):
        self.method = method
        self.prev_gray = None
        self.flow_history = deque(maxlen=FLOW_HISTORY_SIZE)

    def compute_flow(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None

        prev = cv2.resize(self.prev_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)
        curr = cv2.resize(gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)

        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, FLOW_LEVELS, FLOW_WINSIZE,
            FLOW_ITERATIONS, FLOW_POLY_N, FLOW_POLY_SIGMA, 0
        )

        flow = cv2.resize(flow, (gray.shape[1], gray.shape[0]))
        flow /= FLOW_SCALE

        self.prev_gray = gray.copy()
        self.flow_history.append(flow)
        return flow

    def get_flow_magnitude(self, flow):
        if flow is None:
            return None
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return mag

# -------------------------------
# Motion Region Analyzer
# -------------------------------
class MotionRegionAnalyzer:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.motion_history = deque(maxlen=MOTION_HISTORY_SIZE)

    def analyze_regions(self, mag, frame_shape):
        h, w = frame_shape[:2]
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

        return grid, types

# -------------------------------
# Visualization
# -------------------------------
def visualize_flow_vectors(frame, flow):
    for y in range(0, frame.shape[0], VECTOR_STEP):
        for x in range(0, frame.shape[1], VECTOR_STEP):
            fx, fy = flow[y, x]
            if np.hypot(fx, fy) < STATIC_THRESHOLD:
                continue
            cv2.arrowedLine(
                frame, (x, y),
                (int(x + fx * VECTOR_SCALE), int(y + fy * VECTOR_SCALE)),
                (0, 255, 0), 1
            )



# -------------------------------
# Ego-Motion Estimator
# -------------------------------
class EgoMotionEstimator:
    def __init__(self):
        self.history = deque(maxlen=10)

    def estimate(self, flow):
        vectors = flow.reshape(-1, 2)
        mags = np.linalg.norm(vectors, axis=1)
        valid = vectors[mags > 0.5]

        if len(valid) < 10:
            return np.array([0.0, 0.0])

        ego = np.median(valid, axis=0)
        self.history.append(ego)
        return ego

    def smooth(self):
        if not self.history:
            return np.array([0.0, 0.0])
        return np.mean(self.history, axis=0)

    def compensate(self, flow, ego):
        compensated = flow.copy()
        compensated[..., 0] -= ego[0]
        compensated[..., 1] -= ego[1]
        return compensated


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    flow_computer = OpticalFlowComputer(FLOW_METHOD)
    region_analyzer = MotionRegionAnalyzer(REGION_GRID_SIZE)
    ego_estimator = EgoMotionEstimator()

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Optical flow ---
        flow = flow_computer.compute_flow(gray)

        output = frame.copy()

        if flow is not None:
            # --- Ego-motion estimation ---
            ego_motion = ego_estimator.estimate(flow)
            ego_smoothed = ego_estimator.smooth()

            # --- Compensate flow ---
            flow_compensated = ego_estimator.compensate(flow, ego_smoothed)

            # --- Magnitude + region analysis ---
            mag = flow_computer.get_flow_magnitude(flow_compensated)
            motion_grid, motion_types = region_analyzer.analyze_regions(
                mag, frame.shape
            )
            region_analyzer.update_history(motion_grid)

            # --- Visualization ---
            visualize_flow_vectors(output, flow_compensated)

        # --- FPS ---
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time

        cv2.putText(
            output, f"FPS: {fps:.1f}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.imshow("LumenTact - Optical Flow (Ego Compensated)", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

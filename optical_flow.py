"""
Optical Flow Analysis Module for LumenTact
Core dense optical flow computation and motion magnitude analysis.
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

# -------------------------------
# Optical Flow Computer
# -------------------------------
class OpticalFlowComputer:
    """Compute dense optical flow between frames"""
    def __init__(self):
        self.prev_gray = None
        self.flow_history = deque(maxlen=10)
    
    def compute_flow(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None
        
        prev = cv2.resize(self.prev_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)
        curr = cv2.resize(gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)

        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5,
            levels=FLOW_LEVELS,
            winsize=FLOW_WINSIZE,
            iterations=FLOW_ITERATIONS,
            poly_n=FLOW_POLY_N,
            poly_sigma=FLOW_POLY_SIGMA,
            flags=0
        )

        flow = cv2.resize(flow, (gray.shape[1], gray.shape[0]))
        flow /= FLOW_SCALE

        self.prev_gray = gray.copy()
        self.flow_history.append(flow)
        return flow

    def flow_magnitude(self, flow):
        if flow is None:
            return None
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return mag

# -------------------------------
# Visualization
# -------------------------------
def visualize_flow_vectors(frame, flow, step=10, scale=3):
    h, w = frame.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            mag = np.sqrt(fx**2 + fy**2)
            if mag < STATIC_THRESHOLD:
                continue
            end = (int(x + fx * scale), int(y + fy * scale))
            cv2.arrowedLine(frame, (x, y), end, (0, 255, 0), 1)

# -------------------------------
# Main Loop
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    flow_comp = OpticalFlowComputer()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = flow_comp.compute_flow(gray)

        output = frame.copy()
        if flow is not None:
            mag = flow_comp.flow_magnitude(flow)
            visualize_flow_vectors(output, flow)

            avg_motion = np.mean(mag)
            cv2.putText(output, f"Avg motion: {avg_motion:.2f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        cv2.putText(output, f"FPS: {fps:.1f}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.imshow("Optical Flow - Core", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

"""
Crosswalk and Traffic Light Detection Module for LumenTact
Detects zebra crossings and traffic lights for navigation assistance.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Configuration
# -------------------------------
YOLO_MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5

STRIPE_DETECTION_THRESHOLD = 0.7
MIN_STRIPE_WIDTH = 20
MIN_STRIPE_COUNT = 3
STRIPE_SPACING_TOLERANCE = 0.3

CROSSWALK_ROI_TOP = 0.5
CROSSWALK_ROI_BOTTOM = 0.9

DETECTION_HISTORY_SIZE = 10

# -------------------------------
# Traffic Light State Enum
# -------------------------------
class TrafficLightState(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"

# -------------------------------
# Crosswalk Stripe Detector
# -------------------------------
class CrosswalkDetector:
    def __init__(self):
        self.detection_history = deque(maxlen=DETECTION_HISTORY_SIZE)

    def detect_white_stripes(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def find_horizontal_stripes(self, binary):
        h, w = binary.shape
        stripes = []
        slice_h = h // 20

        for y in range(0, h - slice_h, slice_h):
            slice_ = binary[y:y + slice_h, :]
            projection = np.sum(slice_, axis=0) / (slice_h * 255)

            in_stripe = False
            start_x = 0

            for x in range(w):
                if projection[x] > STRIPE_DETECTION_THRESHOLD:
                    if not in_stripe:
                        start_x = x
                        in_stripe = True
                else:
                    if in_stripe:
                        width = x - start_x
                        if width > MIN_STRIPE_WIDTH:
                            stripes.append({
                                'y': y + slice_h // 2,
                                'x_start': start_x,
                                'x_end': x,
                                'width': width
                            })
                        in_stripe = False

        return stripes


def main():
    cap = cv2.VideoCapture(0)
    detector = CrosswalkDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, _ = frame.shape[:2]
        roi = frame[int(h * CROSSWALK_ROI_TOP):int(h * CROSSWALK_ROI_BOTTOM), :]
        binary = detector.detect_white_stripes(roi)

        cv2.imshow("Stripe Binary", binary)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

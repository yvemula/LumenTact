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
    
# -------------------------------
# Crosswalk Zebra Pattern Detection
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
                            stripes.append({'y': y + slice_h // 2, 'x_start': start_x, 'x_end': x, 'width': width})
                        in_stripe = False
        return stripes

    def detect_zebra_pattern(self, stripes):
        if len(stripes) < MIN_STRIPE_COUNT:
            return None

        stripes_sorted = sorted(stripes, key=lambda s: s['y'])
        groups = []
        current = [stripes_sorted[0]]

        for s in stripes_sorted[1:]:
            prev = current[-1]
            spacing = s['y'] - prev['y']
            if len(current) > 1:
                avg_spacing = np.mean([current[j]['y'] - current[j-1]['y'] for j in range(1, len(current))])
                spacing_diff = abs(spacing - avg_spacing) / avg_spacing
            else:
                spacing_diff = 0
            if spacing_diff < STRIPE_SPACING_TOLERANCE and 10 < spacing < 80:
                current.append(s)
            else:
                if len(current) >= MIN_STRIPE_COUNT:
                    groups.append(current)
                current = [s]

        if len(current) >= MIN_STRIPE_COUNT:
            groups.append(current)

        if groups:
            largest = max(groups, key=len)
            y_min = min(s['y'] for s in largest)
            y_max = max(s['y'] for s in largest)
            x_min = min(s['x_start'] for s in largest)
            x_max = max(s['x_end'] for s in largest)
            return {'stripes': largest, 'bbox': (x_min, y_min, x_max, y_max), 'count': len(largest), 'confidence': min(len(largest)/8.0,1.0)}
        return None

    def detect_crosswalk(self, frame):
        h, _ = frame.shape[:2]
        roi_y1, roi_y2 = int(h * CROSSWALK_ROI_TOP), int(h * CROSSWALK_ROI_BOTTOM)
        roi = frame[roi_y1:roi_y2, :]
        binary = self.detect_white_stripes(roi)
        stripes = self.find_horizontal_stripes(binary)
        crosswalk = self.detect_zebra_pattern(stripes)
        if crosswalk:
            bbox = crosswalk['bbox']
            crosswalk['bbox'] = (bbox[0], bbox[1]+roi_y1, bbox[2], bbox[3]+roi_y1)
            for s in crosswalk['stripes']:
                s['y'] += roi_y1
            self.detection_history.append(True)
            return crosswalk, binary
        self.detection_history.append(False)
        return None, binary

    def get_detection_confidence(self):
        return sum(self.detection_history)/len(self.detection_history) if self.detection_history else 0.0



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

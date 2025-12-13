"""
LumenTact – Hazard Detection Core
Adds stair and curb reasoning with confidence smoothing.
"""

import cv2
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Configuration
# -------------------------------
CANNY_LOW = 50
CANNY_HIGH = 150
BLUR_KERNEL = 5

HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 50
HOUGH_MAX_LINE_GAP = 10

ROI_TOP_RATIO = 0.4
ROI_BOTTOM_RATIO = 1.0

STAIR_SPACING = 30
CURB_HEIGHT = 20
HISTORY_SIZE = 10

# -------------------------------
# Detection History
# -------------------------------
class DetectionHistory:
    def __init__(self):
        self.stairs = deque(maxlen=HISTORY_SIZE)
        self.curbs = deque(maxlen=HISTORY_SIZE)

    def update(self, stairs, curbs):
        self.stairs.append(int(stairs))
        self.curbs.append(int(curbs))

    def stair_conf(self):
        return sum(self.stairs) / len(self.stairs) if self.stairs else 0

    def curb_conf(self):
        return sum(self.curbs) / len(self.curbs) if self.curbs else 0

# -------------------------------
# Geometry
# -------------------------------
def angle(x1, y1, x2, y2):
    return abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

def is_horizontal(x1, y1, x2, y2):
    a = angle(x1, y1, x2, y2)
    return a < 15 or a > 165

# -------------------------------
# Hazard Detection
# -------------------------------
def detect_stairs(lines):
    horizontals = [l for l in lines if is_horizontal(*l)]
    horizontals.sort(key=lambda l: (l[1] + l[3]) / 2)

    groups, group = [], []
    for line in horizontals:
        if not group:
            group.append(line)
            continue

        prev_y = (group[-1][1] + group[-1][3]) / 2
        curr_y = (line[1] + line[3]) / 2

        if abs(curr_y - prev_y) < STAIR_SPACING * 1.5:
            group.append(line)
        else:
            if len(group) >= 3:
                groups.append(group)
            group = [line]

    if len(group) >= 3:
        groups.append(group)

    return groups

def detect_curbs(edges, lines):
    curbs = []
    for x1, y1, x2, y2 in lines:
        if is_horizontal(x1, y1, x2, y2):
            band = edges[int(y1):int(y1 + CURB_HEIGHT), int(x1):int(x2)]
            if band.size > 0 and np.sum(band) > CURB_HEIGHT * 50:
                curbs.append((x1, y1, x2, y2))
    return curbs

# -------------------------------
# Main
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    history = DetectionHistory()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h = frame.shape[0]
        roi_y1 = int(h * ROI_TOP_RATIO)
        roi = frame[roi_y1:, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

        raw = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                              minLineLength=HOUGH_MIN_LINE_LENGTH,
                              maxLineGap=HOUGH_MAX_LINE_GAP)

        lines = []
        if raw is not None:
            for l in raw:
                x1, y1, x2, y2 = l[0]
                lines.append([x1, y1 + roi_y1, x2, y2 + roi_y1])

        stairs = detect_stairs(lines)
        curbs = detect_curbs(edges, lines)
        history.update(bool(stairs), bool(curbs))

        output = frame.copy()
        cv2.putText(output, f"Stair confidence: {history.stair_conf():.2f}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, f"Curb confidence: {history.curb_conf():.2f}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("LumenTact – Hazard Core", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

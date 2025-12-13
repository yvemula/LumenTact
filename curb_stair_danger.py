"""
LumenTact – Core Vision Pipeline
Implements ROI-based edge detection and Hough line analysis.
"""

import cv2
import numpy as np
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

# -------------------------------
# Line Geometry Utilities
# -------------------------------
def calculate_angle(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    return abs(np.degrees(np.arctan2(dy, dx)))

def is_horizontal(x1, y1, x2, y2, thresh=15):
    angle = calculate_angle(x1, y1, x2, y2)
    return angle < thresh or angle > 180 - thresh

def is_vertical(x1, y1, x2, y2, thresh=75):
    angle = calculate_angle(x1, y1, x2, y2)
    return thresh < angle < (180 - thresh)

def merge_lines(lines, dist_thresh=20):
    if not lines:
        return []

    merged = []
    for x1, y1, x2, y2 in lines:
        merged.append([x1, y1, x2, y2])
    return merged

# -------------------------------
# Frame Processing
# -------------------------------
def process_frame(frame):
    h, w = frame.shape[:2]
    roi_y1 = int(h * ROI_TOP_RATIO)
    roi_y2 = int(h * ROI_BOTTOM_RATIO)

    roi = frame[roi_y1:roi_y2, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    lines_raw = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    lines = []
    if lines_raw is not None:
        for l in lines_raw:
            x1, y1, x2, y2 = l[0]
            lines.append([x1, y1 + roi_y1, x2, y2 + roi_y1])

    return edges, merge_lines(lines), (0, roi_y1, w, roi_y2)

# -------------------------------
# Main
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges, lines, roi = process_frame(frame)
        display = frame.copy()

        for x1, y1, x2, y2 in lines:
            cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow("Core Vision Pipeline", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

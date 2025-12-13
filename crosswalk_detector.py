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
# Traffic Light Analyzer
# -------------------------------
class TrafficLightAnalyzer:
    def __init__(self):
        self.light_history = deque(maxlen=LIGHT_STATE_HISTORY)

    def detect_light_color(self, roi):
        if roi.size == 0:
            return TrafficLightState.UNKNOWN
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(cv2.inRange(hsv, np.array(RED_RANGE_1[0]), np.array(RED_RANGE_1[1])),
                                  cv2.inRange(hsv, np.array(RED_RANGE_2[0]), np.array(RED_RANGE_2[1])))
        yellow_mask = cv2.inRange(hsv, np.array(YELLOW_RANGE[0]), np.array(YELLOW_RANGE[1]))
        green_mask = cv2.inRange(hsv, np.array(GREEN_RANGE[0]), np.array(GREEN_RANGE[1]))

        counts = {'RED': np.sum(red_mask>0), 'YELLOW': np.sum(yellow_mask>0), 'GREEN': np.sum(green_mask>0)}
        if max(counts.values()) < 50:
            return TrafficLightState.UNKNOWN
        return TrafficLightState(max(counts, key=counts.get))

    def analyze_traffic_lights(self, frame, detections):
        lights = []
        for bbox, cls_name, conf in detections:
            if cls_name in TRAFFIC_LIGHT_CLASSES:
                x1, y1, x2, y2 = map(int, bbox)
                roi = frame[y1:y2, x1:x2]
                state = self.detect_light_color(roi)
                lights.append({'bbox':(x1,y1,x2,y2),'state':state,'confidence':conf,'class':cls_name})
        return lights

    def get_dominant_state(self, lights):
        if not lights: return TrafficLightState.UNKNOWN
        states = [l['state'] for l in lights]
        if TrafficLightState.RED in states: return TrafficLightState.RED
        if TrafficLightState.YELLOW in states: return TrafficLightState.YELLOW
        if TrafficLightState.GREEN in states: return TrafficLightState.GREEN
        return TrafficLightState.UNKNOWN

    def update_history(self, state):
        self.light_history.append(state)

    def get_stable_state(self):
        if not self.light_history: return TrafficLightState.UNKNOWN
        counts = {}
        for s in self.light_history: counts[s]=counts.get(s,0)+1
        return max(counts,key=counts.get)


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

# -------------------------------
# Visualization
# -------------------------------
def draw_crosswalk(frame, crosswalk, show_stripes=True):
    x1,y1,x2,y2 = map(int,crosswalk['bbox'])
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
    if show_stripes:
        for s in crosswalk['stripes']:
            cv2.line(frame,(s['x_start'],s['y']),(s['x_end'],s['y']),(255,0,255),2)
    label=f"CROSSWALK ({crosswalk['count']} stripes) {crosswalk['confidence']:.2f}"
    cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

def draw_traffic_light(frame, light):
    x1,y1,x2,y2 = light['bbox']
    state = light['state']
    color = {(TrafficLightState.RED):(0,0,255),
             (TrafficLightState.YELLOW):(0,255,255),
             (TrafficLightState.GREEN):(0,255,0)}.get(state,(128,128,128))
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
    label = f"{state.value}"
    label_size,_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
    cv2.rectangle(frame,(x1,y1-label_size[1]-10),(x1+label_size[0],y1),color,-1)
    cv2.putText(frame,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

def draw_safety_status(frame,status,color):
    h,w = frame.shape[:2]
    cv2.rectangle(frame,(w//2-250,h-80),(w//2+250,h-20),(0,0,0),-1)
    cv2.rectangle(frame,(w//2-250,h-80),(w//2+250,h-20),color,3)
    cv2.putText(frame,status.replace('_',' '),(w//2-200,h-40),cv2.FONT_HERSHEY_SIMPLEX,1.0,color,3)

def draw_roi_zones(frame):
    h,w = frame.shape[:2]
    cv2.rectangle(frame,(0,int(h*CROSSWALK_ROI_TOP)),(w,int(h*CROSSWALK_ROI_BOTTOM)),(0,255,0),2)
    cv2.line(frame,(0,int(h*TRAFFIC_LIGHT_ROI_TOP)),(w,int(h*TRAFFIC_LIGHT_ROI_TOP)),(255,0,0),2)

# -------------------------------
# Main loop
# -------------------------------
def main():
    print("[INFO] Loading YOLO model...")
    yolo = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    crosswalk_detector = CrosswalkDetector()
    traffic_analyzer = TrafficLightAnalyzer()
    safety_monitor = CrossingSafetyMonitor()

    show_stripes, show_roi = SHOW_STRIPES, SHOW_ROI
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # YOLO detections
        results = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = [((int(b.xyxy[0][0]),int(b.xyxy[0][1]),int(b.xyxy[0][2]),int(b.xyxy[0][3])),
                       yolo.names[int(b.cls[0])],float(b.conf[0])) for b in results[0].boxes]

        # Crosswalk
        crosswalk, _ = crosswalk_detector.detect_crosswalk(frame)
        conf = crosswalk_detector.get_detection_confidence()

        # Traffic lights
        lights = traffic_analyzer.analyze_traffic_lights(frame,detections)
        dominant = traffic_analyzer.get_dominant_state(lights)
        traffic_analyzer.update_history(dominant)
        stable = traffic_analyzer.get_stable_state()

        # Safety
        dist = frame.shape[0]-crosswalk['bbox'][3] if crosswalk else float('inf')
        status,color = safety_monitor.assess_crossing_safety(crosswalk is not None, stable, dist)

        output = frame.copy()
        if show_roi: draw_roi_zones(output)
        if crosswalk: draw_crosswalk(output,crosswalk,show_stripes)
        for l in lights: draw_traffic_light(output,l)
        draw_safety_status(output,status,color)

        # FPS
        curr_time = time.time()
        fps = 1/(curr_time-prev_time) if curr_time>prev_time else 0
        prev_time=curr_time
        cv2.putText(output,f"FPS: {fps:.1f}",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.imshow("LumenTact - Crosswalk Detector",output)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'): break
        elif key==ord('s'): show_stripes = not show_stripes
        elif key==ord('r'): show_roi = not show_roi

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Crosswalk detection terminated.")

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()

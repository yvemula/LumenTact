from ultralytics import YOLO
import cv2
import numpy as np
import time
import pyttsx3

# -------------------------------
# Config
# -------------------------------
OBSTACLE_CLASSES = ["person", "chair", "table", "potted plant", "tv", "refrigerator", "vase"]
D_THRESH = 120  # safe distance threshold (in pixels)
SHOW_BARS = True  # set False to hide the distance bar overlay
SMOOTHING_ALPHA = 0.6  # smoothing factor for distances
FPS_SMOOTHING = 0.9    # for display smoothing

# -------------------------------
# Initialize modules
# -------------------------------
model = YOLO('yolov8n.pt')
engine = pyttsx3.init()
engine.setProperty('rate', 180)

# -------------------------------
# Mock haptic/audio feedback
# -------------------------------
def play_audio(msg):
    print(f"[AUDIO] {msg}")
    engine.say(msg)
    engine.runAndWait()

def vibrate_pattern(pattern):
    print(f"[HAPTIC] {pattern}")

def draw_feedback(frame, left, front, right, max_w):
    """Draw distance bars on frame"""
    bar_w = 30
    h, w, _ = frame.shape

    def bar_height(dist):
        return int((1 - min(dist / max_w, 1)) * (h * 0.8))

    cv2.rectangle(frame, (10, h - bar_height(left)), (10 + bar_w, h), (255, 0, 0), -1)
    cv2.rectangle(frame, (w//2 - bar_w//2, h - bar_height(front)), (w//2 + bar_w//2, h), (0, 255, 0), -1)
    cv2.rectangle(frame, (w - bar_w - 10, h - bar_height(right)), (w - 10, h), (0, 0, 255), -1)

# -------------------------------
# Webcam setup
# -------------------------------
cap = cv2.VideoCapture(0)  # 0 for default webcam
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("[INFO] Starting webcam obstacle detection...")
time.sleep(2.0)

prev_time = time.time()
fps = 0
prev_left, prev_front, prev_right = 640, 640, 640

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame not captured, exiting...")
        break

    # -------------------------------
    # YOLO Inference
    # -------------------------------
    results = model(frame, verbose=False)
    result = results[0]
    W, H = result.orig_shape[1], result.orig_shape[0]

    left_dist, front_dist, right_dist = W, W, W

    for box in result.boxes:
        cls_name = model.names[int(box.cls)]
        if cls_name not in OBSTACLE_CLASSES:
            continue
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        # Draw detection box
        color = (0, 255, 255)
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(frame, cls_name, (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Determine sector (left, front, right)
        if cx < W / 3:
            left_dist = min(left_dist, cx)
        elif cx < 2 * W / 3:
            front_dist = min(front_dist, cx)
        else:
            right_dist = min(right_dist, cx)

    # -------------------------------
    # Smooth the distances
    # -------------------------------
    left_dist = SMOOTHING_ALPHA * prev_left + (1 - SMOOTHING_ALPHA) * left_dist
    front_dist = SMOOTHING_ALPHA * prev_front + (1 - SMOOTHING_ALPHA) * front_dist
    right_dist = SMOOTHING_ALPHA * prev_right + (1 - SMOOTHING_ALPHA) * right_dist

    prev_left, prev_front, prev_right = left_dist, front_dist, right_dist

    # -------------------------------
    # Decision Making
    # -------------------------------
    if front_dist < D_THRESH:
        if left_dist > right_dist and left_dist > D_THRESH:
            action = "TURN_LEFT"
        elif right_dist > left_dist and right_dist > D_THRESH:
            action = "TURN_RIGHT"
        else:
            action = "STOP"
    else:
        action = "MOVE_FORWARD"

    # -------------------------------
    # Feedback
    # -------------------------------
    if SHOW_BARS:
        draw_feedback(frame, left_dist, front_dist, right_dist, W)

    if action == "MOVE_FORWARD":
        msg = "Path clear — move forward."
        pattern = "pulse"
        color = (0, 255, 0)
    elif action == "TURN_LEFT":
        msg = "Obstacle ahead — turn left."
        pattern = "vibrate-left"
        color = (255, 255, 0)
    elif action == "TURN_RIGHT":
        msg = "Obstacle ahead — turn right."
        pattern = "vibrate-right"
        color = (255, 0, 255)
    else:
        msg = "Stop! No safe path ahead."
        pattern = "long-buzz"
        color = (0, 0, 255)

    vibrate_pattern(pattern)

    # Display action text
    cv2.putText(frame, f"Action: {action}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # -------------------------------
    # FPS Calculation
    # -------------------------------
    now = time.time()
    fps = FPS_SMOOTHING * fps + (1 - FPS_SMOOTHING) * (1 / (now - prev_time))
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # -------------------------------
    # Show frame
    # -------------------------------
    cv2.imshow("YOLOv8 Obstacle Awareness", frame)

    # -------------------------------
    # Exit control
    # -------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        # optional: audio output only on-demand
        play_audio(msg)

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam session ended.")

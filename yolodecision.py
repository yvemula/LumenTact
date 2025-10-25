from ultralytics import YOLO
import os
import time

# -------------------------------
# Config
# -------------------------------
IMAGE_PATH = 'data/coco2017/val2017/000000000139.jpg'  # replace with your image
OBSTACLE_CLASSES = ["person", "chair", "table", "potted plant", "tv", "refrigerator", "vase"]  # define obstacles
D_THRESH = 50  # safe distance threshold (in pixels)

# -------------------------------
# Mock audio/haptic functions
# -------------------------------
def play_audio(msg):
    """Plays an audio message or sound cue."""
    # placeholder — real system could use `playsound` or `pyttsx3` for voice feedback
    print(f"[AUDIO] {msg}")

def vibrate_pattern(pattern):
    """Simulate haptic feedback pattern."""
    print(f"[HAPTIC] {pattern}")

def visual_feedback(left, front, right, max_w):
    """Show a simple text-based distance bar for debug."""
    def bar(dist):
        level = int((1 - min(dist / max_w, 1)) * 20)
        return "█" * level + "-" * (20 - level)
    print(f"[VISUAL] L:[{bar(left)}] F:[{bar(front)}] R:[{bar(right)}]")


# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO('yolov8n.pt')

# -------------------------------
# Run inference
# -------------------------------
results = model(IMAGE_PATH)
result = results[0]

W, H = result.orig_shape[1], result.orig_shape[0]  # width, height

left_dist = W
front_dist = W
right_dist = W

for box in result.boxes:
    cls_name = model.names[int(box.cls)]
    if cls_name not in OBSTACLE_CLASSES:
        continue
    xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
    cx = (xmin + xmax) / 2
    # Determine sector
    if cx < W/3:
        left_dist = min(left_dist, cx)
    elif cx < 2*W/3:
        front_dist = min(front_dist, cx)
    else:
        right_dist = min(right_dist, cx)


# -------------------------------
# Analyze obstacles
# -------------------------------
W, H = result.orig_shape[1], result.orig_shape[0]  # width, height

left_dist = W
front_dist = W
right_dist = W

for box in result.boxes:
    cls_name = model.names[int(box.cls)]
    if cls_name not in OBSTACLE_CLASSES:
        continue
    xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
    cx = (xmin + xmax) / 2
    # Determine sector
    if cx < W/3:
        left_dist = min(left_dist, cx)
    elif cx < 2*W/3:
        front_dist = min(front_dist, cx)
    else:
        right_dist = min(right_dist, cx)



# -------------------------------
# Decision making
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
visual_feedback(left_dist, front_dist, right_dist, W)

if action == "MOVE_FORWARD":
    play_audio("Path clear — move forward.")
    vibrate_pattern("pulse")
elif action == "TURN_LEFT":
    play_audio("Obstacle ahead — turn left.")
    vibrate_pattern("vibrate-left")
elif action == "TURN_RIGHT":
    play_audio("Obstacle ahead — turn right.")
    vibrate_pattern("vibrate-right")
elif action == "STOP":
    play_audio("Stop! No safe path ahead.")
    vibrate_pattern("long-buzz")

print(f"Decision: {action}")
print(f"Distances → Left: {left_dist:.1f}, Front: {front_dist:.1f}, Right: {right_dist:.1f}")

# -------------------------------
# show the image that we do the detection on
# -------------------------------
result.show()

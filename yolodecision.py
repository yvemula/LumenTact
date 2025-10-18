from ultralytics import YOLO

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
    print(f"[Audio] {msg}")

# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO('yolov8n.pt')

# -------------------------------
# Run inference
# -------------------------------
results = model(IMAGE_PATH)
result = results[0]
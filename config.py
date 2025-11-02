# LumenTact/config.py
"""
Central configuration file for the LumenTact project.
"""

# --- Detection Model Configuration ---
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.7
# Custom trained model (example path)
# DEFAULT_MODEL = "runs/detect/lumtact_run/weights/best.pt"


# --- Haptic Feedback Configuration ---

# Defines the base "threat" level of an object class.
# This is combined with distance (approximated by bounding box height)
# to get the final urgency score.
CLASS_URGENCY_MAP = {
    "person": 0.6,
    "car": 0.6,
    "bicycle": 0.4,
    "traffic light": 0.4,
    "stop sign": 0.4,
    "bench": 0.2,
    "DEFAULT": 0.1,
}

# Defines the normalized x-coordinate boundaries for haptic direction.
# [0.0] --LEFT-- [LEFT_BOUND] --CENTER-- [RIGHT_BOUND] --RIGHT-- [1.0]
DIRECTION_THRESHOLDS = {"LEFT_BOUND": 0.33, "RIGHT_BOUND": 0.67}

# Urgency score (calculated) to Intensity (1-5) mapping
INTENSITY_MAPPING = {
    (0.0, 0.3): 1,  # Urgency < 0.3
    (0.3, 0.5): 2,  # 0.3 <= Urgency < 0.5
    (0.5, 0.7): 3,  # 0.5 <= Urgency < 0.7
    (0.7, 0.9): 4,  # 0.7 <= Urgency < 0.9
    (0.9, 1.1): 5,  # Urgency >= 0.9
}


# --- Hardware Configuration ---

# Set the active hardware mode:
# "MOCK":   Prints feedback to console (default, safe)
# "SERIAL": Attempts to connect to a serial device (e.g., Arduino, ESP32)
HARDWARE_MODE = "MOCK"

# SerialController settings (only used if HARDWARE_MODE is "SERIAL")
SERIAL_PORT = "/dev/ttyUSB0"  # Example for Linux
# SERIAL_PORT = "COM3"       # Example for Windows
SERIAL_BAUD_RATE = 115200
SERIAL_TIMEOUT = 1

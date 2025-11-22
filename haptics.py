import sys
import os
import logging
from unittest.mock import MagicMock
from io import StringIO
import re

# Mocking the module structure for relative imports in the provided files
# to allow for execution in a single block.


# --- 1. Mock the Logger from LumenTact/utils.py ---
def setup_logger(name, level=logging.INFO):
    """Simple logger that prints to stdout with a specific format."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if a handler is already attached to avoid duplicates
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(name)-12.12s | %(levelname)-5.5s | %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


# --- 2. Define config and hardware modules ---
class config:
    """Contents of LumenTact/config.py"""

    CLASS_URGENCY_MAP = {
        "person": 0.6,
        "car": 0.6,
        "bicycle": 0.4,
        "traffic light": 0.4,
        "stop sign": 0.4,
        "bench": 0.2,
        "DEFAULT": 0.1,
    }
    DIRECTION_THRESHOLDS = {"LEFT_BOUND": 0.33, "RIGHT_BOUND": 0.67}
    INTENSITY_MAPPING = {
        (0.0, 0.3): 1,
        (0.3, 0.5): 2,
        (0.5, 0.7): 3,
        (0.7, 0.9): 4,
        (0.9, 1.1): 5,
    }
    HARDWARE_MODE = "MOCK"


class hardware:
    """Contents of LumenTact/hardware.py (MockController part)"""

    log = setup_logger("hardware_module")

    class BaseHapticController:
        def __init__(self):
            self.connected = False

        def connect(self):
            pass

        def disconnect(self):
            pass

        def send_feedback(self, signal_type, intensity, direction):
            pass

    class MockController(BaseHapticController):
        def connect(self):
            hardware.log.info("[MOCK_HW] Mock Haptic Controller CONNECTED.")
            self.connected = True
            return True

        def disconnect(self):
            hardware.log.info("[MOCK_HW] Mock Haptic Controller DISCONNECTED.")
            self.connected = False

        def send_feedback(self, signal_type, intensity, direction):
            if not self.connected:
                hardware.log.warn("[MOCK_HW] Cannot send feedback: Not connected.")
                return

            duration_ms = 100
            if signal_type == "STRONG VIBRATION":
                duration_ms = 300
            elif signal_type == "MILD PULSE":
                duration_ms = 150
            elif signal_type == "MILD TAP":
                duration_ms = 50

            intensity_scale = min(max(intensity / 5.0, 0.0), 1.0)

            # Print the formatted output used in the original file
            print(f"[MOCK_HW] -> VIBRATE: {direction} motor")
            print(f"[MOCK_HW]    -> TYPE: {signal_type}")
            print(
                f"[MOCK_HW]    -> INTENSITY: {intensity_scale * 100:.0f}% ({intensity}/5)"
            )
            print(f"[MOCK_HW]    -> DURATION: {duration_ms}ms")

    def get_controller():
        mode = config.HARDWARE_MODE.upper()
        if mode != "MOCK":
            hardware.log.warn(
                f"Unknown HARDWARE_MODE '{config.HARDWARE_MODE}'. Defaulting to MOCK."
            )
        return hardware.MockController()


# --- 3. Define the Haptics logic (LumenTact/haptics.py) ---
haptics_log = setup_logger("haptics_module")


def _get_signal_details(class_name, box_data):
    """
    Internal function to calculate urgency, intensity, and direction.
    """
    xc, yc, w, h = box_data

    # 1. Get base urgency from config
    base_urgency = config.CLASS_URGENCY_MAP.get(
        class_name, config.CLASS_URGENCY_MAP["DEFAULT"]
    )

    # 2. Calculate urgency score
    distance_modifier = h * 0.4  # h is normalized height (proxy for proximity)
    urgency_score = base_urgency + distance_modifier

    # 3. Map urgency score to intensity level using config
    intensity_level = 1  # Default
    for (min_urg, max_urg), intensity in config.INTENSITY_MAPPING.items():
        if min_urg <= urgency_score < max_urg:
            intensity_level = intensity
            break

    # 4. Determine direction using config
    direction = "CENTER"
    if xc < config.DIRECTION_THRESHOLDS["LEFT_BOUND"]:
        direction = "LEFT"
    elif xc > config.DIRECTION_THRESHOLDS["RIGHT_BOUND"]:
        direction = "RIGHT"

    return {
        "class_name": class_name,
        "box_data": box_data,
        "urgency": urgency_score,
        "intensity": intensity_level,
        "direction": direction,
    }


def process_frame_detections(all_detections, controller):
    """
    Processes all detections in a single frame, selects the most critical object,
    and sends the final haptic feedback signal via the provided controller.
    """
    if not controller or not controller.connected:
        log.error("[HAPTIC] Invalid or disconnected controller. Cannot process frame.")
        return None

    if not all_detections:
        log.debug("[HAPTIC] Frame analysis: No objects detected. Sending ALL_CLEAR.")
        # Send an "all-clear" signal to stop any ongoing feedback
        controller.send_feedback("ALL_CLEAR", 0, "NONE")
        return None

    processed_detections = [
        _get_signal_details(d["class_name"], d["box_data"]) for d in all_detections
    ]

    # Find the single most critical object to report
    most_critical = max(processed_detections, key=lambda d: d["urgency"])

    class_name = most_critical["class_name"]
    intensity = most_critical["intensity"]
    direction = most_critical["direction"]
    urgency = most_critical["urgency"]

    # Determine signal type
    signal_type = "MILD TAP"
    if intensity >= 3:
        signal_type = "STRONG VIBRATION"
    elif intensity == 2:
        signal_type = "MILD PULSE"

    log.info(
        f"[HAPTIC] Most critical: '{class_name}' (Urgency: {urgency:.2f}) -> {signal_type} @ {intensity} / {direction}"
    )

    # Send the final command to the hardware controller
    controller.send_feedback(signal_type, intensity, direction)

    return most_critical


if __name__ == "__main__":
    # Update test bed to use the new system
    from . import hardware

    log.info("--- Test Cases for Priority System ---")

    # Create a mock controller for testing
    test_controller = hardware.get_controller()  # Gets Mock by default
    test_controller.connect()

    log.info("\n--- Scenario 1: Close Car vs Far Person (Car wins) ---")
    test_detections_1 = [
        {"class_name": "car", "box_data": (0.5, 0.9, 0.4, 0.8)},
        {"class_name": "person", "box_data": (0.8, 0.7, 0.2, 0.3)},
        {"class_name": "bench", "box_data": (0.1, 0.5, 0.1, 0.1)},
    ]
    process_frame_detections(test_detections_1, test_controller)

    log.info("\n--- Scenario 2: All Clear (No detections) ---")
    # This should trigger the new ALL_CLEAR signal
    process_frame_detections([], test_controller)

    log.info("\n--- Scenario 3: Far Person vs Close Bench (Person wins) ---")
    test_detections_2 = [
        {"class_name": "person", "box_data": (0.2, 0.5, 0.1, 0.1)},  # Far person
        {"class_name": "bench", "box_data": (0.8, 0.8, 0.1, 0.8)},  # Close bench
    ]
    process_frame_detections(test_detections_2, test_controller)

    test_controller.disconnect()

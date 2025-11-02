# LumenTact/haptics.py
from . import config
from .utils import setup_logger

log = setup_logger(__name__)


def _get_signal_details(class_name, box_data):
    """
    Internal function to calculate urgency, intensity, and direction.
    Reads all parameters from config.py.
    """
    xc, yc, w, h = box_data

    # 1. Get base urgency from config
    base_urgency = config.CLASS_URGENCY_MAP.get(
        class_name, config.CLASS_URGENCY_MAP["DEFAULT"]
    )

    # 2. Calculate urgency score
    distance_modifier = h * 0.4
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
        # log.debug("[HAPTIC] Frame analysis: No objects detected.")
        # Optionally, send an "all-clear" signal
        # controller.send_feedback("ALL_CLEAR", 0, "NONE")
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

    log.info("\n--- Scenario 2: Far Person vs Close Bench (Person wins) ---")
    test_detections_2 = [
        {"class_name": "person", "box_data": (0.2, 0.5, 0.1, 0.1)},  # Far person
        {"class_name": "bench", "box_data": (0.8, 0.8, 0.1, 0.8)},  # Close bench
    ]
    process_frame_detections(test_detections_2, test_controller)

    test_controller.disconnect()

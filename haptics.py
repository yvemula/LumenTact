# LumenTact/haptics.py
from . import config
from .utils import setup_logger

log = setup_logger(__name__)


def _get_signal_details(class_name, box_data):
    """
    Internal function to calculate urgency, intensity, and direction
    based on the configuration settings.
    """
    xc, yc, w, h = box_data

    # 1. Get base urgency from config
    # Uses the dictionary from config.py
    base_urgency = config.CLASS_URGENCY_MAP.get(
        class_name, config.CLASS_URGENCY_MAP["DEFAULT"]
    )

    # 2. Calculate urgency score
    # h is normalized height (proxy for proximity) as defined in the logic
    distance_modifier = h * 0.4
    urgency_score = base_urgency + distance_modifier

    # 3. Map urgency score to intensity level using config
    intensity_level = 1  # Default
    for (min_urg, max_urg), intensity in config.INTENSITY_MAPPING.items():
        if min_urg <= urgency_score < max_urg:
            intensity_level = intensity
            break

    # 4. Determine direction using config thresholds
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
    # Validate controller connection
    if not controller or not controller.connected:
        log.error("[HAPTIC] Invalid or disconnected controller. Cannot process frame.")
        return None

    # Handle case with no detections
    if not all_detections:
        log.debug("[HAPTIC] Frame analysis: No objects detected. Sending ALL_CLEAR.")
        # Send an "all-clear" signal to stop any ongoing feedback
        controller.send_feedback("ALL_CLEAR", 0, "NONE")
        return None

    # Process urgency for all detections
    processed_detections = [
        _get_signal_details(d["class_name"], d["box_data"]) for d in all_detections
    ]

    # Find the single most critical object to report
    most_critical = max(processed_detections, key=lambda d: d["urgency"])

    class_name = most_critical["class_name"]
    intensity = most_critical["intensity"]
    direction = most_critical["direction"]
    urgency = most_critical["urgency"]

    # Determine signal type based on intensity
    # Logic preserved from the mock implementation
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

import time

# New configuration map for base urgency
CLASS_URGENCY_MAP = {
    "person": 0.6,
    "car": 0.6,
    "bicycle": 0.4,
    "traffic light": 0.4,
    "stop sign": 0.4,
    "bench": 0.2,
    # Add a default for unmapped classes
    "DEFAULT": 0.1
}

def calculate_urgency_and_intensity(class_name, box_data):
    """
    Calculates an urgency score and corresponding haptic intensity.

    :param class_name: The name of the detected object (e.g., 'person').
    :param box_data: The normalized bounding box data (x_center, y_center, width, height).
                     These values are normalized to the image dimensions [0, 1].
    :return: A tuple (urgency_score, intensity_level, direction)
    """
    xc, yc, w, h = box_data

    # 1. Determine Base Urgency
    base_urgency = CLASS_URGENCY_MAP.get(class_name, CLASS_URGENCY_MAP["DEFAULT"])

    # 2. Distance Modifier (normalized height h is a proxy for closeness/distance)
    distance_modifier = h * 0.4
    urgency_score = base_urgency + distance_modifier

    # 3. Determine Intensity Level
    if urgency_score < 0.3:
        intensity_level = 1
    elif urgency_score < 0.5:
        intensity_level = 2
    elif urgency_score < 0.7:
        intensity_level = 3
    elif urgency_score < 0.9:
        intensity_level = 4
    else:
        intensity_level = 5

    # 4. Determine Direction (Normalized x_center)
    direction = "CENTER"
    if xc < 0.33:
        direction = "LEFT"
    elif xc > 0.67:
        direction = "RIGHT"

    return urgency_score, intensity_level, direction


def generate_haptic_feedback(class_name, box_data):
    """
    Generates a mock haptic feedback signal based on the detected object's properties.
    """
    urgency, intensity, direction = calculate_urgency_and_intensity(class_name, box_data)

    signal_type = "MILD TAP"
    if intensity >= 3:
        signal_type = "STRONG VIBRATION"
    elif intensity == 2:
        signal_type = "MILD PULSE"
        
    feedback = f"{signal_type} @ INTENSITY {intensity} - LOCATION: {direction}"

    print(
        f"[HAPTIC] Detected '{class_name}' (normalized height: {box_data[3]:.2f}, direction: {direction}). Feedback: {feedback}"
    )

    return feedback


if __name__ == "__main__":
    print("--- Test Cases ---")
    generate_haptic_feedback("person", (0.5, 0.5, 0.05, 0.9))
    generate_haptic_feedback("car", (0.8, 0.9, 0.4, 0.2))
    generate_haptic_feedback("stop sign", (0.2, 0.7, 0.2, 0.5))
    generate_haptic_feedback("cat", (0.5, 0.5, 0.1, 0.3))
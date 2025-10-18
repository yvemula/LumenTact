import time


def calculate_urgency_and_intensity(class_name, box_data):
    """
    Calculates an urgency score and corresponding haptic intensity.

    :param class_name: The name of the detected object (e.g., 'person').
    :param box_data: The normalized bounding box data (x_center, y_center, width, height).
                     These values are normalized to the image dimensions [0, 1].
    :return: A tuple (urgency_score, intensity_level)
    """
    _, _, _, h = box_data

    base_urgency = 0.0
    if class_name in ["person", "car"]:
        base_urgency = 0.6
    elif class_name in ["bicycle", "traffic light", "stop sign"]:
        base_urgency = 0.4
    elif class_name in ["bench"]:
        base_urgency = 0.2

    distance_modifier = h * 0.4
    urgency_score = base_urgency + distance_modifier

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

    return urgency_score, intensity_level


def generate_haptic_feedback(class_name, box_data):
    """
    Generates a mock haptic feedback signal based on the detected object's properties.
    """
    urgency, intensity = calculate_urgency_and_intensity(class_name, box_data)

    signal_type = "MILD TAP"
    if intensity >= 3:
        signal_type = "STRONG VIBRATION"
    elif intensity == 2:
        signal_type = "MILD PULSE"

    feedback = f"{signal_type} @ INTENSITY {intensity}"

    print(
        f"[HAPTIC] Detected '{class_name}' (normalized height: {box_data[3]:.2f}). Feedback: {feedback}"
    )

    return feedback


if __name__ == "__main__":
    generate_haptic_feedback("person", (0.5, 0.5, 0.05, 0.1))
    generate_haptic_feedback("car", (0.5, 0.9, 0.4, 0.8))
    generate_haptic_feedback("stop sign", (0.2, 0.7, 0.2, 0.3))

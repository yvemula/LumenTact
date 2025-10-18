import time


def generate_haptic_feedback(class_name):
    """
    Generates a mock haptic feedback signal based on the detected object's class.

    In a real system, this function would interface with a haptic device API
    (e.g., via Bluetooth or USB) to send a physical vibration/pulse pattern.
    """
    signal = ""

    if class_name in ["person", "bicycle"]:
        signal = "PULSE (moving obstacle)"
    elif class_name in ["car", "traffic light", "stop sign"]:
        signal = "BUZZ (critical warning)"
    elif class_name in ["bench"]:
        signal = "SHORT TAP (static object)"
    else:
        signal = "MILD TING"

    print(f"[HAPTIC] Detected '{class_name}'. Generating feedback: {signal}")

    return signal


if __name__ == "__main__":
    generate_haptic_feedback("person")
    generate_haptic_feedback("stop sign")
    generate_haptic_feedback("cup")

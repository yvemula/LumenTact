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

def _get_signal_details(class_name, box_data):
    """
    Internal function to calculate urgency, intensity, and direction for a single detection.
    
    :param class_name: The name of the detected object (e.g., 'person').
    :param box_data: The normalized bounding box data (x_center, y_center, width, height).
    :return: A dictionary containing all calculated details for a single box.
    """
    xc, yc, w, h = box_data

    # 1. Determine Base Urgency from the map
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

    # 4. Determine Direction (Normalized x_center: LEFT, CENTER, or RIGHT)
    direction = "CENTER"
    if xc < 0.33:
        direction = "LEFT"
    elif xc > 0.67:
        direction = "RIGHT"
        
    return {
        'class_name': class_name,
        'box_data': box_data,
        'urgency': urgency_score,
        'intensity': intensity_level,
        'direction': direction
    }


def process_frame_detections(all_detections):
    """
    Processes all detections in a single frame, selects the most critical object 
    based on urgency score, and generates the final haptic feedback signal.
    
    This function prevents sensory overload by ensuring only ONE signal is generated per frame.

    :param all_detections: A list of dictionaries, where each dict is 
                           {'class_name': str, 'box_data': list [xc, yc, w, h]}.
    :return: The generated feedback string or None.
    """
    if not all_detections:
        print("[HAPTIC] Frame analysis: No objects detected. No feedback.")
        return None

    # Calculate full details (including urgency) for all detections
    processed_detections = [_get_signal_details(d['class_name'], d['box_data']) for d in all_detections]

    # Find the most urgent detection - this is the core of the priority system
    most_critical = max(processed_detections, key=lambda d: d['urgency'])

    # Extract final details
    class_name = most_critical['class_name']
    intensity = most_critical['intensity']
    direction = most_critical['direction']
    box_data = most_critical['box_data'] 

    # Generate feedback string based on the most critical object
    signal_type = "MILD TAP"
    if intensity >= 3:
        signal_type = "STRONG VIBRATION"
    elif intensity == 2:
        signal_type = "MILD PULSE"
        
    feedback = f"{signal_type} @ INTENSITY {intensity} - LOCATION: {direction}"

    print(
        f"[HAPTIC] Frame analysis complete. Most critical: '{class_name}' (Urgency: {most_critical['urgency']:.2f}, normalized height: {box_data[3]:.2f}, direction: {direction}). Feedback: {feedback}"
    )

    return feedback


if __name__ == "__main__":
    print("--- Test Cases for Priority System ---")
    
    print("\nScenario 1: Close Car vs Far Person (Car wins)")
    # Car: High urgency (0.92) -> Intensity 5, Center
    # Person: Medium urgency (0.72) -> Intensity 4, Right
    test_detections = [
        {"class_name": "car", "box_data": (0.5, 0.9, 0.4, 0.8)},
        {"class_name": "person", "box_data": (0.8, 0.7, 0.2, 0.3)},
        {"class_name": "bench", "box_data": (0.1, 0.5, 0.1, 0.1)},
    ]
    process_frame_detections(test_detections)
    
    print("\nScenario 2: Tie-break (Closer Traffic Light wins)")
    # Traffic Light 1: Urgency (0.6) -> Intensity 3, Left
    # Traffic Light 2: Urgency (0.72) -> Intensity 4, Right (This one is chosen)
    test_tie_break = [
        {"class_name": "traffic light", "box_data": (0.2, 0.5, 0.1, 0.5)},
        {"class_name": "traffic light", "box_data": (0.8, 0.8, 0.1, 0.8)},
    ]
    process_frame_detections(test_tie_break)
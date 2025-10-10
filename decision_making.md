## Decision Making in AI

**What is it?**

Decision Making is the important step between the model and the output to user. It is what we are trying to attain as our end goal for this semester. We are taking our model and trying to output left, right, forward, stop, as a baseline for a user who is visually impaired. From the data/information gathered from a camera/image/video we want incorporate decision making to accurately output the correct movement choice to the user. 

## Steps/Flow:

**Input:** Model outputs (obstacle map, object positions, path probabilities, etc.)

**Analyze:** Determine safe walking zones, obstacle proximity, and movement direction.

**Decision Logic:** Choose the optimal action (e.g., forward, left, right, stop).

**Feedback:** Trigger haptic or audio feedback accordingly.



## Pseudo Code

```

function decision_making(obstacle_map, free_space_map, D_thresh):

    # Step 1: Analyze surroundings
    front_distance  = average_distance_in_sector(obstacle_map, "front")
    left_distance   = average_distance_in_sector(obstacle_map, "left")
    right_distance  = average_distance_in_sector(obstacle_map, "right")

    # Step 2: Decision logic
    if front_distance < D_thresh:
        if left_distance > right_distance and left_distance > D_thresh:
            action = "TURN_LEFT"
        elif right_distance > left_distance and right_distance > D_thresh:
            action = "TURN_RIGHT"
        else:
            action = "STOP"
    else:
        action = "MOVE_FORWARD"

    # Step 3: Output feedback
    if action == "MOVE_FORWARD":
        send_haptic_feedback("none")
        play_audio("Forward")
    elif action == "TURN_LEFT":
        send_haptic_feedback("vibrate_left")
        play_audio("Turn Left")
    elif action == "TURN_RIGHT":
        send_haptic_feedback("vibrate_right")
        play_audio("Turn Right")
    elif action == "STOP":
        send_haptic_feedback("strong_vibration")
        play_audio("Stop - obstacle ahead")

    return action

```
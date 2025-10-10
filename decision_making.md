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
function decisionMaking(obstacleMap, freeSpaceMap, Dthresh):

    # Step 1: Analyze surroundings
    frontDistance = averageDistanceInSector(obstacleMap, "front")
    leftDistance  = averageDistanceInSector(obstacleMap, "left")
    rightDistance = averageDistanceInSector(obstacleMap, "right")

    # Step 2: Decision logic
    if frontDistance < Dthresh:
        if leftDistance > rightDistance and leftDistance > Dthresh:
            action = "TURNLEFT"
        elif rightDistance > leftDistance and rightDistance > Dthresh:
            action = "TURNRIGHT"
        else:
            action = "STOP"
    else:
        action = "MOVEFORWARD"

    # Step 3: Output feedback
    if action == "MOVEFORWARD":
        sendHapticFeedback("none")
        playAudio("Forward")
    elif action == "TURNLEFT":
        sendHapticFeedback("vibrateleft")
        playAudio("Turn Left")
    elif action == "TURNRIGHT":
        sendHapticFeedback("vibrateright")
        playAudio("Turn Right")
    elif action == "STOP":
        sendHapticFeedback("strongvibration")
        playAudio("Stop - obstacle ahead")

    return action


```
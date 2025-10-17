# ASCII belt emulator: indices 0..7 = [front-right quadrant clockwise], 0 = front
PATTERNS = {
    "STOP":      {"motors": list(range(8)), "seq": [(0.30, True), (0.30, False)]*2},
    "VEER_LEFT": {"motors": [6,7,0],        "seq": [(0.15, True), (0.15, False)]*3},
    "VEER_RIGHT":{"motors": [1,2,3],        "seq": [(0.15, True), (0.15, False)]*3},
    "STEP_UP":   {"motors": [7,0,1],        "seq": [(0.10, True), (0.10, False), (0.40, True)]},
    "STEP_DOWN": {"motors": [7,0,1],        "seq": [(0.40, True), (0.10, False), (0.10, True)]},
    "DUCK":      {"motors": [7,0,1,2],      "seq": [(0.60, True)]},
    "CAUTION":   {"motors": [0],            "seq": [(0.10, True), (0.40, False)]},
}

def render_belt(active_idxs):
    belt = ["·"]*8
    for i in active_idxs: belt[i]="●"
    # 0 front, show as [6 7] [0] [1 2] etc for intuition
    return f"[{belt[6]} {belt[7]}] [{belt[0]}] [{belt[1]} {belt[2]}]  [{belt[5]} {belt[4]}] [{belt[3]}]"

def play(action:str, strength:float=1.0):
    import time, sys
    cfg = PATTERNS.get(action, PATTERNS["CAUTION"])
    motors = cfg["motors"]
    seq = cfg["seq"]
    print(f"\nHAPTIC → {action} (intensity={strength:.2f})")
    for dur, on in seq:
        sys.stdout.write(("\r" if on else "\r") + render_belt(motors if on else []))
        sys.stdout.flush()
        time.sleep(dur * max(0.3, min(1.0, strength)))
    print("\r" + render_belt([]))

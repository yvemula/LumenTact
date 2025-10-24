# src/tests_decision.py
from decision import choose_action, NavThresholds, DEFAULT_PER_CLASS

W,H = 1280,720

def box(xc, hfrac):
    w = 0.1 * W
    h = hfrac * H
    x1 = (xc*W - w/2); x2 = (xc*W + w/2)
    y2 = H; y1 = y2 - h
    return (x1,y1,x2,y2)

def test_center_stop():
    dets = [("person", 0.9, box(0.5, 0.6))]
    assert choose_action(dets, W, H, DEFAULT_PER_CLASS) == "STOP"

def test_veer_left():
    dets = [("person",0.9, box(0.7,0.6))]
    assert choose_action(dets, W, H, DEFAULT_PER_CLASS) in ("VEER_LEFT","CAUTION","STOP")  # steering logic

def test_overhang_priority():
    dets = [("overhang",0.9, box(0.5,0.5)), ("person",0.9, box(0.5,0.6))]
    assert choose_action(dets, W, H, DEFAULT_PER_CLASS) == "DUCK"

if __name__ == "__main__":
    for f in [test_center_stop, test_veer_left, test_overhang_priority]:
        f()
    print("All decision tests passed.")

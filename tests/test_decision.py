from decision import decide_action

def mk(cx, conf=0.9): return {"bbox":(cx,0.5,0.2,0.4), "conf":conf}
def test_stop_center_threat():
    dets = [mk(0.5, 0.9)]
    a,_ = decide_action(dets, fps=30)
    assert a in ("STOP","VEER_LEFT","VEER_RIGHT")
def test_veer_left_when_right_crowded():
    dets = [mk(0.8,0.8), mk(0.75,0.7), mk(0.7,0.7)]
    a,_ = decide_action(dets, fps=30)
    assert a=="VEER_LEFT"

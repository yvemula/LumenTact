from decision import decide_action

def mk(cx, conf=0.9): return {"bbox":(cx,0.5,0.2,0.4), "conf":conf, "cls":"person"}

def test_stop_or_veer_on_center_threat():
    dets = [mk(0.5, 0.9)]
    action,_ = decide_action(dets, fps=30, frame_shape=(720,1280,3), trav_mask=None, depth_map=None)
    assert action in ("STOP","VEER_LEFT","VEER_RIGHT")

def test_veer_left_when_right_crowded():
    dets = [mk(0.8,0.8), mk(0.75,0.7), mk(0.7,0.7)]
    action,_ = decide_action(dets, fps=30, frame_shape=(720,1280,3), trav_mask=None, depth_map=None)
    assert action=="VEER_LEFT"

import numpy as np
from decision import DecisionEngine


def make_engine(**kwargs):
    return DecisionEngine(**kwargs)


def test_stop_when_center_depth_is_near():
    H, W = 240, 320
    mask = np.ones((H, W), dtype=np.uint8)
    depth = np.ones((H, W), dtype=np.float32)
    depth[int(H * 0.8):, int(W * 0.4):int(W * 0.6)] = 0.05  # near obstacle in center-bottom

    engine = make_engine(near_depth_stop=0.2)
    decision = engine.decide(mask, depth)

    assert decision.cmd == "STOP"
    assert "near obstacle" in decision.reason
    assert decision.confidence == 1.0


def test_left_command_when_corridor_on_left():
    H, W = 240, 320
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[int(H * 0.6):, : int(W * 0.3)] = 1  # traversable only on left side
    depth = np.ones((H, W), dtype=np.float32)

    engine = make_engine(deadband=0.05)
    decision = engine.decide(mask, depth)

    assert decision.cmd == "LEFT"
    assert decision.steer < 0
    assert 0.0 <= decision.confidence <= 1.0


def test_forward_when_corridor_centered():
    H, W = 240, 320
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[int(H * 0.6):, :] = 1  # uniform traversable floor
    depth = np.ones((H, W), dtype=np.float32)

    engine = make_engine(deadband=0.15)
    decision = engine.decide(mask, depth)

    assert decision.cmd == "FORWARD"
    assert abs(decision.steer) <= 0.3
    assert decision.confidence > 0.5

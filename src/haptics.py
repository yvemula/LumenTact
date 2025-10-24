# src/haptics.py
import time

class HapticsBase:
    def send(self, action: str, strength: float = 1.0): raise NotImplementedError

class NoOpHaptics(HapticsBase):
    def send(self, action: str, strength: float = 1.0):
        print(f"[HAPTIC] {action} ({strength:.2f})")

class DRV2605LHaptics(HapticsBase):
    """
    One motor per DRV2605L board. Wire ADDR pins to set unique I2C addresses if multiple.
    """
    def __init__(self, addresses=(0x5A,), intensity_scale=1.0):
        import board, busio
        from adafruit_drv2605 import DRV2605
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ds = [DRV2605(self.i2c, address=a) for a in addresses]
        for d in self.ds: d.use_erm = True
        self.intensity_scale = intensity_scale

    def _effect(self, d, effect_id, duration=0.2):
        d.sequence[0] = (effect_id, 0)
        d.sequence[1] = (0, 0)
        d.play(); time.sleep(duration); d.stop()

    def send(self, action: str, strength: float = 1.0):
        # Map actions to motor indices/effects; tune for your belt layout
        if action == "STOP":
            for d in self.ds: self._effect(d, 47, 0.25)        # strong click all
            time.sleep(0.1)
            for d in self.ds: self._effect(d, 47, 0.25)
        elif action == "VEER_LEFT":
            for i in [0,1,2][:len(self.ds)]: self._effect(self.ds[i], 12, 0.12)
        elif action == "VEER_RIGHT":
            idx = list(range(len(self.ds)))[-3:] or [0]
            for i in idx: self._effect(self.ds[i], 12, 0.12)
        elif action == "STEP_UP":
            for i in range(min(3,len(self.ds))):
                self._effect(self.ds[i], 14, 0.08)
                time.sleep(0.05)
            self._effect(self.ds[0], 70, 0.2)
        elif action == "STEP_DOWN":
            self._effect(self.ds[0], 70, 0.2)
            for i in range(min(3,len(self.ds))):
                self._effect(self.ds[i], 14, 0.08)
                time.sleep(0.05)
        elif action == "DUCK":
            for d in self.ds: self._effect(d, 70, 0.35)
        else:  # CAUTION
            self._effect(self.ds[0], 10, 0.12)

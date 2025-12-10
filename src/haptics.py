import time
import threading
import board
import busio
from adafruit_drv2605 import DRV2605

class DRV2605LHaptics(HapticsBase):
    def __init__(self, addresses=(0x5A,), motor_type="ERM"):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ds = []
        
        # Setup drivers (Assuming unique addresses or Mux handling elsewhere)
        for addr in addresses:
            try:
                d = DRV2605(self.i2c, address=addr)
                # "ERM" = Eccentric Rotating Mass (Standard vibration motor)
                # "LRA" = Linear Resonant Actuator (Coin style, requires calibration)
                d.use_erm = (motor_type == "ERM") 
                self.ds.append(d)
            except ValueError:
                print(f"[ERROR] Haptic driver at {hex(addr)} not found.")

        self.active_thread = None

    def _play_pattern(self, action):
        """Internal method to run sequences without blocking the main thread."""
        
        # --- Helper for triggering a specific motor ---
        def fire(motor_idx, effect_id, wait=0.2):
            if motor_idx < len(self.ds):
                d = self.ds[motor_idx]
                d.sequence[0] = (effect_id, 0) # Set effect
                d.sequence[1] = (0, 0)         # End sequence
                d.play()
                time.sleep(wait)
                d.stop()

        # --- Pattern Logic ---
        if action == "STOP":
            # Pulse all motors twice
            for _ in range(2):
                for i in range(len(self.ds)): fire(i, 47, 0) 
                time.sleep(0.3)
                
        elif action == "VEER_LEFT":
            # Fire left-most motors
            for i in [0, 1]: fire(i, 12, 0.1)

        elif action == "VEER_RIGHT":
            # Fire right-most motors (handle variable count)
            indices = list(range(len(self.ds)))[-2:]
            for i in indices: fire(i, 12, 0.1)
            
        elif action == "STEP_UP":
            # Ripple effect: 1 -> 2 -> 3
            for i in range(len(self.ds)):
                fire(i, 14, 0.1)

        # Default fallback
        else:
            fire(0, 10, 0.15)

    def send(self, action: str, strength: float = 1.0):
        """
        Public method: Fires the haptic pattern in a background thread.
        This prevents the 'sleeps' from freezing your main sensor loop.
        """
        # If a vibration is already happening, you might want to join it or ignore new ones
        if self.active_thread and self.active_thread.is_alive():
             # Option A: Ignore new command until old one finishes
             return 
             # Option B: In a real system, you might want to kill the old thread 
             # and start the new urgent one immediately.

        self.active_thread = threading.Thread(target=self._play_pattern, args=(action,))
        self.active_thread.start()
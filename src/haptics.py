import time
import threading

# Try importing hardware libs; handle failure gracefully for PC testing
try:
    import board
    import busio
    from adafruit_drv2605 import DRV2605
    HARDWARE_AVAILABLE = True
except (ImportError, NotImplementedError):
    HARDWARE_AVAILABLE = False

class HapticsBase:
    def send(self, action: str, strength: float = 1.0): 
        raise NotImplementedError

class NoOpHaptics(HapticsBase):
    """Mocks haptics by printing to console (for testing/debugging)."""
    def send(self, action: str, strength: float = 1.0):
        if action != "CAUTION": # Reduce log spam
            print(f"[HAPTIC MOCK] >> {action} << (Strength: {strength})")

class DRV2605LHaptics(HapticsBase):
    """
    Drives multiple DRV2605L motors via I2C.
    Uses threading to ensure motor delays do not block the main camera loop.
    """
    def __init__(self, addresses=(0x5A,), motor_type="ERM"):
        if not HARDWARE_AVAILABLE:
            raise RuntimeError("Board/Busio not found. Cannot init hardware.")
            
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ds = []
        
        print(f"[HAPTIC] Initializing {len(addresses)} motors...")
        for addr in addresses:
            try:
                d = DRV2605(self.i2c, address=addr)
                # ERM = Eccentric Rotating Mass (Standard vibration motor)
                # LRA = Linear Resonant Actuator (Coin style, requires calibration)
                d.use_erm = (motor_type == "ERM") 
                self.ds.append(d)
            except ValueError:
                print(f"[ERR] Haptic driver at {hex(addr)} not found on I2C bus.")

        self.active_thread = None

    def _play_pattern(self, action):
        """
        Internal worker function that runs in a separate thread.
        Effect IDs are from TI DRV2605L Datasheet.
        """
        def fire(indices, effect_id, duration=0.0):
            # Helper to fire specific motors
            targets = [self.ds[i] for i in indices if i < len(self.ds)]
            for d in targets:
                d.sequence[0] = (effect_id, 0) # Set effect
                d.sequence[1] = (0, 0)         # End sequence
                d.play()
            
            # Wait for effect to finish physically
            if duration > 0:
                time.sleep(duration)
                for d in targets: d.stop()

        # --- PATTERN LOGIC ---
        if action == "STOP":
            # Urgent: Strong double pulse on ALL motors
            # Effect 47: Pulsing Strong 1
            for _ in range(2):
                fire(range(len(self.ds)), 47, 0.3)
                time.sleep(0.1)
                
        elif action == "VEER_LEFT":
            # Sharp Triple Click on LEFT motors
            # Effect 12: Triple Click 100%
            fire([0, 1], 12, 0.25)

        elif action == "VEER_RIGHT":
            # Sharp Triple Click on RIGHT motors
            # Calculate last 2 indices dynamically
            cnt = len(self.ds)
            indices = [cnt-2, cnt-1] if cnt >= 2 else [0]
            fire(indices, 12, 0.25)
            
        elif action == "STEP_UP":
            # Ripple effect: 0 -> 1 -> 2
            # Effect 14: Strong Buzz 60%
            for i in range(len(self.ds)):
                fire([i], 14, 0.15)
                time.sleep(0.05)

        elif action == "DUCK":
            # Long buzz on all
            # Effect 15: Strong Buzz 80%
            fire(range(len(self.ds)), 15, 0.5)

        else: # CAUTION / DEFAULT
            # Soft bump on center
            fire([0], 10, 0.1)

    def send(self, action: str, strength: float = 1.0):
        """
        Public Trigger: Launches thread if one isn't already running.
        """
        # Prevent overlapping vibrations (optional: allow overlap for STOP)
        if self.active_thread and self.active_thread.is_alive():
            if action == "STOP": 
                # STOP overrides everything, let it queue or ignore
                pass 
            else:
                return # Ignore new commands while vibrating

        self.active_thread = threading.Thread(target=self._play_pattern, args=(action,))
        self.active_thread.start()
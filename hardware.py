# LumenTact/hardware.py
import time


class HapticController:
    """
    Mock Hardware Interface.

    This class simulates the connection to physical haptic motors.
    In a real implementation, this class would handle Serial, I2C, or GPIO communication.
    """

    def __init__(self, num_motors=3):
        self.num_motors = num_motors
        self.motor_pins = {"LEFT": 0, "CENTER": 1, "RIGHT": 2}
        print(f"[HARDWARE_MOCK] Initialized HapticController with {num_motors} motors.")

    def _activate_motor(self, motor_name, intensity_level, duration_ms):
        """Simulates activating a single motor."""
        motor_id = self.motor_pins.get(motor_name)
        if motor_id is None:
            return

        # Convert intensity (1-5) to a voltage/PWM scale (0.0 - 1.0)
        intensity_scale = min(max(intensity_level / 5.0, 0.0), 1.0)

        print(f"[HARDWARE_MOCK] -> VIBRATE: Motor {motor_name} (ID: {motor_id})")
        print(
            f"[HARDWARE_MOCK]    -> INTENSITY: {intensity_scale * 100:.0f}% ({intensity_level}/5)"
        )
        print(f"[HARDWARE_MOCK]    -> DURATION: {duration_ms}ms")

        # In a real app, you would start the motor, time.sleep(duration_ms / 1000.0), and stop it.
        # time.sleep(duration_ms / 1000.0)

    def send_feedback(self, signal_type, intensity, direction):
        """
        Translates a haptic signal into a specific motor command.
        """
        duration_ms = 100  # Default duration

        if signal_type == "STRONG VIBRATION":
            duration_ms = 300
        elif signal_type == "MILD PULSE":
            duration_ms = 150
        elif signal_type == "MILD TAP":
            duration_ms = 50

        # Send command to the correct motor
        self._activate_motor(direction, intensity, duration_ms)


# Singleton instance to be used by other modules
# This prevents re-initializing the "hardware" connection every time.
GLOBAL_CONTROLLER = HapticController()

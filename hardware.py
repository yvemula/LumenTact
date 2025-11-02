# LumenTact/hardware.py
import time
import abc
from . import config
from .utils import setup_logger

log = setup_logger(__name__)


class BaseHapticController(abc.ABC):
    """
    Abstract Base Class for all haptic controllers.
    Defines the standard interface for connecting, disconnecting,
    and sending feedback.
    """

    def __init__(self):
        self.connected = False
        log.info(f"Initialized {self.__class__.__name__}")

    @abc.abstractmethod
    def connect(self):
        """Establish connection to the hardware."""
        pass

    @abc.abstractmethod
    def disconnect(self):
        """Safely close connection to the hardware."""
        pass

    @abc.abstractmethod
    def send_feedback(self, signal_type, intensity, direction):
        """
        Translates a haptic signal into a specific motor command.
        """
        pass


class MockController(BaseHapticController):
    """
    Mock Hardware Interface for development and testing.
    Prints all haptic commands to the console instead of activating hardware.
    """

    def connect(self):
        log.info("[MOCK_HW] Mock Haptic Controller CONNECTED.")
        self.connected = True
        return True

    def disconnect(self):
        log.info("[MOCK_HW] Mock Haptic Controller DISCONNECTED.")
        self.connected = False

    def send_feedback(self, signal_type, intensity, direction):
        if not self.connected:
            log.warn("[MOCK_HW] Cannot send feedback: Not connected.")
            return

        duration_ms = 100
        if signal_type == "STRONG VIBRATION":
            duration_ms = 300
        elif signal_type == "MILD PULSE":
            duration_ms = 150
        elif signal_type == "MILD TAP":
            duration_ms = 50

        intensity_scale = min(max(intensity / 5.0, 0.0), 1.0)

        log.info(f"[MOCK_HW] -> VIBRATE: {direction} motor")
        log.info(f"[MOCK_HW]    -> TYPE: {signal_type}")
        log.info(
            f"[MOCK_HW]    -> INTENSITY: {intensity_scale * 100:.0f}% ({intensity}/5)"
        )
        log.info(f"[MOCK_HW]    -> DURATION: {duration_ms}ms")


class SerialController(BaseHapticController):
    """
    Hardware interface for a serial device (Arduino, ESP32, etc.).

    This controller expects the serial device to understand
    a simple command protocol, e.g., "DIRECTION,INTENSITY,DURATION\n"
    Example: "L,5,300\n" (Left, Intensity 5, 300ms)
    """

    def __init__(self, port, baud_rate, timeout):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_connection = None

    def connect(self):
        try:
            # Dynamically import serial, so it's not a hard dependency
            # for mock-only users.
            import serial

            self.serial_connection = serial.Serial(
                self.port, self.baud_rate, timeout=self.timeout
            )
            time.sleep(2)  # Wait for serial connection to establish
            log.info(
                f"[SERIAL_HW] Successfully connected to {self.port} at {self.baud_rate} baud."
            )
            self.connected = True
            return True
        except ImportError:
            log.error("[SERIAL_HW] 'pyserial' library not found.")
            log.error("Please run 'pip install pyserial' to use the SerialController.")
            self.connected = False
            return False
        except serial.SerialException as e:
            log.error(f"[SERIAL_HW] Failed to connect to {self.port}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            log.info(f"[SERIAL_HW] Disconnected from {self.port}.")
        self.connected = False

    def send_feedback(self, signal_type, intensity, direction):
        if not self.connected:
            log.warn("[SERIAL_HW] Cannot send feedback: Not connected.")
            return

        # 1. Determine duration
        duration_ms = 100
        if signal_type == "STRONG VIBRATION":
            duration_ms = 300
        elif signal_type == "MILD PULSE":
            duration_ms = 150
        elif signal_type == "MILD TAP":
            duration_ms = 50

        # 2. Map direction to a command code (e.g., L, C, R)
        dir_code = direction[0].upper()  # L, C, R

        # 3. Create command string
        #    FORMAT: "DIRECTION,INTENSITY,DURATION\n"
        #    This is just an example; customize it to match your device's firmware.
        command = f"{dir_code},{intensity},{duration_ms}\n"

        try:
            self.serial_connection.write(command.encode("ascii"))
            log.debug(f"[SERIAL_HW] Sent command: {command.strip()}")
        except Exception as e:
            log.error(f"[SERIAL_HW] Error writing to serial port: {e}")
            # Attempt to disconnect to prevent further errors
            self.disconnect()


def get_controller():
    """
    Factory function to get the correct hardware controller
    based on the settings in config.py.
    """
    mode = config.HARDWARE_MODE.upper()

    if mode == "SERIAL":
        log.info("Hardware mode: SERIAL")
        return SerialController(
            port=config.SERIAL_PORT,
            baud_rate=config.SERIAL_BAUD_RATE,
            timeout=config.SERIAL_TIMEOUT,
        )

    if mode != "MOCK":
        log.warn(f"Unknown HARDWARE_MODE '{config.HARDWARE_MODE}'. Defaulting to MOCK.")

    log.info("Hardware mode: MOCK")
    return MockController()

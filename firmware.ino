/*
 * LumenTact Firmware
 * Compatible with: Arduino Uno, Nano, ESP32, etc.
 * * Protocol: "DIRECTION,INTENSITY,DURATION\n"
 * Example:  "L,5,300" -> Left Motor, Max Intensity, 300ms
 * Directions: L (Left), C (Center), R (Right)
 */

// --- Pin Configuration (Adjust for your specific board) ---
const int PIN_LEFT   = 9;  // PWM Capable Pin
const int PIN_CENTER = 10; // PWM Capable Pin
const int PIN_RIGHT  = 11; // PWM Capable Pin

// --- State Variables ---
unsigned long motorEndTime[3] = {0, 0, 0}; // Timestamp when motor should stop
const int MOTOR_L = 0;
const int MOTOR_C = 1;
const int MOTOR_R = 2;

void setup() {
  // Initialize Serial to match config.py (115200 baud)
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }

  // Initialize Motor Pins
  pinMode(PIN_LEFT, OUTPUT);
  pinMode(PIN_CENTER, OUTPUT);
  pinMode(PIN_RIGHT, OUTPUT);

  // Startup Sequence (Vibrate all briefly)
  digitalWrite(PIN_CENTER, HIGH);
  delay(200);
  digitalWrite(PIN_CENTER, LOW);
  
  Serial.println("LumenTact Firmware Ready");
}

void loop() {
  // 1. Process Incoming Serial Commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    parseCommand(command);
  }

  // 2. Update Motor States (Non-blocking)
  unsigned long currentMillis = millis();
  
  updateMotor(PIN_LEFT,   MOTOR_L, currentMillis);
  updateMotor(PIN_CENTER, MOTOR_C, currentMillis);
  updateMotor(PIN_RIGHT,  MOTOR_R, currentMillis);
}

void parseCommand(String cmd) {
  // Expected Format: "D,I,T" (Direction, Intensity, Time)
  // Example: "L,5,300"
  
  int firstComma = cmd.indexOf(',');
  int secondComma = cmd.indexOf(',', firstComma + 1);

  if (firstComma == -1 || secondComma == -1) return; // Invalid format

  String dirStr = cmd.substring(0, firstComma);
  String intStr = cmd.substring(firstComma + 1, secondComma);
  String durStr = cmd.substring(secondComma + 1);

  char direction = dirStr.charAt(0);
  int intensity = intStr.toInt(); // 1-5
  int duration = durStr.toInt();  // Milliseconds

  // Map 1-5 intensity to PWM (0-255)
  int pwmValue = map(intensity, 1, 5, 100, 255);
  if (intensity <= 0) pwmValue = 0;

  // Activate the correct motor
  unsigned long endTime = millis() + duration;

  if (direction == 'L') {
    activateMotor(PIN_LEFT, MOTOR_L, pwmValue, endTime);
  } else if (direction == 'C') {
    activateMotor(PIN_CENTER, MOTOR_C, pwmValue, endTime);
  } else if (direction == 'R') {
    activateMotor(PIN_RIGHT, MOTOR_R, pwmValue, endTime);
  }
}

void activateMotor(int pin, int motorIndex, int pwm, unsigned long endTime) {
  analogWrite(pin, pwm);
  motorEndTime[motorIndex] = endTime;
}

void updateMotor(int pin, int motorIndex, unsigned long currentMillis) {
  if (motorEndTime[motorIndex] > 0) {
    if (currentMillis >= motorEndTime[motorIndex]) {
      // Time is up, turn off motor
      analogWrite(pin, 0);
      motorEndTime[motorIndex] = 0;
    }
  }
}
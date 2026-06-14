#define USE_US_TIMER

#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>

// =========================
// Wi-Fi Settings
// =========================
const char *WIFI_SSID = "EdNet";
const char *WIFI_PASSWORD = "Huawei@123";

// =========================
// MQTT Settings — must match src/recognize_mqtt.py
//   Broker: 157.173.101.159:1883
//   Topic:  vision/corene/movement  (plain text, QoS 0)
//   Commands from Python:
//     LEFT   — pan while face is left of center
//     RIGHT  — pan while face is right of center
//     CENTER — hold (no sweep)
//     SEARCH — sweep until IDLE/CENTER/LEFT/RIGHT received
//     IDLE   — hold position (stop search/track)
//   Python SEARCH heartbeat: default 0.2s (--mqtt-search-heartbeat)
//   Python min republish:    default 0.15s (--mqtt-min-interval)
//   Keep COMMAND_TIMEOUT_MS well above those intervals.
// =========================
const char *MQTT_SERVER = "157.173.101.159";
const uint16_t MQTT_PORT = 1883;

const char* MQTT_TOPIC = "vision/corene/movement";
const char* MQTT_CLIENT_ID_PREFIX = "corene-face-servo";
const IPAddress MQTT_FALLBACK_IPS[] = {
  IPAddress(3, 126, 147, 153),
  IPAddress(3, 124, 122, 176),
  IPAddress(18, 197, 232, 142),
  IPAddress(3, 123, 123, 192),
  IPAddress(3, 120, 204, 188)
};
const uint8_t MQTT_FALLBACK_IP_COUNT =
    sizeof(MQTT_FALLBACK_IPS) / sizeof(MQTT_FALLBACK_IPS[0]);

// =========================
// Servo Configuration
// =========================
const uint8_t SERVO_PIN = 14;

const int SERVO_MIN_ANGLE = 0;
const int SERVO_MAX_ANGLE = 180;
const int SERVO_CENTER_ANGLE = 90;
const int SERVO_MIN_PULSE_US = 500;
const int SERVO_MAX_PULSE_US = 2400;

// Tracking — follow LEFT/RIGHT from recognize_mqtt (head in frame)
const float TRACK_STEP = 0.2f;
const unsigned long TRACK_INTERVAL_MS = 40;

// Search — faster sweep while face is lost (IDLE from PC stops immediately)
const float SCAN_STEP = 2.0f;
const unsigned long SCAN_INTERVAL_MS = 80;

// > recognize_mqtt --mqtt-search-heartbeat (0.2s) and --mqtt-min-interval (0.15s)
const unsigned long COMMAND_TIMEOUT_MS = 5000;

const bool REVERSE_SERVO = true;

// Serial debugging
const bool DEBUG_SERVO = true;
const unsigned long DEBUG_INTERVAL_MS = 500;  // Print every 500ms during search

// =========================
// Command Types - Matching Python recognizer
// =========================
enum MovementCommand {
  CMD_IDLE,
  CMD_LEFT,
  CMD_RIGHT,
  CMD_CENTER,
  CMD_SEARCH
};

// =========================
// Global Objects
// =========================
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
Servo panServo;

// =========================
// State Variables
// =========================
MovementCommand currentCommand = CMD_IDLE;
MovementCommand previousCommand = CMD_IDLE;

float servoAngle = SERVO_CENTER_ANGLE;
int sweepDirection = 1;

unsigned long lastMoveAt = 0;
unsigned long lastReconnectAttempt = 0;
unsigned long lastCommandAt = 0;
unsigned long lastDebugAt = 0;
String mqttClientId = "";

// ======================================================
// Servo Functions
// ======================================================

int angleToPulse(float angle) {
  float normalized =
      (angle - SERVO_MIN_ANGLE) /
      float(SERVO_MAX_ANGLE - SERVO_MIN_ANGLE);

  return SERVO_MIN_PULSE_US +
      int((normalized * (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US)) + 0.5f);
}

void setServoAngle(float angle) {
  float oldAngle = servoAngle;
  
  if (angle < SERVO_MIN_ANGLE) {
    angle = SERVO_MIN_ANGLE;
  }

  if (angle > SERVO_MAX_ANGLE) {
    angle = SERVO_MAX_ANGLE;
  }

  servoAngle = angle;
  panServo.writeMicroseconds(angleToPulse(servoAngle));
  
  // Print angle change only for significant movements (>0.5°)
  if (DEBUG_SERVO && abs(servoAngle - oldAngle) > 0.5f) {
    Serial.print("[SERVO] ");
    Serial.print(oldAngle, 1);
    Serial.print("° → ");
    Serial.print(servoAngle, 1);
    Serial.print("° (Δ=");
    Serial.print(servoAngle - oldAngle, 1);
    Serial.println("°)");
  }
}

void applyTrackingStep(int logicalDirection) {
  int direction = REVERSE_SERVO ? -logicalDirection : logicalDirection;
  setServoAngle(servoAngle + (direction * TRACK_STEP));
}

// ======================================================
// Command Parsing
// ======================================================

MovementCommand parseCommand(String message) {
  message.trim();
  message.toUpperCase();

  if (message.startsWith("CMD_")) {
    message = message.substring(4);
  }

  // Primary strings (recognize_mqtt.py MOVEMENT_* constants)
  if (message == "LEFT")   return CMD_LEFT;
  if (message == "RIGHT")  return CMD_RIGHT;
  if (message == "CENTER") return CMD_CENTER;
  if (message == "SEARCH") return CMD_SEARCH;
  if (message == "IDLE")   return CMD_IDLE;

  // Aliases (BENAX / dashboard normalizeMovement)
  if (message == "MOVED_LEFT" || message == "MOVE_LEFT")   return CMD_LEFT;
  if (message == "MOVED_RIGHT" || message == "MOVE_RIGHT") return CMD_RIGHT;
  if (message == "CENTERED" || message == "STOP" || message == "STOPPED") return CMD_IDLE;
  if (message == "SEARCHING" || message == "SCAN" || message == "OUT_OF_FRAME") return CMD_SEARCH;

  return CMD_IDLE;
}

// ======================================================
// Helper: Command to String
// ======================================================

const char* commandToString(MovementCommand cmd) {
  switch(cmd) {
    case CMD_IDLE:   return "IDLE";
    case CMD_LEFT:   return "LEFT";
    case CMD_RIGHT:  return "RIGHT";
    case CMD_CENTER: return "CENTER";
    case CMD_SEARCH: return "SEARCH";
    default:         return "UNKNOWN";
  }
}

// ======================================================
// MQTT Callback
// ======================================================

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  previousCommand = currentCommand;
  currentCommand = parseCommand(message);
  lastCommandAt = millis();

  if (currentCommand != previousCommand) {
    Serial.println();
    Serial.println("┌─────────────────────────────────────────┐");
    Serial.print("│ [MQTT] ");
    Serial.print(commandToString(previousCommand));
    Serial.print(" → ");
    Serial.print(commandToString(currentCommand));
    Serial.println("                    │");
    Serial.print("│ Raw: ");
    Serial.print(message);
    Serial.println("                          │");
    Serial.println("└─────────────────────────────────────────┘");
  }

  // Refresh servo hold when IDLE/CENTER stops an active SEARCH sweep
  if (currentCommand == CMD_IDLE || currentCommand == CMD_CENTER) {
    panServo.writeMicroseconds(angleToPulse(servoAngle));
  }
}

// ======================================================
// Serial Input (for manual testing)
// ======================================================

void handleSerial() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    previousCommand = currentCommand;
    MovementCommand newCmd = parseCommand(input);
    currentCommand = newCmd;
    lastCommandAt = millis();

    Serial.print("[SERIAL] ");
    Serial.print(commandToString(previousCommand));
    Serial.print(" → ");
    Serial.print(commandToString(currentCommand));
    Serial.print(" | Angle: ");
    Serial.print(servoAngle, 1);
    Serial.println("°");
  }
}

// ======================================================
// Wi-Fi Connection
// ======================================================

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.println("[WiFi] Connecting...");

  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(1000);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startAttemptTime = millis();

  while (
      WiFi.status() != WL_CONNECTED &&
      millis() - startAttemptTime < 15000
  ) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("[WiFi] Connected!");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("[WiFi] RSSI: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    Serial.println();
    Serial.println("[WiFi] Connection Failed");
  }
}

// ======================================================
// MQTT Connection
// ======================================================

bool connectMqtt() {
  if (mqttClient.connected()) {
    return true;
  }

  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  if (millis() - lastReconnectAttempt < 5000) {
    return false;
  }

  lastReconnectAttempt = millis();

  IPAddress brokerIp;
  bool dnsResolved = false;
  bool connectedProbe = false;

  Serial.print("[MQTT] Resolving ");
  Serial.print(MQTT_SERVER);
  Serial.print("...");
  if (WiFi.hostByName(MQTT_SERVER, brokerIp)) {
    dnsResolved = true;
    Serial.print(" ");
    Serial.println(brokerIp);
  } else {
    Serial.println(" DNS failed");
  }

  for (uint8_t attempt = 0; attempt <= MQTT_FALLBACK_IP_COUNT; attempt++) {
    if (!dnsResolved || attempt > 0) {
      uint8_t fallbackIndex = dnsResolved ? attempt - 1 : attempt;
      if (fallbackIndex >= MQTT_FALLBACK_IP_COUNT) {
        break;
      }
      brokerIp = MQTT_FALLBACK_IPS[fallbackIndex];
      Serial.print("[MQTT] DNS fallback IP ");
      Serial.println(brokerIp);
    }

    WiFiClient probe;
    Serial.print("[MQTT] TCP probe ");
    Serial.print(brokerIp);
    Serial.print(":");
    Serial.print(MQTT_PORT);
    Serial.print("...");
    if (probe.connect(brokerIp, MQTT_PORT)) {
      Serial.println(" ok");
      probe.stop();
      connectedProbe = true;
      break;
    }

    Serial.println(" failed");
    probe.stop();
  }

  if (!connectedProbe) {
    return false;
  }

  mqttClient.setServer(brokerIp, MQTT_PORT);
  Serial.print("[MQTT] Connecting as ");
  Serial.print(mqttClientId);
  Serial.print("...");

  bool connected = mqttClient.connect(mqttClientId.c_str());

  if (!connected) {
    Serial.print("Failed, rc=");
    Serial.println(mqttClient.state());
    return false;
  }

  Serial.println("Connected");
  mqttClient.subscribe(MQTT_TOPIC);
  Serial.print("[MQTT] Subscribed to: ");
  Serial.println(MQTT_TOPIC);

  return true;
}

// ======================================================
// Servo Logic - Matching Python recognizer state machine
// ======================================================

void handleServo() {
  unsigned long now = millis();

  // Auto idle timeout
  if ((now - lastCommandAt) > COMMAND_TIMEOUT_MS) {
    if (currentCommand != CMD_IDLE && DEBUG_SERVO) {
      Serial.println();
      Serial.print("[WATCHDOG] No command for ");
      Serial.print((now - lastCommandAt) / 1000);
      Serial.println("s → forcing IDLE");
      Serial.println();
    }
    previousCommand = currentCommand;
    currentCommand = CMD_IDLE;
  }

  // ──────────────────────────────────────
  // IDLE: Hold position completely
  // ──────────────────────────────────────
  if (currentCommand == CMD_IDLE) {
    return;  // Do nothing, hold current angle
  }

  // ──────────────────────────────────────
  // CENTER: Actively move toward center
  // ──────────────────────────────────────
  if (currentCommand == CMD_CENTER) {
    // if (abs(servoAngle - SERVO_CENTER_ANGLE) > TRACK_STEP) {
    //   if (servoAngle < SERVO_CENTER_ANGLE) {
    //     setServoAngle(servoAngle + TRACK_STEP);
    //   } else {
    //     setServoAngle(servoAngle - TRACK_STEP);
    //   }
    // }
    return;
  }

  // ──────────────────────────────────────
  // SEARCH: Sweep back and forth; recognize_mqtt SEARCH heartbeat every 0.2s
  // ──────────────────────────────────────
  if (currentCommand == CMD_SEARCH) {
    if (now - lastMoveAt < SCAN_INTERVAL_MS) {
      return;
    }
    lastMoveAt = now;

    setServoAngle(servoAngle + (sweepDirection * SCAN_STEP));

    // Reverse direction at limits
    if (servoAngle >= SERVO_MAX_ANGLE) {
      sweepDirection = -1;
      if (DEBUG_SERVO) {
        Serial.println();
        Serial.println("┌───────────────────────────────────────┐");
        Serial.println("│ ▲ REACHED MAX (180°)                  │");
        Serial.println("│ ◄ REVERSING ← LEFT                    │");
        Serial.println("└───────────────────────────────────────┘");
      }
    }

    if (servoAngle <= SERVO_MIN_ANGLE) {
      sweepDirection = 1;
      if (DEBUG_SERVO) {
        Serial.println();
        Serial.println("┌───────────────────────────────────────┐");
        Serial.println("│ ▼ REACHED MIN (0°)                    │");
        Serial.println("│ ► REVERSING → RIGHT                   │");
        Serial.println("└───────────────────────────────────────┘");
      }
    }

    // Print position periodically
    if (DEBUG_SERVO && (now - lastDebugAt >= DEBUG_INTERVAL_MS)) {
      lastDebugAt = now;
      Serial.print("[SEARCH] ");
      Serial.print(sweepDirection > 0 ? "→" : "←");
      Serial.print(" Angle: ");
      Serial.print(servoAngle, 1);
      Serial.print("°");
      
      // Progress bar
      int barLen = 20;
      int filled = map((int)servoAngle, 0, 180, 0, barLen);
      Serial.print(" [");
      for (int i = 0; i < barLen; i++) {
        Serial.print(i < filled ? "█" : "░");
      }
      Serial.println("]");
    }

    return;
  }

  // ──────────────────────────────────────
  // LEFT / RIGHT: Fine tracking
  // ──────────────────────────────────────
  if (now - lastMoveAt < TRACK_INTERVAL_MS) {
    return;
  }
  lastMoveAt = now;

  if (currentCommand == CMD_LEFT) {
    applyTrackingStep(-1);
  } else if (currentCommand == CMD_RIGHT) {
    applyTrackingStep(1);
  }
}

// ======================================================
// Setup
// ======================================================

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("═══════════════════════════════════════════");
  Serial.println("  ESP32 Face Servo Controller");
  Serial.println("  MQTT-Controlled Tracking & Search");
  Serial.println("═══════════════════════════════════════════");
  Serial.println();
  
  mqttClientId =
      String(MQTT_CLIENT_ID_PREFIX) +
      "-" +
      String((uint32_t)ESP.getEfuseMac(), HEX);
  Serial.print("[SYS] MQTT Client ID: ");
  Serial.println(mqttClientId);

  // ESP32 Servo Setup
  ESP32PWM::allocateTimer(0);
  panServo.setPeriodHertz(50);
  panServo.attach(SERVO_PIN, SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US);

  setServoAngle(SERVO_CENTER_ANGLE);
  Serial.println();
  Serial.println("[SYS] ── Configuration ──");
  Serial.print("[SYS] Servo pin: ");
  Serial.println(SERVO_PIN);
  Serial.print("[SYS] Center: ");
  Serial.print(SERVO_CENTER_ANGLE);
  Serial.println("°");
  Serial.print("[SYS] Track: ");
  Serial.print(TRACK_STEP, 2);
  Serial.print("° every ");
  Serial.print(TRACK_INTERVAL_MS);
  Serial.println("ms (face following)");
  Serial.print("[SYS] Search: ");
  Serial.print(SCAN_STEP, 1);
  Serial.print("° every ");
  Serial.print(SCAN_INTERVAL_MS);
  Serial.println("ms (face finding)");
  Serial.print("[SYS] Full sweep: ~");
  Serial.print((180.0 / SCAN_STEP) * SCAN_INTERVAL_MS / 1000.0, 1);
  Serial.println(" seconds");
  Serial.print("[SYS] Timeout: ");
  Serial.print(COMMAND_TIMEOUT_MS / 1000);
  Serial.println(" seconds");

  // MQTT Setup
  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);

  // WiFi
  connectWiFi();

  lastCommandAt = millis();
  Serial.println();
  Serial.println("═══════════════════════════════════════════");
  Serial.println("  COMMANDS:");
  Serial.println("    IDLE   = Hold position");
  Serial.println("    CENTER = Return to 90°");
  Serial.println("    LEFT   = Track face (slow steps)");
  Serial.println("    RIGHT  = Track face (slow steps)");
  Serial.println("    SEARCH = Sweep 0-180° (~7 sec)");
  Serial.println("═══════════════════════════════════════════");
  Serial.println();
  Serial.println("[SYS] Ready. Waiting for MQTT commands...");
  Serial.println();
}

// ======================================================
// Main Loop
// ======================================================

void loop() {
  // Wi-Fi reconnect
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  // MQTT reconnect
  if (!mqttClient.connected()) {
    connectMqtt();
  }

  mqttClient.loop();

  // Serial commands (for testing: type LEFT, RIGHT, CENTER, SEARCH, IDLE)
  handleSerial();

  // Servo movement
  handleServo();

  delay(1);
}
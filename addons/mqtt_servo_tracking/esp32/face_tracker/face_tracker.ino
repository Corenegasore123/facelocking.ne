/*
 * Corene ESP32 pan servo — MQTT subscriber (ESP32 Dev Module only).
 *
 * Wiring: brown -> GND, red -> 5V/VIN, yellow signal -> D14 (GPIO14)
 *
 * Subscribes: vision/Corene/servo_control
 * Commands: MOVED_LEFT, MOVED_RIGHT, CENTERED, SEARCHING, STOPPED
 */
#define USE_US_TIMER

#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>

const char* WIFI_SSID = "EdNet";
const char* WIFI_PASSWORD = "Huawei@123";

const char* MQTT_SERVER = "157.173.101.159";
const uint16_t MQTT_PORT = 1883;
const char* MQTT_TOPIC = "vision/Corene/servo_control";
const char* MQTT_CLIENT_ID_PREFIX = "corene-face-servo";

const uint8_t SERVO_PIN = 14;  // D14 / GPIO14

const int SERVO_MIN_ANGLE = 0;
const int SERVO_MAX_ANGLE = 180;
const int SERVO_CENTER_ANGLE = 90;
const int TRACK_STEP = 1;
const int SEARCH_STEP = 1;

const unsigned long TRACK_INTERVAL_MS = 55;
const unsigned long SEARCH_INTERVAL_MS = 90;
const unsigned long COMMAND_TIMEOUT_MS = 800;
const unsigned long WIFI_CONNECT_TIMEOUT_MS = 10000;
const unsigned long MQTT_RECONNECT_INTERVAL_MS = 5000;

const bool REVERSE_SERVO = true;

enum MovementCommand {
  CMD_IDLE,
  CMD_LEFT,
  CMD_RIGHT,
  CMD_CENTER,
  CMD_SEARCH
};

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
Servo panServo;
char mqttClientId[40];

MovementCommand currentCommand = CMD_IDLE;
MovementCommand lastPrintedCommand = CMD_IDLE;
int servoAngle = SERVO_CENTER_ANGLE;
int sweepDirection = 1;
unsigned long lastMoveAt = 0;
unsigned long lastReconnectAttempt = 0;
unsigned long lastCommandAt = 0;

void setServoAngle(int angle) {
  angle = constrain(angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
  if (angle == servoAngle) return;
  servoAngle = angle;
  panServo.write(servoAngle);
}

void applyTrackingStep(int logicalDirection) {
  int direction = REVERSE_SERVO ? -logicalDirection : logicalDirection;
  setServoAngle(servoAngle + (direction * TRACK_STEP));
}

bool messageHas(const String& message, const char* token) {
  return message.indexOf(token) >= 0;
}

MovementCommand parseCommand(String message) {
  message.trim();
  message.toUpperCase();

  if (message.startsWith("CMD_")) {
    message = message.substring(4);
  }

  if (message == "LEFT" || message == "MOVED_LEFT") return CMD_LEFT;
  if (message == "RIGHT" || message == "MOVED_RIGHT") return CMD_RIGHT;
  if (message == "CENTER" || message == "CENTERED") return CMD_CENTER;
  if (
      message == "SEARCH" || message == "SEARCHING" || message == "SCAN" ||
      message == "OUT_OF_FRAME"
  ) {
    return CMD_SEARCH;
  }
  if (message == "IDLE" || message == "STOP" || message == "STOPPED") return CMD_IDLE;
  return CMD_IDLE;
}

const char* commandName(MovementCommand command) {
  switch (command) {
    case CMD_LEFT: return "MOVED_LEFT";
    case CMD_RIGHT: return "MOVED_RIGHT";
    case CMD_CENTER: return "CENTERED";
    case CMD_SEARCH: return "SEARCHING";
    default: return "STOPPED";
  }
}

void applyIncomingCommand(MovementCommand newCommand) {
  if (newCommand != currentCommand) {
    lastMoveAt = 0;
  }
  currentCommand = newCommand;
  lastCommandAt = millis();
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  MovementCommand newCommand = parseCommand(message);
  applyIncomingCommand(newCommand);

  if (newCommand != lastPrintedCommand) {
    Serial.print("[MQTT] Command: ");
    Serial.println(commandName(newCommand));
    if (newCommand == CMD_SEARCH) {
      Serial.println("[STATE] SEARCHING: sweeping servo 0-180");
    } else if (newCommand == CMD_IDLE) {
      Serial.println("[STATE] STOPPED: holding position");
    }
    lastPrintedCommand = newCommand;
  }
}

void handleSerial() {
  if (!Serial.available()) return;

  String input = Serial.readStringUntil('\n');
  input.trim();
  MovementCommand newCmd = parseCommand(input);
  if (newCmd != CMD_IDLE || input.indexOf("IDLE") >= 0 || input.indexOf("STOP") >= 0) {
    applyIncomingCommand(newCmd);
    Serial.print("[SERIAL] Executing: ");
    Serial.println(input);
  }
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.print("[WiFi] Connecting");
  WiFi.mode(WIFI_STA);
  WiFi.persistent(false);
  WiFi.setAutoReconnect(true);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < WIFI_CONNECT_TIMEOUT_MS) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected");
    Serial.print("[WiFi] IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("[WiFi] RSSI: ");
    Serial.println(WiFi.RSSI());
  } else {
    Serial.println("\n[WiFi] Failed");
  }
}

bool connectMqtt() {
  if (mqttClient.connected()) return true;
  if (WiFi.status() != WL_CONNECTED) return false;

  if (millis() - lastReconnectAttempt < MQTT_RECONNECT_INTERVAL_MS) return false;
  lastReconnectAttempt = millis();

  uint32_t chipId = (uint32_t)(ESP.getEfuseMac() & 0xFFFFFF);
  snprintf(mqttClientId, sizeof(mqttClientId), "%s-%06X", MQTT_CLIENT_ID_PREFIX, chipId);

  Serial.print("[MQTT] Connecting as ");
  Serial.print(mqttClientId);
  Serial.print("...");
  if (!mqttClient.connect(mqttClientId)) {
    Serial.print(" Failed, rc=");
    Serial.println(mqttClient.state());
    return false;
  }

  Serial.println(" Connected");
  if (mqttClient.subscribe(MQTT_TOPIC)) {
    Serial.print("[MQTT] Subscribed: ");
    Serial.println(MQTT_TOPIC);
  } else {
    Serial.println("[MQTT] Subscribe failed");
  }
  return true;
}

void handleServo() {
  unsigned long now = millis();

  // Keep SEARCHING until PC sends STOPPED after re-acquiring the speaker.
  if (currentCommand != CMD_SEARCH && (now - lastCommandAt) > COMMAND_TIMEOUT_MS) {
    currentCommand = CMD_IDLE;
  }

  if (currentCommand == CMD_CENTER) {
    currentCommand = CMD_IDLE;
    return;
  }

  if (currentCommand == CMD_SEARCH) {
    if (now - lastMoveAt < SEARCH_INTERVAL_MS) return;
    lastMoveAt = now;

    setServoAngle(servoAngle + (sweepDirection * SEARCH_STEP));
    if (servoAngle >= SERVO_MAX_ANGLE) sweepDirection = -1;
    if (servoAngle <= SERVO_MIN_ANGLE) sweepDirection = 1;
    lastCommandAt = now;
    return;
  }

  if (now - lastMoveAt < TRACK_INTERVAL_MS) return;
  lastMoveAt = now;

  if (currentCommand == CMD_LEFT) {
    applyTrackingStep(-1);
    lastCommandAt = now;
  } else if (currentCommand == CMD_RIGHT) {
    applyTrackingStep(1);
    lastCommandAt = now;
  }
}

void setup() {
  Serial.begin(115200);
  delay(10);
  Serial.println("\n[SYS] Corene ESP32 face-servo initializing...");

  ESP32PWM::allocateTimer(0);
  panServo.setPeriodHertz(50);
  panServo.attach(SERVO_PIN, 500, 2400);
  setServoAngle(SERVO_CENTER_ANGLE);

  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setKeepAlive(30);
  mqttClient.setSocketTimeout(3);

  connectWiFi();
  lastCommandAt = millis();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (WiFi.status() == WL_CONNECTED && !mqttClient.connected()) connectMqtt();
  if (mqttClient.connected()) mqttClient.loop();

  handleSerial();
  handleServo();
  delay(1);
}

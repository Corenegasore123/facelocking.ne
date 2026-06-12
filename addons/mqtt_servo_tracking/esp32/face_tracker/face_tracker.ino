/*
 * BENAX / Corene — ESP32 pan servo (MQTT subscriber).
 * Use this sketch only (ESP32 Dev Module). Do not flash ESP8266 code.
 *
 * Board:  ESP32 Dev Module
 * Libs:   PubSubClient, ESP32Servo
 *
 * Wiring:
 *   Servo BROWN  -> GND (common with ESP32 GND)
 *   Servo RED    -> 5V external supply (not 3V3)
 *   Servo YELLOW -> D14 (GPIO14)
 *
 * Commands (from recognize_mqtt.py):
 *   MOVED_LEFT, MOVED_RIGHT, CENTERED, OUT_OF_FRAME, STOPPED
 *   Also: LEFT, RIGHT, CENTER, SEARCH, IDLE, SCAN
 */
#define USE_US_TIMER

#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>

const char* WIFI_SSID = "EdNet";
const char* WIFI_PASSWORD = "Huawei@123";// =========================
// MQTT Settings
// =========================
const char* MQTT_SERVER = "157.173.101.159";
const uint16_t MQTT_PORT = 1883;

const char* MQTT_TOPIC = "vision/Corene/movement";
const char* MQTT_CLIENT_ID = "corene-face-servo-esp32";

// =========================
// Servo Configuration
// =========================
const uint8_t SERVO_PIN = 14; // Board label D14 

const int SERVO_MIN_ANGLE = 0;
const int SERVO_MAX_ANGLE = 180;
const int SERVO_CENTER_ANGLE = 90;

const int TRACK_STEP = 2;
const int SEARCH_STEP = 2;

const unsigned long TRACK_INTERVAL_MS = 35;
const unsigned long SEARCH_INTERVAL_MS = 60;
const unsigned long COMMAND_TIMEOUT_MS = 1500;

const bool REVERSE_SERVO = true;

// =========================
// Command Types
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

int servoAngle = SERVO_CENTER_ANGLE;
int sweepDirection = 1;

unsigned long lastMoveAt = 0;
unsigned long lastReconnectAttempt = 0;
unsigned long lastCommandAt = 0;

// ======================================================
// Servo Functions
// ======================================================

void setServoAngle(int angle) {
  angle = constrain(angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);

  servoAngle = angle;
  panServo.write(servoAngle);
}

void applyTrackingStep(int logicalDirection) {

  int direction =
      REVERSE_SERVO ? -logicalDirection : logicalDirection;

  setServoAngle(
      servoAngle + (direction * TRACK_STEP)
  );
}

// ======================================================
// Command Parsing (BENAX + internal aliases)
// ======================================================

bool messageHas(const String& message, const char* token) {
  return message.indexOf(token) >= 0;
}

MovementCommand parseCommand(String message) {
  message.trim();
  String upper = message;
  upper.toUpperCase();

  if (upper.startsWith("CMD_")) {
    upper = upper.substring(4);
  }

  if (
      upper == "LEFT" ||
      messageHas(upper, "MOVE_LEFT") ||
      messageHas(upper, "MOVED_LEFT")
  ) {
    return CMD_LEFT;
  }

  if (
      upper == "RIGHT" ||
      messageHas(upper, "MOVE_RIGHT") ||
      messageHas(upper, "MOVED_RIGHT")
  ) {
    return CMD_RIGHT;
  }

  if (upper == "CENTER" || upper == "CENTERED") {
    return CMD_CENTER;
  }

  if (
      upper == "SEARCH" || upper == "SCAN" ||
      upper == "OUT_OF_FRAME" || messageHas(upper, "NO_FACE")
  ) {
    return CMD_SEARCH;
  }

  if (upper == "IDLE" || upper == "STOPPED" || upper == "STOP") {
    return CMD_IDLE;
  }

  return CMD_IDLE;
}

void applyIncomingCommand(MovementCommand newCommand) {
  if (newCommand != currentCommand) {
    lastMoveAt = 0;
  }
  currentCommand = newCommand;
  lastCommandAt = millis();
}

// ======================================================
// MQTT Callback
// ======================================================

void mqttCallback(
    char* topic,
    byte* payload,
    unsigned int length
) {

  String message = "";

  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  MovementCommand newCommand = parseCommand(message);
  applyIncomingCommand(newCommand);

  Serial.print("[MQTT] Received: ");
  Serial.println(message);
}

// ======================================================
// Serial Input
// ======================================================

void handleSerial() {

  if (Serial.available() > 0) {

    String input =
        Serial.readStringUntil('\n');

    input.trim();

    MovementCommand newCmd = parseCommand(input);
    applyIncomingCommand(newCmd);

    Serial.print("[SERIAL] Executing: ");
    Serial.println(input);
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

  // Disconnect old attempts first
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

  if (millis() - lastReconnectAttempt < 5000) {
    return false;
  }

  lastReconnectAttempt = millis();

  Serial.print("[MQTT] Connecting...");

  bool connected =
      mqttClient.connect(MQTT_CLIENT_ID);

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
// Servo Logic
// ======================================================

void handleServo() {

  unsigned long now = millis();

  // Only STOPPED/IDLE times out. LEFT, RIGHT, SEARCH run until a new MQTT command arrives.
  if (
      currentCommand == CMD_IDLE &&
      (now - lastCommandAt) > COMMAND_TIMEOUT_MS
  ) {
    currentCommand = CMD_IDLE;
  }

  // ----------------------
  // CENTER
  // ----------------------
  if (currentCommand == CMD_CENTER) {

    setServoAngle(SERVO_CENTER_ANGLE);

    currentCommand = CMD_IDLE;

    return;
  }

  // ----------------------
  // SEARCH MODE
  // ----------------------
  if (currentCommand == CMD_SEARCH) {

    if (
        now - lastMoveAt <
        SEARCH_INTERVAL_MS
    ) {
      return;
    }

    lastMoveAt = now;

    setServoAngle(
        servoAngle +
        (sweepDirection * SEARCH_STEP)
    );

    if (servoAngle >= SERVO_MAX_ANGLE) {
      sweepDirection = -1;
    }

    if (servoAngle <= SERVO_MIN_ANGLE) {
      sweepDirection = 1;
    }

    lastCommandAt = now;
    return;
  }

  // ----------------------
  // LEFT / RIGHT TRACKING
  // ----------------------
  if (
      now - lastMoveAt <
      TRACK_INTERVAL_MS
  ) {
    return;
  }

  lastMoveAt = now;

  if (currentCommand == CMD_LEFT) {
    applyTrackingStep(-1);
    lastCommandAt = now;
  } else if (currentCommand == CMD_RIGHT) {
    applyTrackingStep(1);
    lastCommandAt = now;
  }
}

// ======================================================
// Setup
// ======================================================

void setup() {

  Serial.begin(115200);

  delay(500);

  Serial.println();
  Serial.println("[SYS] ESP32 Face Servo Initializing...");
  Serial.println("[SYS] Production firmware (replaces any servo_test upload).");

  // ----------------------
  // ESP32 Servo Setup
  // ----------------------
  ESP32PWM::allocateTimer(0);

  panServo.setPeriodHertz(50);

  panServo.attach(
      SERVO_PIN,
      500,
      2400
  );

  setServoAngle(SERVO_CENTER_ANGLE);

  // ----------------------
  // MQTT Setup
  // ----------------------
  mqttClient.setServer(
      MQTT_SERVER,
      MQTT_PORT
  );

  mqttClient.setCallback(mqttCallback);

  // ----------------------
  // WiFi
  // ----------------------
  connectWiFi();

  lastCommandAt = millis();
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

  // Serial commands
  handleSerial();

  // Servo movement
  handleServo();

  delay(1);
}
# Face Locking Recognition System

A real-time face recognition project for enrolling people, recognizing them from a camera, locking onto a selected face, logging evidence, and steering an ESP32 servo camera over MQTT for the Corene vision system.

The project is CPU-first. It works with the normal `onnxruntime` package, so people without GPUs do not need CUDA, cuDNN, or `onnxruntime-gpu`.

## Project Layout

- `src/enroll.py` - capture face samples and build the face database.
- `src/recognize.py` - run local face recognition and face locking.
- `src/rebuild_db.py` - rebuild `data/db/face_db.npz` from existing enrollment crops.
- `addons/mqtt_servo_tracking/recognize_mqtt.py` - face locking plus MQTT movement/status publishing.
- `addons/mqtt_servo_tracking/esp32/face_tracker/face_tracker.ino` - ESP32 servo firmware.
- `addons/mqtt_servo_tracking/FLOWCHART.md` - Mermaid flowcharts (Recognize → Track → Command).
- `dashboard/index.html` - Corene MQTT dashboard for live movement and lock status.
- `logs/` - session CSV evidence logs and face-lock action history files.

## Requirements

Use Python 3.10, 3.11, 3.12, or 3.13. Python 3.14 is not recommended because some computer-vision packages may not have wheels for it yet.

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

The included requirements use:

```text
onnxruntime
```

That is the CPU ONNX Runtime package. If only CPU ONNX Runtime is installed, the recognizer automatically uses `CPUExecutionProvider`.

Required model files:

- `models/embedder_arcface.onnx`
- `models/face_landmarker.task`

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Put the required model files in `models/`.

3. Enroll a person:

   ```bash
   python -m src.enroll
   ```

   Controls:

   - `SPACE` - capture one sample.
   - `a` - toggle auto-capture.
   - `s` - save enrollment to the database.
   - `q` - quit.

4. Rebuild the database if you already have crops in `data/enroll/`:

   ```bash
   python -m src.rebuild_db
   ```

5. Run local recognition:

   ```bash
   python -m src.recognize
   ```

   Controls:

   - `+` or `=` - increase the recognition distance threshold.
   - `-` - decrease the threshold.
   - `r` - reload the face database.
   - `d` - toggle debug overlay.
   - `l` - lock or unlock the selected recognized face.
   - `q` - quit.

## CPU And GPU Notes

For CPU-only users, keep `onnxruntime` in `requirements.txt`. No separate CPU-only Python files are needed.

When the app starts, it checks ONNX Runtime providers. If only CPU is available, it selects CPU automatically. If GPU-capable ONNX Runtime packages are installed, it prompts for a provider and keeps CPU as a fallback.

Optional GPU setups:

- NVIDIA CUDA: use `onnxruntime-gpu` in a GPU-specific environment.
- Windows DirectML: use `onnxruntime-directml` in a GPU-specific environment.

Avoid installing multiple ONNX Runtime variants into the same environment unless you know they are compatible.

## Face Locking

During recognition, press `l` when a known face is selected. The app locks onto that identity, tracks movement, and records actions such as:

- Face locked or unlocked.
- Head moved left or right.
- Smile or blink events detected from landmarks.
- Face temporarily lost or reacquired.

Session CSV evidence logs are written to `logs/` while `recognize_mqtt.py` runs:

```text
session_YYYYMMDD_HHMMSS.csv
```

Columns: `timestamp`, `iso_time`, `speaker_id`, `confidence`, `movement_command`, `error_x`, `faces_detected`, `locked`, `locked_face_found`.

Face-lock action history files are also written on unlock:

```text
[Name]_history_[timestamp].txt
```

## MQTT Servo Addon

The MQTT addon keeps the original recognizer separate and publishes servo commands for the ESP32.

Run it from the repo root:

```bash
python addons/mqtt_servo_tracking/recognize_mqtt.py
```

Default MQTT settings (Corene):

- Broker: `broker.hivemq.com`
- MQTT port: `1883`
- Browser WebSocket URL: `ws://broker.hivemq.com:8000/mqtt`
- Wi-Fi SSID (ESP32): `EdNet` (set in firmware)
- Movement topic: `vision/Corene/movement`
- Status topic: `vision/Corene/status`

Movement payloads on `vision/Corene/movement`:

- `LEFT` / `MOVED_LEFT` — speaker left of center.
- `RIGHT` / `MOVED_RIGHT` — speaker right of center.
- `CENTER` / `CENTERED` — speaker centered.
- `SEARCH` / `OUT_OF_FRAME` — speaker lost; pan sweep.
- `IDLE` / `STOPPED` — hold / no active lock.

Evidence logs also include BENAX `assessment_command` aliases (`MOVED_LEFT`, `CENTERED`, etc.).

**Unlock options:**

- Press **`l`** or **`u`** to manually unlock the speaker.
- Auto-unlock after **`--unlock-timeout-sec`** (default 40s) if the speaker is not seen.
- Auto-unlock after **`--search-unlock-sec`** (default 30s) if still searching without reacquire.
- Disable search-unlock with `--search-unlock-sec 0`.

Dashboard JSON is published on `vision/Corene/status`, including movement, lock state, target name, confidence, face count, horizontal error, FPS, threshold, and provider.

Useful addon flags:

```bash
python addons/mqtt_servo_tracking/recognize_mqtt.py --mqtt-broker broker.hivemq.com --mqtt-topic vision/Corene/movement --mqtt-status-topic vision/Corene/status --camera-width 1280 --camera-height 720 --max-faces 3 --detect-every 2 --recognize-every 3 --deadzone-px 80 --center-exit-hysteresis-px 30 --command-hold-sec 0.25 --search-delay-sec 0.8 --reacquire-hold-sec 0.30 --command-confirm-frames 2 --mqtt-min-interval 0.15 --mqtt-status-min-interval 0.25 --log-interval 0.25
```

## Dashboard

Open:

```text
dashboard/index.html
```

The Corene dashboard uses MQTT over WebSockets and defaults to:

```text
ws://broker.hivemq.com:8000/mqtt
```

It listens to:

- `vision/Corene/movement`
- `vision/Corene/status`

The JSON status topic is authoritative for the displayed command. Plain MQTT port `1883` is for Python and the ESP32, not browsers.

## ESP32 Servo Setup

1. Open `addons/mqtt_servo_tracking/esp32/face_tracker/face_tracker.ino`.
2. Set `WIFI_SSID` and `WIFI_PASSWORD` to your network.
3. Confirm MQTT settings:

   ```cpp
   MQTT_SERVER = "broker.hivemq.com";
   MQTT_TOPIC = "vision/Corene/movement";
   ```

4. Wire the servo:

   - Signal → D14 (GPIO14)
   - Power → external 5 V supply (not ESP32 3.3 V)
   - GND → common ground with ESP32

5. Install Arduino libraries:

   - `PubSubClient`
   - `ESP32Servo`

6. Upload (re-flash after any firmware change):

   ```powershell
   powershell -ExecutionPolicy Bypass -File addons/mqtt_servo_tracking/esp32/upload.ps1 -Port COM5
   ```

   Sketch path: `addons/mqtt_servo_tracking/esp32/face_tracker/face_tracker.ino`

For a different ESP32 board:

```powershell
powershell -ExecutionPolicy Bypass -File addons/mqtt_servo_tracking/esp32/upload.ps1 -Port COM5 -Fqbn esp32:esp32:esp32dev
```

## Tuning Tips

- CPU-first defaults use `640x480`, `--max-faces 3`, `--detect-every 2`, and `--recognize-every 3`.
- Run with `--profile` to show frame, detection, and recognition timing in the recognition window.
- Increase `--detect-every` or `--recognize-every` if CPU usage is still too high.
- Lower `--max-faces` or use `--locked-max-faces 1` for smoother locked-face tracking.
- Increase `--deadzone-px` if the servo moves while the face is already centered.
- Increase `--command-hold-sec` if the dashboard or servo still reacts to short command blips.
- Increase `--search-delay-sec` if brief recognition drops trigger `SEARCH` too quickly.
- Increase `--command-confirm-frames` if `LEFT` and `RIGHT` flicker.
- Lower the recognition threshold if false positives happen.
- Raise the recognition threshold if known faces are not accepted.

## Troubleshooting

- Empty database: run `python -m src.enroll` or `python -m src.rebuild_db`.
- Camera not available: check the camera index in the recognizer code if your webcam is not device `1`.
- Dashboard offline: confirm WebSocket URL `ws://broker.hivemq.com:8000/mqtt` (include `/mqtt` path).
- ESP32 not moving: re-flash firmware after broker changes; confirm Wi-Fi, broker `broker.hivemq.com`, topic `vision/Corene/movement`, servo power, and Serial Monitor output.
- CPU-only machine: keep `onnxruntime`; do not install `onnxruntime-gpu`.

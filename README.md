# Face Locking Recognition System (BENAX / Corene)

A real-time **single-speaker** face recognition and camera tracking system: enroll one speaker, lock onto that identity, ignore other faces, publish BENAX motor commands over MQTT, and drive an ESP32 pan servo.

CPU-first: works with `onnxruntime` (no GPU required).

## Project Layout

| Path | Purpose |
|------|---------|
| `src/enroll.py` | Capture 10–30 samples and build `data/db/face_db.npz` |
| `src/recognize.py` | Local recognition without MQTT |
| `addons/mqtt_servo_tracking/recognize_mqtt.py` | Speaker lock + MQTT + logging |
| `addons/mqtt_servo_tracking/esp32/face_tracker/face_tracker.ino` | ESP32 servo firmware |
| `diagrams/pipeline_flowchart.mmd` | Recognize → Track → Command flowchart (Mermaid) |
| `dashboard/index.html` | Live MQTT dashboard |
| `docs/MQTT_BROKER.md` | Broker, topics, Mosquitto notes |
| `docs/POWER_AND_SAFETY.md` | Wiring, power, unlock safety |
| `logs/session_*.csv` | BENAX session CSV logs |
| `logs/evidence/*.jsonl` | Detailed JSONL evidence |

## Requirements

- Python 3.10–3.13
- `pip install -r requirements.txt`
- Models: `models/embedder_arcface.onnx`, `models/face_landmarker.task`

## Quick Start

```bash
pip install -r requirements.txt
python -m src.enroll
python addons/mqtt_servo_tracking/recognize_mqtt.py
```

Enrollment controls: `SPACE` capture, `a` auto-capture, `s` save, `q` quit.

Tracker controls: `l` / `u` lock/unlock, `q` quit, `d` debug, `+`/`-` threshold.

## MQTT (default)

| Setting | Value |
|---------|--------|
| Broker | `157.173.101.159` |
| Port | `1883` |
| Dashboard WebSocket | `ws://157.173.101.159:9001` |
| Movement topic | `vision/Corene/servo_control` |
| Status topic | `vision/Corene/status` |

### BENAX motor commands (published on movement topic)

- `MOVED_LEFT` — speaker left of center  
- `MOVED_RIGHT` — speaker right of center  
- `CENTERED` — speaker centered  
- `SEARCHING` — speaker lost, pan search  
- `STOPPED` — hold / unlocked  

ESP32 firmware accepts these plus internal aliases (`LEFT`, `SEARCH`, etc.).

## Logging (BENAX evidence)

**CSV** (`logs/session_YYYYMMDD_HHMMSS.csv`):

`timestamp`, `iso_time`, `speaker_id`, `confidence`, `movement_command`, `assessment_command`, `error_x`, `faces_detected`, `locked`, `locked_face_found`

**JSONL** (`logs/evidence/face_tracking_evidence_*.jsonl`) — full per-frame detail.

**Action history** (`logs/<Name>_history_*.txt`) — lock, lost, reacquired, unlock events.

Disable CSV: `--disable-csv-log`. Interval: `--log-interval 0.25`.

## Speaker lock behaviour

1. Auto-lock onto enrolled speaker when seen.  
2. Track with `MOVED_LEFT` / `MOVED_RIGHT` / `CENTERED`.  
3. On loss → `SEARCHING` sweep immediately.  
4. On reacquire → stop search and track again.  
5. Unlock: manual (`l`/`u`), `--unlock-timeout-sec 40`, or `--search-unlock-sec 30`.

## Flowchart

Open `diagrams/pipeline_flowchart.mmd` in [mermaid.live](https://mermaid.live) or Cursor Markdown preview.

## ESP32 setup

1. Open `addons/mqtt_servo_tracking/esp32/face_tracker/face_tracker.ino`  
2. Set `WIFI_SSID` / `WIFI_PASSWORD`  
3. Confirm `MQTT_SERVER = "157.173.101.159"`  
4. Wire servo signal → **D8 (GPIO15)**, power → **5 V**, GND → common ground (WiFi defaults: `RCA-OUTDOR`)  
5. Upload (re-flash after any firmware change):

```powershell
powershell -ExecutionPolicy Bypass -File addons/mqtt_servo_tracking/esp32/upload.ps1 -Port COM5
```

See `docs/POWER_AND_SAFETY.md` for supply and safety notes.

## Dashboard

Open `dashboard/index.html`. Default WebSocket: `ws://157.173.101.159:9001`.

## Assessment alignment

| Requirement | Status |
|-------------|--------|
| Single-speaker enrollment | `src/enroll.py` |
| Speaker lock, ignore others | `recognize_mqtt.py` |
| Recognize → Track → Command flowchart | `diagrams/pipeline_flowchart.mmd` |
| BENAX MQTT commands | Published on movement topic |
| Re-acquisition / search | `SEARCHING` + unlock rules |
| CSV + JSON evidence logs | `logs/session_*.csv` + `logs/evidence/` |
| ESP embedded MQTT subscriber | ESP32 firmware (ESP8266-compatible protocol) |
| Power / safety documentation | `docs/POWER_AND_SAFETY.md` |

## Troubleshooting

- Empty DB → `python -m src.enroll`  
- Dashboard offline → WebSocket `ws://157.173.101.159:9001`  
- Servo not moving → re-flash ESP32, check 5 V power, broker, topic, Serial Monitor @ 115200  

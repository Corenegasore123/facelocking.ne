# Face Locking — Corene (BENAX)

Single-speaker enrollment, face lock, MQTT servo tracking, evidence logging.

## Layout

```
FaceLocking/
├── config/corene.py              # Broker, topics, tuning
├── src/
│   ├── recognize_mqtt.py         # Main tracker + MQTT (Corene topics)
│   ├── face_locking.py           # Alternate tracker (same project layout)
│   ├── enroll.py                 # Enrollment
│   ├── rebuild_db.py             # Rebuild DB from crops
│   ├── lock_state.py / unlock.py # Persisted speaker lock
│   ├── validate_system.py        # Pre-flight checks
│   ├── servo_search.py           # MQTT search test
│   ├── list_cameras.py           # Camera probe
│   ├── vision/                   # haar_5pt, embed, onnx_providers
│   └── dashboard/websocket_backend.py
├── firmware/esp32/face_tracker/  # ESP32 sketch + upload.ps1
├── dashboard/index.html
├── data/db | enroll | logs
├── diagrams/pipeline_flowchart.mmd
├── models/
└── scripts/init_project.py
```

## Setup

```bash
pip install -r requirements.txt
python scripts/init_project.py
```

Models in `models/`: `face_landmarker.task`, `embedder_arcface.onnx`

## Run

```bash
python -m src.validate_system
python -m src.enroll
python -m src.recognize_mqtt
```

Recognition without MQTT: `python -m src.recognize_mqtt --disable-mqtt`

## ESP32

Servo signal **GPIO14**, 5 V, common GND. Upload:

```powershell
powershell -ExecutionPolicy Bypass -File firmware/esp32/upload.ps1 -Port COM5
```

## MQTT (`config/corene.py`)

Broker `157.173.101.159:1883` · topic `vision/corene/movement` (ESP32 subscribes here)  
Commands: `LEFT`, `RIGHT`, `CENTER`, `SEARCH`, `IDLE`  
Dashboard status: `vision/corene/status` · heartbeat: `vision/corene/heartbeat`

## Keys

**Enroll:** SPACE capture · `a` auto · `s` save · `q` quit (auto-saves if enough samples)  
**Tracker:** `l` lock/unlock · `d` debug · `+`/`-` threshold · `q` quit

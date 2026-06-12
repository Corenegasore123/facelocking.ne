# Face Locking — Corene (BENAX)

Single-speaker enrollment, face lock, MQTT servo tracking, evidence logging.

## Layout

```
FaceLocking/
├── config/corene.py              # Broker, topics, tuning
├── src/
│   ├── face_locking.py           # Main app (tracker + MQTT)
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
python -m src.face_locking
```

Recognition without MQTT: `python -m src.face_locking --disable-mqtt`

## ESP32

Servo signal **GPIO14**, 5 V, common GND. Upload:

```powershell
powershell -ExecutionPolicy Bypass -File firmware/esp32/upload.ps1 -Port COM5
```

## MQTT (`config/corene.py`)

Broker `157.173.101.159:1883` · topic `vision/Corene/servo_control`  
Commands: `MOVED_LEFT`, `MOVED_RIGHT`, `CENTERED`, `SEARCHING`, `STOPPED`

## Keys

**Enroll:** SPACE capture · `a` auto · `s` save · `q` quit (auto-saves if enough samples)  
**Tracker:** `l` lock/unlock · `d` debug · `+`/`-` threshold · `q` quit

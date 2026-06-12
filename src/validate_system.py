"""Pre-flight checks for the Corene FaceLocking system."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from config.corene import cfg

ESP32_SKETCH = Path(__file__).resolve().parents[1] / "firmware" / "esp32" / "face_tracker" / "face_tracker.ino"


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" - {detail}"
    print(line)
    return ok


def main() -> int:
    all_ok = True
    print("=" * 60)
    print("Corene FaceLocking — system validation")
    print("=" * 60)

    for model_name in ("face_landmarker.task", "embedder_arcface.onnx"):
        path = cfg.models_dir / model_name
        all_ok &= check(f"Model: {model_name}", path.exists(), str(path))

    all_ok &= check("Face database", cfg.db_path.exists(), str(cfg.db_path))
    if cfg.db_path.exists():
        try:
            import numpy as np

            data = np.load(cfg.db_path, allow_pickle=True)
            identities = list(data.files)
            all_ok &= check("Enrolled identities", bool(identities), ", ".join(identities) or "none")
        except Exception as exc:
            all_ok &= check("Face database readable", False, str(exc))

    if cfg.enroll_dir.exists():
        crop_dirs = [p for p in cfg.enroll_dir.iterdir() if p.is_dir()]
        if crop_dirs:
            counts = ", ".join(f"{p.name}={len(list(p.glob('*.jpg')))}" for p in crop_dirs)
            all_ok &= check("Enrollment crops", True, counts)

    try:
        import cv2

        cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
        opened = cap.isOpened()
        if opened:
            cap.release()
        all_ok &= check(f"Camera index {cfg.camera_index}", opened)
    except Exception as exc:
        all_ok &= check(f"Camera index {cfg.camera_index}", False, str(exc))

    try:
        import paho.mqtt.client as mqtt

        client = mqtt.Client(client_id=f"validate-{int(time.time())}")
        client.connect(cfg.mqtt_broker, cfg.mqtt_port, 60)
        info = client.publish(cfg.movement_topic, payload=cfg.cmd_stop, qos=0)
        info.wait_for_publish(timeout=3)
        client.disconnect()
        all_ok &= check(
            "MQTT broker publish",
            info.rc == mqtt.MQTT_ERR_SUCCESS,
            f"{cfg.mqtt_broker}:{cfg.mqtt_port} topic={cfg.movement_topic}",
        )
    except Exception as exc:
        all_ok &= check("MQTT broker publish", False, str(exc))

    all_ok &= check("ESP32 firmware sketch", ESP32_SKETCH.exists(), str(ESP32_SKETCH))
    all_ok &= check("Main tracker", (Path(__file__).parent / "face_locking.py").exists())
    dashboard_html = cfg.models_dir.parent / "dashboard" / "index.html"
    all_ok &= check("Dashboard UI", dashboard_html.exists(), str(dashboard_html))

    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    all_ok &= check("Evidence log directory", cfg.logs_dir.exists(), str(cfg.logs_dir))

    print("=" * 60)
    if all_ok:
        print("All checks passed. Run: python -m src.face_locking")
        return 0
    print("Some checks failed. Fix the items above before the demo.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Corene team defaults — broker, MQTT topics, tracking tunables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CoreneConfig:
    team_id: str = "Corene"
    default_speaker: str = "Corene"

    mqtt_broker: str = "157.173.101.159"
    mqtt_port: int = 1883
    # Match firmware/esp32/face_tracker_servo/face_tracker_servo.ino
    movement_topic: str = "vision/corene/movement"
    status_topic: str = "vision/corene/status"
    heartbeat_topic: str = "vision/corene/heartbeat"
    dashboard_ws_url: str = "ws://127.0.0.1:5501"
    dashboard_http_port: int = 5500
    websocket_relay_port: int = 5501

    db_path: Path = REPO_ROOT / "data" / "db" / "face_db.npz"
    db_meta_path: Path = REPO_ROOT / "data" / "db" / "face_db.json"
    enroll_dir: Path = REPO_ROOT / "data" / "enroll"
    logs_dir: Path = REPO_ROOT / "data" / "logs"
    lock_file: Path = REPO_ROOT / "data" / "locked_identity.json"
    models_dir: Path = REPO_ROOT / "models"

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720

    # Plain text on movement topic — must match Arduino parseCommand()
    cmd_left: str = "LEFT"
    cmd_right: str = "RIGHT"
    cmd_center: str = "CENTER"
    cmd_search: str = "SEARCH"
    cmd_out_of_frame: str = "SEARCH"
    cmd_stop: str = "IDLE"
    heartbeat_interval_sec: float = 30.0

    # Tracking (aligned with stable Pascaline-style behaviour)
    match_threshold: float = 0.40
    deadzone_px: float = 65.0
    center_exit_hysteresis_px: float = 45.0
    side_switch_hysteresis_px: float = 35.0
    error_smooth_alpha: float = 0.22
    kps_smooth_alpha: float = 0.4
    command_confirm_frames: int = 4
    track_publish_interval_sec: float = 0.2
    track_hold_frames: int = 10
    search_missing_frames: int = 12
    reacquire_frames: int = 5
    search_delay_sec: float = 0.3
    search_cooldown_sec: float = 2.0
    out_of_frame_miss_frames: int = 30
    unlock_timeout_sec: float = 40.0

    # ESP32
    servo_pin: int = 14


cfg = CoreneConfig()

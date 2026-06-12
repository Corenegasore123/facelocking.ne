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
    movement_topic: str = "vision/Corene/servo_control"
    status_topic: str = "vision/Corene/status"
    heartbeat_topic: str = "vision/Corene/heartbeat"
    dashboard_ws_url: str = "ws://157.173.101.159:9001"
    websocket_relay_port: int = 9002

    db_path: Path = REPO_ROOT / "data" / "db" / "face_db.npz"
    db_meta_path: Path = REPO_ROOT / "data" / "db" / "face_db.json"
    enroll_dir: Path = REPO_ROOT / "data" / "enroll"
    logs_dir: Path = REPO_ROOT / "data" / "logs"
    lock_file: Path = REPO_ROOT / "data" / "locked_identity.json"
    models_dir: Path = REPO_ROOT / "models"

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720

    # BENAX motor command strings (plain text on movement topic)
    cmd_left: str = "MOVED_LEFT"
    cmd_right: str = "MOVED_RIGHT"
    cmd_center: str = "CENTERED"
    cmd_search: str = "SEARCHING"
    cmd_out_of_frame: str = "OUT_OF_FRAME"
    cmd_stop: str = "STOPPED"
    heartbeat_interval_sec: float = 30.0

    # Tracking (aligned with stable Pascaline-style behaviour)
    match_threshold: float = 0.40
    deadzone_px: float = 45.0
    center_exit_hysteresis_px: float = 30.0
    error_smooth_alpha: float = 0.35
    command_confirm_frames: int = 2
    track_publish_interval_sec: float = 0.15
    track_lost_tolerance_frames: int = 2
    search_missing_frames: int = 12
    reacquire_frames: int = 5
    search_delay_sec: float = 0.3
    search_cooldown_sec: float = 2.0
    out_of_frame_miss_frames: int = 30
    unlock_timeout_sec: float = 40.0

    # ESP32
    servo_pin: int = 14


cfg = CoreneConfig()

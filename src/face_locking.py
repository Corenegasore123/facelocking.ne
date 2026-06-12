"""
Corene face lock + MQTT servo tracking (BENAX assessment).

Pipeline: Haar -> FaceMesh 5pt -> ArcFace -> single-speaker lock -> MQTT commands.

Run:  python -m src.face_locking

Keys: q quit | r reload DB | l lock/unlock | d debug | +/- threshold
"""
from __future__ import annotations
import argparse
import csv
import json
import time
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort
try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

# Reuse your known-good alignment method
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.corene import cfg
from src.vision.haar_5pt import align_face_5pt
from src.lock_state import load_lock, save_lock
from src.vision.onnx_providers import select_provider_interactive, get_provider_display_name

# -------------------------
# Data
# -------------------------
@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray # (5,2) float32 in FULL-frame coords

class ActionType(Enum):
    FACE_LOCKED = auto()
    FACE_LOST = auto()

@dataclass
class Action:
    type: ActionType
    timestamp: float
    details: str = ""

@dataclass
class FaceLock:
    target_name: str
    target_emb: np.ndarray
    last_seen: float = field(default_factory=time.time)
    history: List[Action] = field(default_factory=list)
    consecutive_frames: int = 0
    
    def update_position(self, kps: np.ndarray) -> List[Action]:
        self.last_seen = time.time()
        self.consecutive_frames += 1
        return []

@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool

# -------------------------
# Math helpers
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)

def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    """
    Build a nicer face-like bbox from 5 points with asymmetric padding.
    kps: (5,2) in full-frame coords
    """
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    """
    Minimal geometry sanity:
    - eyes not collapsed
    - mouth generally below nose
    """
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < float(min_eye_dist):
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True

# -------------------------
# DB helpers
# -------------------------
def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    try:
        data = np.load(str(db_path), allow_pickle=True)
        out: Dict[str, np.ndarray] = {}
        for k in data.files:
            out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
        return out
    except Exception as e:
        print(f"Warning: Failed to load database {db_path}: {e}. Starting with empty DB.")
        return {}

# -------------------------
# Embedder
# -------------------------
class ArcFaceEmbedderONNX:
    """
    ArcFace-style ONNX embedder.
    Input: 112x112 BGR -> internally RGB + (x-127.5)/128, NHWC float32.
    Output: (1,D) or (D,)
    """
    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
        providers: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        self.debug = bool(debug)
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        model_input = self.sess.get_inputs()[0]
        self.in_name = model_input.name
        self.out_name = self.sess.get_outputs()[0].name
        self.input_layout = self._detect_input_layout(model_input.shape)
        if self.debug:
            print("[embed] model:", model_path)
            print("[embed] providers:", self.sess.get_providers())
            print("[embed] input:", model_input.name, model_input.shape, model_input.type, self.input_layout)
            print("[embed] output:", self.sess.get_outputs()[0].name, self.sess.get_outputs()[0].shape, self.sess.get_outputs()[0].type)

    @staticmethod
    def _detect_input_layout(shape) -> str:
        if len(shape) == 4:
            if shape[1] == 3:
                return "NCHW"
            if shape[3] == 3:
                return "NHWC"
        return "NHWC"

    def _preprocess(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        img = aligned_bgr_112
        if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        if self.input_layout == "NCHW":
            x = np.transpose(rgb, (2, 0, 1))[None, ...]
        else:
            x = rgb[None, ...]
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = v.astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32)

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_bgr_112)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        emb = np.asarray(y, dtype=np.float32).reshape(-1)
        return self._l2_normalize(emb)

# -------------------------
# Multi-face Haar + FaceMesh(ROI) 5pt
# -------------------------
class HaarFaceMesh5pt:
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        model_path: str = "models/face_landmarker.task",
        min_size: Tuple[int, int] = (70, 70),
        debug: bool = False,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")
        
        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe"
            )
        
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model not found: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, # Use IMAGE mode for ROI processing
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # 5pt indices
        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32) # (x,y,w,h)

    def _roi_facemesh_5pt(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = roi_bgr.shape[:2]
        if H < 20 or W < 20:
            return None
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_image)
        
        if not res.face_landmarks:
            return None
        
        lm = res.face_landmarks[0]
        idxs = [self.IDX_LEFT_EYE, self.IDX_RIGHT_EYE, self.IDX_NOSE_TIP, self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT]
        pts = []
        for i in idxs:
            p = lm[i]
            pts.append([p.x * W, p.y * H])
        kps = np.array(pts, dtype=np.float32)
        # enforce left/right ordering
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]
        return kps

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_faces(gray)
        if faces.shape[0] == 0:
            return []
        
        # sort by area desc, keep top max_faces
        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(areas)[::-1]
        faces = faces[order][:max_faces]
        
        out: List[FaceDet] = []
        for (x, y, w, h) in faces:
            # expand ROI a bit for FaceMesh stability
            mx, my = 0.25 * w, 0.35 * h
            rx1, ry1, rx2, ry2 = _clip_xyxy(x - mx, y - my, x + w + mx, y + h + my, W, H)
            roi = frame_bgr[ry1:ry2, rx1:rx2]
            kps_roi = self._roi_facemesh_5pt(roi)
            if kps_roi is None:
                if self.debug:
                    print("[recognize] FaceMesh none for ROI -> skip")
                continue
            
            # map ROI kps back to full-frame coords
            kps = kps_roi.copy()
            kps[:, 0] += float(rx1)
            kps[:, 1] += float(ry1)
            
            # sanity: eye distance relative to Haar width
            if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * float(w))):
                if self.debug:
                    print("[recognize] 5pt geometry failed -> skip")
                continue
            
            # build bbox from kps (centered)
            bb = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
            x1, y1, x2, y2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)
            
            out.append(
                FaceDet(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    score=1.0,
                    kps=kps.astype(np.float32),
                )
            )
        return out

    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()

# -------------------------
# Matcher
# -------------------------
class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.40):
        """
        Face database matcher with cosine similarity.
        
        Args:
            db: Dictionary mapping names to embedding vectors
            dist_thresh: Cosine distance threshold (default 0.40 for better recall).
                        Lower = stricter (fewer false positives, more false negatives).
                        Higher = more lenient (more false positives, fewer false negatives).
        """
        self.db = db
        self.dist_thresh = float(dist_thresh)
        # pre-stack for speed
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self):
        self._names = sorted(self.db.keys())
        if self._names:
            self._mat = np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
        else:
            self._mat = None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None or len(self._names) == 0:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
        e = emb.reshape(1, -1).astype(np.float32) # (1,D)
        # cosine similarity since both sides are normalized: sim = dot
        sims = (self._mat @ e.T).reshape(-1) # (K,)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_dist = 1.0 - best_sim
        ok = best_dist <= self.dist_thresh
        return MatchResult(
            name=self._names[best_i] if ok else None,
            distance=float(best_dist),
            similarity=float(best_sim),
            accepted=bool(ok),
        )

# -------------------------
# UI Helpers
# -------------------------
def draw_text_with_shadow(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    shadow_offset: int = 1,
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    font: int = cv2.FONT_HERSHEY_DUPLEX,
) -> None:
    """
    Draw text with a shadow/outline for better readability.
    Uses FONT_HERSHEY_DUPLEX for a cleaner, more modern look.
    """
    x, y = pos
    # Draw shadow (lighter shadow for less bold appearance)
    for dx, dy in [(shadow_offset, shadow_offset), (-shadow_offset, shadow_offset), 
                   (shadow_offset, -shadow_offset), (-shadow_offset, -shadow_offset)]:
        cv2.putText(img, text, (x + dx, y + dy), font, font_scale, shadow_color, thickness, cv2.LINE_AA)
    # Draw main text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_text_box(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.7,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.7,
    padding: int = 8,
    font: int = cv2.FONT_HERSHEY_DUPLEX,
) -> Tuple[int, int]:
    """
    Draw text with a semi-transparent background box for better readability.
    Returns (width, height) of the text box.
    """
    x, y = pos
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
    
    # Create overlay for transparency
    overlay = img.copy()
    box_x1 = x - padding
    box_y1 = y - text_height - padding
    box_x2 = x + text_width + padding
    box_y2 = y + baseline + padding
    
    # Ensure coordinates are within image bounds
    h, w = img.shape[:2]
    box_x1 = max(0, box_x1)
    box_y1 = max(0, box_y1)
    box_x2 = min(w, box_x2)
    box_y2 = min(h, box_y2)
    
    if box_x2 > box_x1 and box_y2 > box_y1:
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw text
    draw_text_with_shadow(img, text, (x, y), font_scale, text_color, 1, font=font)
    
    return (text_width + padding * 2, text_height + padding * 2 + baseline)

# -------------------------
# MQTT movement control
# -------------------------
MOVEMENT_LEFT = cfg.cmd_left
MOVEMENT_RIGHT = cfg.cmd_right
MOVEMENT_CENTER = cfg.cmd_center
MOVEMENT_SEARCH = cfg.cmd_search
MOVEMENT_OUT_OF_FRAME = cfg.cmd_out_of_frame
MOVEMENT_IDLE = cfg.cmd_stop
SEARCH_MOVEMENT_COMMANDS = frozenset({MOVEMENT_SEARCH, MOVEMENT_OUT_OF_FRAME})
TRACK_MOVEMENT_COMMANDS = frozenset({
    MOVEMENT_LEFT,
    MOVEMENT_RIGHT,
    MOVEMENT_CENTER,
})
DEFAULT_MQTT_BROKER = cfg.mqtt_broker
DEFAULT_MOVEMENT_TOPIC = cfg.movement_topic
DEFAULT_STATUS_TOPIC = cfg.status_topic


def compute_face_error_x(kps: np.ndarray, frame_width: int) -> float:
    face_center_x = float(np.mean(kps[:, 0]))
    return face_center_x - (float(frame_width) / 2.0)


def command_from_error_with_hysteresis(
    error_x: float,
    deadzone_px: float,
    center_exit_hysteresis_px: float,
    previous_command: str,
) -> str:
    # Wider threshold when leaving CENTERED prevents MOVED_LEFT/MOVED_RIGHT oscillation around center.
    if previous_command == MOVEMENT_CENTER:
        if abs(error_x) <= (float(deadzone_px) + float(center_exit_hysteresis_px)):
            return MOVEMENT_CENTER

    if abs(error_x) <= float(deadzone_px):
        return MOVEMENT_CENTER
    if error_x < 0:
        return MOVEMENT_LEFT
    return MOVEMENT_RIGHT


def build_dashboard_status(
    movement_command: str,
    movement_error_x: float,
    face_lock: Optional[FaceLock],
    faces_count: int,
    locked_face_found: bool,
    confidence_score: Optional[float],
    match_distance: Optional[float],
    fps: Optional[float],
    threshold: float,
    provider_name: str,
) -> Dict[str, object]:
    return {
        "timestamp": time.time(),
        "movement": movement_command,
        "error_x": round(float(movement_error_x), 2),
        "locked": face_lock is not None,
        "target": face_lock.target_name if face_lock else None,
        "locked_face_found": bool(locked_face_found),
        "confidence_score": round(float(confidence_score), 4) if confidence_score is not None else None,
        "match_distance": round(float(match_distance), 4) if match_distance is not None else None,
        "faces": int(faces_count),
        "fps": round(float(fps), 2) if fps is not None else None,
        "threshold": round(float(threshold), 3),
        "provider": provider_name,
    }


class OperationalLogger:
    def __init__(self, csv_path: Path, movement_topic: str):
        self.csv_path = csv_path
        self.movement_topic = movement_topic
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "timestamp_iso",
            "timestamp_epoch",
            "speaker_id",
            "locked",
            "locked_face_found",
            "confidence_score",
            "match_distance",
            "faces",
            "error_x",
            "movement_command",
            "fps",
            "threshold",
            "provider",
            "mqtt_topic",
        ]
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_status(self, status: Dict[str, object]) -> None:
        row = {
            "timestamp_iso": datetime.fromtimestamp(float(status["timestamp"])).isoformat(timespec="milliseconds"),
            "timestamp_epoch": status["timestamp"],
            "speaker_id": status.get("target") or "none",
            "locked": status.get("locked"),
            "locked_face_found": status.get("locked_face_found"),
            "confidence_score": status.get("confidence_score"),
            "match_distance": status.get("match_distance"),
            "faces": status.get("faces"),
            "error_x": status.get("error_x"),
            "movement_command": status.get("movement"),
            "fps": status.get("fps"),
            "threshold": status.get("threshold"),
            "provider": status.get("provider"),
            "mqtt_topic": self.movement_topic,
        }
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class MqttMovementPublisher:
    def __init__(
        self,
        broker_host: str,
        broker_port: int,
        topic: str,
        status_topic: str,
        heartbeat_topic: str,
        client_id: str,
        min_publish_interval: float = 0.15,
        status_min_publish_interval: float = 0.25,
    ):
        self.topic = topic
        self.status_topic = status_topic
        self.heartbeat_topic = heartbeat_topic
        self.min_publish_interval = float(max(0.0, min_publish_interval))
        self.status_min_publish_interval = float(max(0.0, status_min_publish_interval))
        self.last_command: Optional[str] = None
        self.last_publish_at = 0.0
        self.last_status_publish_at = 0.0
        self.connected = False

        if mqtt is None:
            raise RuntimeError("paho-mqtt is not installed. Add it to requirements and run pip install -r requirements.txt")

        self.client = mqtt.Client(client_id=client_id, clean_session=True, protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=5)
        self.client.connect_async(broker_host, int(broker_port), keepalive=30)
        self.client.loop_start()

    def _on_connect(self, _client, _userdata, _flags, rc, _properties=None):
        self.connected = (rc == 0)
        if self.connected:
            print(f"[MQTT] Connected to broker")
            print(f"[MQTT] Movement topic: {self.topic}")
            print(f"[MQTT] Status topic: {self.status_topic}")
            print(f"[MQTT] Heartbeat topic: {self.heartbeat_topic}")
        else:
            print(f"[MQTT] Connect failed with code {rc}")

    def _on_disconnect(self, _client, _userdata, *args):
        # paho-mqtt V1 passes: (rc)
        # paho-mqtt V2 passes: (disconnect_flags, reason_code, properties)
        rc = 0
        if len(args) == 1:
            rc = int(args[0])
        elif len(args) >= 2:
            rc = int(args[1])

        self.connected = False
        if rc != 0:
            print(f"[MQTT] Unexpected disconnect (rc={rc}), retrying...")

    def publish(self, command: str, force: bool = False):
        now = time.time()
        keepalive_interval = (
            min(self.min_publish_interval, 0.08)
            if command in TRACK_MOVEMENT_COMMANDS
            else self.min_publish_interval
        )
        if (
            not force
            and command == self.last_command
            and (now - self.last_publish_at) < keepalive_interval
        ):
            return
        if not self.connected:
            return

        info = self.client.publish(self.topic, payload=command, qos=0, retain=False)
        if info.rc == mqtt.MQTT_ERR_SUCCESS:
            self.last_command = command
            self.last_publish_at = now

    def publish_status(self, status: Dict[str, object], force: bool = False):
        now = time.time()
        if not force and (now - self.last_status_publish_at) < self.status_min_publish_interval:
            return
        if not self.connected:
            return

        payload = json.dumps(status, separators=(",", ":"))
        info = self.client.publish(self.status_topic, payload=payload, qos=0, retain=False)
        if info.rc == mqtt.MQTT_ERR_SUCCESS:
            self.last_status_publish_at = now

    def publish_heartbeat(self, speaker_id: Optional[str] = None, force: bool = False) -> None:
        if not self.connected:
            return
        payload = json.dumps(
            {
                "node": "pc",
                "team": cfg.team_id,
                "status": "ONLINE",
                "speaker_id": speaker_id or "none",
                "timestamp": int(time.time()),
            },
            separators=(",", ":"),
        )
        self.client.publish(self.heartbeat_topic, payload=payload, qos=0, retain=False)

    def close(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face lock tracking with MQTT direction publishing for ESP32 servo control.",
    )
    parser.add_argument("--mqtt-broker", default=cfg.mqtt_broker, help="MQTT broker host/IP.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument(
        "--mqtt-topic",
        default=DEFAULT_MOVEMENT_TOPIC,
        help="MQTT topic to publish movement commands.",
    )
    parser.add_argument(
        "--mqtt-status-topic",
        default=DEFAULT_STATUS_TOPIC,
        help="MQTT topic to publish dashboard status JSON.",
    )
    parser.add_argument(
        "--mqtt-client-id",
        default=f"face-lock-{int(time.time())}",
        help="MQTT client id.",
    )
    parser.add_argument(
        "--speaker-name",
        default=None,
        help="Enrolled speaker to track. Defaults to data/locked_identity.json, then sole DB identity.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=cfg.camera_index,
        help="OpenCV camera index to use for the tracking camera.",
    )
    parser.add_argument(
        "--manual-lock",
        action="store_true",
        help="Require pressing 'l' before tracking instead of automatically locking the enrolled speaker.",
    )
    parser.add_argument(
        "--evidence-log",
        default="",
        help="CSV evidence log path (default: data/logs/<speaker>_evidence_<timestamp>.csv).",
    )
    parser.add_argument(
        "--deadzone-px",
        type=float,
        default=cfg.deadzone_px,
        help="Horizontal pixel deadzone around frame center for CENTERED command.",
    )
    parser.add_argument(
        "--center-exit-hysteresis-px",
        type=float,
        default=cfg.center_exit_hysteresis_px,
        help="Extra pixels required to leave CENTERED and start MOVED_LEFT/MOVED_RIGHT movement.",
    )
    parser.add_argument(
        "--error-smooth-alpha",
        type=float,
        default=cfg.error_smooth_alpha,
        help="EMA smoothing factor for horizontal error (0..1). Lower = smoother.",
    )
    parser.add_argument(
        "--command-confirm-frames",
        type=int,
        default=cfg.command_confirm_frames,
        help="How many consecutive frames are needed before changing MOVED_LEFT/MOVED_RIGHT/CENTERED.",
    )
    parser.add_argument(
        "--search-delay-sec",
        type=float,
        default=cfg.search_delay_sec,
        help="Extra seconds after face loss before SEARCHING is allowed.",
    )
    parser.add_argument(
        "--search-missing-frames",
        type=int,
        default=cfg.search_missing_frames,
        help="Consecutive frames without the locked speaker before SEARCHING starts.",
    )
    parser.add_argument(
        "--reacquire-frames",
        type=int,
        default=cfg.reacquire_frames,
        help="Consecutive frames with the locked speaker before search stops and tracking resumes.",
    )
    parser.add_argument(
        "--search-cooldown-sec",
        type=float,
        default=cfg.search_cooldown_sec,
        help="After re-acquiring the speaker, block SEARCHING for this many seconds.",
    )
    parser.add_argument(
        "--out-of-frame-frames",
        type=int,
        default=cfg.out_of_frame_miss_frames,
        help="Frames without the locked speaker during search before publishing OUT_OF_FRAME.",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=cfg.heartbeat_interval_sec,
        help="Seconds between MQTT heartbeat publishes on vision/Corene/heartbeat.",
    )
    parser.add_argument(
        "--mqtt-min-interval",
        type=float,
        default=cfg.track_publish_interval_sec,
        help="Minimum seconds between repeated identical MQTT commands.",
    )
    parser.add_argument(
        "--mqtt-status-min-interval",
        type=float,
        default=0.25,
        help="Minimum seconds between dashboard status MQTT messages.",
    )
    parser.add_argument(
        "--disable-mqtt",
        action="store_true",
        help="Run face lock tracking without MQTT publishing.",
    )
    return parser.parse_args()

# -------------------------
# Demo
# -------------------------
def save_action_history(face_name: str, actions: List[Action]):
    if not actions:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{face_name}_history_{timestamp}.txt"
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    
    with open(cfg.logs_dir / filename, "w", encoding="utf-8") as f:
        for action in actions:
            time_str = datetime.fromtimestamp(action.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
            f.write(f"{time_str} - {action.type.name}: {action.details}\n")


def create_face_lock(name: str, emb: np.ndarray, kps: np.ndarray, current_time: float) -> FaceLock:
    face_lock = FaceLock(
        target_name=name,
        target_emb=emb,
        last_seen=current_time,
    )
    face_lock.update_position(kps)
    face_lock.history.append(Action(
        ActionType.FACE_LOCKED,
        current_time,
        f"Face locked: {name}",
    ))
    return face_lock

def main():
    args = parse_args()
    args.error_smooth_alpha = float(max(0.01, min(1.0, args.error_smooth_alpha)))
    args.command_confirm_frames = int(max(1, args.command_confirm_frames))
    args.search_delay_sec = float(max(0.0, args.search_delay_sec))
    args.search_missing_frames = int(max(1, args.search_missing_frames))
    args.reacquire_frames = int(max(1, args.reacquire_frames))
    args.search_cooldown_sec = float(max(0.0, args.search_cooldown_sec))
    args.out_of_frame_frames = int(max(1, args.out_of_frame_frames))
    args.heartbeat_interval = float(max(5.0, args.heartbeat_interval))
    args.mqtt_min_interval = float(max(0.0, args.mqtt_min_interval))
    args.mqtt_status_min_interval = float(max(0.0, args.mqtt_status_min_interval))
    args.center_exit_hysteresis_px = float(max(0.0, args.center_exit_hysteresis_px))
    db_path = cfg.db_path
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Select execution provider (CPU/GPU)
    providers = select_provider_interactive()
    provider_name = get_provider_display_name(providers)
    print(f"\nUsing: {provider_name}")
    
    # Note about CUDA warnings
    if "CUDAExecutionProvider" in providers and providers[0] == "CUDAExecutionProvider":
        print("\nNote: CUDA is selected for maximum performance.")
        print("      If you see CUDA errors about missing DLLs, try DirectML instead.\n")
    
    print("=" * 60 + "\n")
    
    det = HaarFaceMesh5pt(
        min_size=(70, 70),
        debug=False,
    )
    embedder = ArcFaceEmbedderONNX(
        model_path=str(cfg.models_dir / "embedder_arcface.onnx"),
        input_size=(112, 112),
        debug=False,
        providers=providers,
    )
    db = load_db_npz(db_path)
    if not db:
        print("Warning: Database is empty. Please enroll identities first.")
        det.close()
        return

    target_speaker: Optional[str] = (
        args.speaker_name.strip() if args.speaker_name else load_lock() or cfg.default_speaker
    )
    if target_speaker and target_speaker not in db and len(db) == 1:
        target_speaker = next(iter(db.keys()))
    if target_speaker:
        if target_speaker not in db:
            print(f"Warning: speaker '{target_speaker}' is not in the face database.")
            print(f"Available identities: {', '.join(sorted(db.keys()))}")
            det.close()
            return
        db = {target_speaker: db[target_speaker]}
    elif len(db) == 1:
        target_speaker = next(iter(db.keys()))
    else:
        print("Warning: Multiple identities are enrolled. Use --speaker-name for strict single-speaker lock.")
    
    # Default threshold 0.40 for better recall (can be adjusted with +/-)
    matcher = FaceDBMatcher(db=db, dist_thresh=cfg.match_threshold)
    
    print(f"Opening camera index: {args.camera_index}")
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera not available")
        det.close()
        return
    
    # Set camera resolution (higher = better face detection, GPU can handle it)
    # Common resolutions: 640x480, 1280x720, 1920x1080
    # With GPU acceleration, higher resolution improves detection quality
    # You can adjust these values based on your camera capabilities
    camera_width = cfg.camera_width
    camera_height = cfg.camera_height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    
    # Verify actual resolution (camera may not support requested resolution)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    if actual_width != camera_width or actual_height != camera_height:
        print(f"  (Requested {camera_width}x{camera_height}, camera using {actual_width}x{actual_height})")
    
    print(f"\nRecognize (multi-face) - Using {provider_name}")
    print("Controls: q=quit, r=reload DB, +/- threshold, d=debug overlay")
    print("          LEFT/RIGHT arrows (or a/f keys) to select face, l=lock/unlock selected face")
    print(
        f"MQTT movement topic: {args.mqtt_topic} @ {args.mqtt_broker}:{args.mqtt_port}"
        if not args.disable_mqtt
        else "MQTT movement publishing disabled"
    )
    if not args.disable_mqtt:
        print(f"MQTT dashboard status topic: {args.mqtt_status_topic}")
    if target_speaker:
        mode = "manual" if args.manual_lock else "automatic"
        print(f"Authorized speaker: {target_speaker} ({mode} lock)")
    if not args.evidence_log:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        speaker_slug = (target_speaker or "session").replace(" ", "_")
        args.evidence_log = str(cfg.logs_dir / f"{speaker_slug}_evidence_{stamp}.csv")
    print(f"Evidence log: {args.evidence_log}")
    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    show_debug = False
    movement_command = MOVEMENT_IDLE
    prev_movement_command = MOVEMENT_IDLE
    movement_error_x = 0.0
    filtered_error_x: Optional[float] = None
    stable_track_command = MOVEMENT_CENTER
    pending_track_command: Optional[str] = None
    pending_track_count = 0
    face_missing_since: Optional[float] = None
    locks_found_streak = 0
    locks_missing_streak = 0
    search_cooldown_until = 0.0
    search_active = False
    out_of_frame_reported = False
    last_locked_center: Optional[Tuple[float, float]] = None
    last_heartbeat_at = time.time()
    mqtt_publisher: Optional[MqttMovementPublisher] = None
    operational_logger = OperationalLogger(Path(args.evidence_log), args.mqtt_topic)

    if args.disable_mqtt:
        print("[MQTT] Disabled by flag (--disable-mqtt)")
    elif mqtt is None:
        print("[MQTT] paho-mqtt is not installed. Movement commands will not be published.")
    else:
        try:
            mqtt_publisher = MqttMovementPublisher(
                broker_host=args.mqtt_broker,
                broker_port=args.mqtt_port,
                topic=args.mqtt_topic,
                status_topic=args.mqtt_status_topic,
                heartbeat_topic=cfg.heartbeat_topic,
                client_id=args.mqtt_client_id,
                min_publish_interval=args.mqtt_min_interval,
                status_min_publish_interval=args.mqtt_status_min_interval,
            )
            mqtt_publisher.publish(MOVEMENT_IDLE, force=True)
            mqtt_publisher.publish_heartbeat(speaker_id=target_speaker, force=True)
        except Exception as e:
            print(f"[MQTT] Failed to initialize publisher: {e}")
            mqtt_publisher = None
    
    # Face locking state
    face_lock: Optional[FaceLock] = None
    max_timeout = cfg.unlock_timeout_sec
    
    # Face selection for locking (when multiple faces present)
    selected_face_index: Optional[int] = None  # Index of currently selected face (None = auto-select first)
    potential_face_to_lock: Optional[Tuple[str, np.ndarray, np.ndarray]] = None  # (name, emb, kps) of selected face
    
    # Temporal smoothing for better accuracy (average recent embeddings per face)
    # Maps face_id -> list of recent embeddings (max smoothing_window)
    face_embedding_history: Dict[int, List[np.ndarray]] = {}
    smoothing_window = 5  # Number of frames to average

    # Additional label smoothing to reduce flicker in displayed names.
    # Maps face_id -> recent predicted labels (e.g., ["Alice", "Alice", "Unknown", ...])
    face_label_history: Dict[int, List[str]] = {}
    label_smoothing_window = 7  # More frames here = more stable, slightly slower to react
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            faces = det.detect(frame, max_faces=5)
            vis = frame.copy()
            
            # compute fps
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()
            
            # draw + recognize each face
            h, w = vis.shape[:2]
            thumb = 112
            pad = 8
            x0 = w - thumb - pad
            y0 = 80
            shown = 0
            
            # Check if we should unlock due to timeout
            current_time = time.time()
            if face_lock and (current_time - face_lock.last_seen) > max_timeout:
                save_action_history(face_lock.target_name, face_lock.history)
                print(f"[FaceLock] Timeout - Unlocked {face_lock.target_name}")
                face_lock = None
                selected_face_index = None
                potential_face_to_lock = None
                filtered_error_x = None
                stable_track_command = MOVEMENT_CENTER
                pending_track_command = None
                pending_track_count = 0
                face_missing_since = None
                locks_found_streak = 0
                locks_missing_streak = 0
                search_cooldown_until = 0.0
                search_active = False
                out_of_frame_reported = False
                last_locked_center = None
                if mqtt_publisher is not None:
                    mqtt_publisher.publish(MOVEMENT_IDLE, force=True)
            
            # Clean up embedding / label history for faces that disappeared
            active_face_ids = set(range(len(faces)))
            face_embedding_history = {k: v for k, v in face_embedding_history.items() if k in active_face_ids}
            face_label_history = {k: v for k, v in face_label_history.items() if k in active_face_ids}
            
            # Reset selection if selected face disappeared
            if selected_face_index is not None and selected_face_index >= len(faces):
                selected_face_index = None
                potential_face_to_lock = None

            locked_face_found = False
            locked_face_kps: Optional[np.ndarray] = None
            locked_confidence_score: Optional[float] = None
            locked_match_distance: Optional[float] = None
            best_lock_match: Optional[Tuple[float, float, int]] = None
            
            for i, f in enumerate(faces):
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                # align -> embed -> temporal smoothing -> match
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb_raw = embedder.embed(aligned)
                
                # Temporal smoothing: average recent embeddings for this face
                if i not in face_embedding_history:
                    face_embedding_history[i] = []
                face_embedding_history[i].append(emb_raw)
                if len(face_embedding_history[i]) > smoothing_window:
                    face_embedding_history[i].pop(0)
                
                # Average embeddings and L2-normalize
                if len(face_embedding_history[i]) > 0:
                    emb_stack = np.stack(face_embedding_history[i], axis=0)
                    emb_smooth = emb_stack.mean(axis=0)
                    emb_smooth = emb_smooth / (np.linalg.norm(emb_smooth) + 1e-12)
                    emb = emb_smooth.astype(np.float32)
                else:
                    emb = emb_raw
                
                mr = matcher.match(emb)

                # -----------------------------
                # Label smoothing (majority vote)
                # -----------------------------
                raw_label = mr.name if mr.name is not None and mr.accepted else "Unknown"
                if i not in face_label_history:
                    face_label_history[i] = []
                face_label_history[i].append(raw_label)
                if len(face_label_history[i]) > label_smoothing_window:
                    face_label_history[i].pop(0)

                hist = face_label_history[i]
                if hist:
                    # Majority label in recent history
                    uniq = set(hist)
                    majority_label = max(uniq, key=hist.count)
                    unknown_count = hist.count("Unknown")
                    unknown_ratio = unknown_count / len(hist)

                    # Only show a name if it dominates recent history and
                    # there aren't too many 'Unknown' frames.
                    if majority_label != "Unknown" and unknown_ratio < 0.5:
                        stable_label = majority_label
                    else:
                        stable_label = "Unknown"
                else:
                    stable_label = raw_label
                
                # Match locked speaker by embedding (ignores label smoothing flicker).
                if face_lock is not None:
                    lock_dist = cosine_distance(emb, face_lock.target_emb)
                    if lock_dist < matcher.dist_thresh:
                        center_x = float(np.mean(f.kps[:, 0]))
                        center_y = float(np.mean(f.kps[:, 1]))
                        if last_locked_center is not None:
                            dx = center_x - last_locked_center[0]
                            dy = center_y - last_locked_center[1]
                            distance_penalty = min(((dx * dx) + (dy * dy)) ** 0.5 / 1000.0, 0.25)
                            lock_score = (1.0 - lock_dist) - distance_penalty
                        else:
                            lock_score = 1.0 - lock_dist
                        if best_lock_match is None or lock_score > best_lock_match[0]:
                            best_lock_match = (lock_score, lock_dist, i)

                is_locked_face = face_lock is not None and best_lock_match is not None and i == best_lock_match[2]
                
                # label (use smoothed/stable label for display to reduce flicker)
                label = stable_label
                status = " (LOCKED)" if is_locked_face else ""
                line1 = f"{label}{status}"
                confidence_pct = max(0.0, min(100.0, mr.similarity * 100.0))
                line2 = f"conf={confidence_pct:.1f}% dist={mr.distance:.3f}"
                
                # Determine if this face is selected for locking
                is_selected = (selected_face_index == i) if selected_face_index is not None else False
                
                # Auto-select first recognized face if no manual selection
                is_authorized_candidate = mr.name and mr.accepted and (target_speaker is None or mr.name == target_speaker)
                if selected_face_index is None and not face_lock and is_authorized_candidate:
                    selected_face_index = i
                    is_selected = True
                    potential_face_to_lock = (mr.name, emb, f.kps)
                
                # Update potential lock target if this is the selected face
                if is_selected and not face_lock:
                    if is_authorized_candidate:
                        potential_face_to_lock = (mr.name, emb, f.kps)
                    else:
                        potential_face_to_lock = None

                if (
                    not args.manual_lock
                    and not face_lock
                    and target_speaker is not None
                    and mr.name == target_speaker
                    and mr.accepted
                ):
                    face_lock = create_face_lock(target_speaker, emb, f.kps, current_time)
                    save_lock(target_speaker)
                    selected_face_index = i
                    potential_face_to_lock = None
                    print(f"[FaceLock] Auto-locked onto {target_speaker} (face {i + 1})")
                
                # color and border: locked > selected > known > unknown
                if is_locked_face:
                    color = (255, 165, 0)  # Orange for locked face
                    border_thickness = 4
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, border_thickness)
                elif is_selected:
                    color = (255, 255, 0)  # Cyan/Yellow for selected face
                    border_thickness = 4
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, border_thickness)
                    # Draw selection indicator
                    cv2.circle(vis, (f.x1 + 15, f.y1 + 15), 8, color, -1)
                    draw_text_with_shadow(vis, "SELECTED", (f.x1, f.y2 + 25), 0.65, color, 1, font=cv2.FONT_HERSHEY_DUPLEX)
                else:
                    color = (0, 255, 0) if mr.accepted else (0, 0, 255)
                    border_thickness = 2
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, border_thickness)
                
                # Draw name label with better font and shadow
                draw_text_with_shadow(vis, line1, (f.x1, max(0, f.y1 - 28)), 0.85, color, 1, font=cv2.FONT_HERSHEY_DUPLEX)
                draw_text_with_shadow(vis, line2, (f.x1, max(0, f.y1 - 6)), 0.65, color, 1, font=cv2.FONT_HERSHEY_DUPLEX)
                
                # Show lock hint for selected face
                if is_selected and not face_lock and potential_face_to_lock:
                    hint_text = f"Press 'l' to lock {potential_face_to_lock[0]}"
                    draw_text_with_shadow(vis, hint_text, (f.x1, f.y2 + 45), 0.65, (255, 255, 0), 1, font=cv2.FONT_HERSHEY_DUPLEX)
                
                # aligned preview thumbnails (stack)
                if y0 + thumb <= h and shown < 4:
                    vis[y0:y0 + thumb, x0:x0 + thumb] = aligned
                    draw_text_with_shadow(
                        vis,
                        f"{i+1}:{label}",
                        (x0, y0 - 6),
                        0.6,
                        color,
                        1,
                        font=cv2.FONT_HERSHEY_DUPLEX,
                    )
                    y0 += thumb + pad
                    shown += 1
                
                if show_debug:
                    dbg = f"kpsLeye=({f.kps[0,0]:.0f},{f.kps[0,1]:.0f})"
                    draw_text_with_shadow(vis, dbg, (10, h - 20), 0.65, (255, 255, 255), 1, font=cv2.FONT_HERSHEY_DUPLEX)

            if face_lock is not None and best_lock_match is not None:
                locked_face_found = True
                locked_face_kps = faces[best_lock_match[2]].kps.copy()
                locked_confidence_score = 1.0 - best_lock_match[1]
                locked_match_distance = best_lock_match[1]
                face_lock.update_position(locked_face_kps)
                last_locked_center = (
                    float(np.mean(locked_face_kps[:, 0])),
                    float(np.mean(locked_face_kps[:, 1])),
                )

            if face_lock and locked_face_found:
                locks_found_streak += 1
                locks_missing_streak = 0
            elif face_lock:
                locks_missing_streak += 1
                locks_found_streak = 0

            speaker_visible = (
                face_lock is not None
                and locks_found_streak >= args.reacquire_frames
                and locked_face_kps is not None
            )
            speaker_missing = (
                face_lock is not None
                and locks_missing_streak >= args.search_missing_frames
            )

            # Movement command for ESP servo.
            # Track left/right while the locked speaker is stably visible; sweep/search when missing.
            reacquired_from_search = False
            if speaker_visible:
                face_missing_since = None
                if search_active:
                    reacquired_from_search = True
                    search_active = False
                    out_of_frame_reported = False
                    search_cooldown_until = current_time + args.search_cooldown_sec
                    filtered_error_x = None
                    stable_track_command = MOVEMENT_CENTER
                    pending_track_command = None
                    pending_track_count = 0
                    movement_command = MOVEMENT_IDLE
                    movement_error_x = 0.0
                    print("[FaceLock] Locked person found. Search stopped immediately.")
                else:
                    raw_error_x = compute_face_error_x(locked_face_kps, frame_width=w)
                    if filtered_error_x is None:
                        filtered_error_x = raw_error_x
                    else:
                        filtered_error_x = (
                            args.error_smooth_alpha * raw_error_x
                            + (1.0 - args.error_smooth_alpha) * filtered_error_x
                        )
                    movement_error_x = float(filtered_error_x)

                    desired_track_command = command_from_error_with_hysteresis(
                        error_x=movement_error_x,
                        deadzone_px=args.deadzone_px,
                        center_exit_hysteresis_px=args.center_exit_hysteresis_px,
                        previous_command=stable_track_command,
                    )

                    if desired_track_command == stable_track_command:
                        pending_track_command = None
                        pending_track_count = 0
                    else:
                        if pending_track_command == desired_track_command:
                            pending_track_count += 1
                        else:
                            pending_track_command = desired_track_command
                            pending_track_count = 1
                        if pending_track_count >= args.command_confirm_frames:
                            stable_track_command = desired_track_command
                            pending_track_command = None
                            pending_track_count = 0

                    movement_command = stable_track_command
            elif face_lock:
                if locks_missing_streak == 1:
                    face_missing_since = current_time
                    face_lock.history.append(Action(
                        ActionType.FACE_LOST,
                        current_time,
                        "Speaker temporarily out of frame or occluded",
                    ))
                lost_for = (
                    current_time - face_missing_since
                    if face_missing_since is not None
                    else 0.0
                )
                if (
                    speaker_missing
                    and lost_for >= args.search_delay_sec
                    and current_time >= search_cooldown_until
                ):
                    search_active = True
                    if locks_missing_streak >= args.out_of_frame_frames:
                        movement_command = MOVEMENT_OUT_OF_FRAME
                        if not out_of_frame_reported:
                            out_of_frame_reported = True
                            face_lock.history.append(Action(
                                ActionType.FACE_LOST,
                                current_time,
                                f"Speaker out of frame ({locks_missing_streak} frames); search continues",
                            ))
                            print("[FaceLock] OUT_OF_FRAME — pan search continues.")
                    else:
                        movement_command = MOVEMENT_SEARCH
                else:
                    movement_command = MOVEMENT_IDLE
                movement_error_x = 0.0
            else:
                filtered_error_x = None
                stable_track_command = MOVEMENT_CENTER
                pending_track_command = None
                pending_track_count = 0
                face_missing_since = None
                movement_command = MOVEMENT_IDLE
                movement_error_x = 0.0

            status_payload = build_dashboard_status(
                movement_command=movement_command,
                movement_error_x=movement_error_x,
                face_lock=face_lock,
                faces_count=len(faces),
                locked_face_found=locked_face_found,
                confidence_score=locked_confidence_score,
                match_distance=locked_match_distance,
                fps=fps,
                threshold=matcher.dist_thresh,
                provider_name=provider_name,
            )
            operational_logger.log_status(status_payload)

            if mqtt_publisher is not None:
                leaving_search = (
                    prev_movement_command in SEARCH_MOVEMENT_COMMANDS
                    and movement_command not in SEARCH_MOVEMENT_COMMANDS
                )
                mqtt_publisher.publish(
                    movement_command,
                    force=leaving_search or reacquired_from_search,
                )
                mqtt_publisher.publish_status(status_payload)
                if current_time - last_heartbeat_at >= args.heartbeat_interval:
                    mqtt_publisher.publish_heartbeat(
                        speaker_id=face_lock.target_name if face_lock else target_speaker,
                        force=True,
                    )
                    last_heartbeat_at = current_time
            prev_movement_command = movement_command
            
            # Draw UI elements with proper spacing and modern styling
            y_offset = 35
            
            # Main header with background box
            header = f"IDs: {len(matcher._names)} | Threshold: {matcher.dist_thresh:.2f}"
            if fps is not None:
                header += f" | FPS: {fps:.1f}"
            draw_text_box(vis, header, (12, y_offset), 0.75, (200, 255, 200), (20, 20, 20), 0.75, 6, cv2.FONT_HERSHEY_DUPLEX)
            y_offset += 40

            movement_text = f"MQTT movement: {movement_command}"
            if movement_command in (MOVEMENT_LEFT, MOVEMENT_RIGHT, MOVEMENT_CENTER):
                movement_text += f" (err_x={movement_error_x:+.1f}px)"
            draw_text_with_shadow(vis, movement_text, (12, y_offset), 0.62, (180, 220, 255), 1, font=cv2.FONT_HERSHEY_DUPLEX)
            y_offset += 28

            if locked_confidence_score is not None:
                confidence_pct = max(0.0, min(100.0, locked_confidence_score * 100.0))
                if confidence_pct >= 80.0:
                    conf_color = (120, 255, 160)
                    conf_quality = "HIGH"
                elif confidence_pct >= 60.0:
                    conf_color = (80, 210, 255)
                    conf_quality = "MEDIUM"
                else:
                    conf_color = (90, 120, 255)
                    conf_quality = "LOW"
                confidence_text = f"Confidence: {confidence_pct:.1f}% ({conf_quality})"
                if locked_match_distance is not None:
                    confidence_text += f" | dist={locked_match_distance:.3f}"
                draw_text_box(vis, confidence_text, (12, y_offset), 0.78, conf_color, (20, 20, 20), 0.76, 7, cv2.FONT_HERSHEY_DUPLEX)
                y_offset += 38
            elif face_lock:
                draw_text_box(vis, "Confidence: no visible locked face", (12, y_offset), 0.72, (120, 160, 255), (30, 0, 0), 0.72, 7, cv2.FONT_HERSHEY_DUPLEX)
                y_offset += 36
            
            # Lock status with enhanced styling
            if face_lock:
                status_text = f"🔒 LOCKED ON: {face_lock.target_name} (Frames: {face_lock.consecutive_frames})"
                draw_text_box(vis, status_text, (12, y_offset), 0.85, (255, 200, 100), (40, 20, 0), 0.8, 8, cv2.FONT_HERSHEY_DUPLEX)
                y_offset += 40
                
                # Last action
                if face_lock.history and len(face_lock.history) > 0:
                    last_action = face_lock.history[-1]
                    action_text = f"Last Action: {last_action.type.name} - {last_action.details}"
                    draw_text_with_shadow(vis, action_text, (12, h - 25), 0.65, (200, 255, 255), 1, font=cv2.FONT_HERSHEY_DUPLEX)
            else:
                if len(faces) > 1:
                    if selected_face_index is not None and selected_face_index < len(faces):
                        status_text = f"👤 Selected face {selected_face_index + 1}/{len(faces)} | ← → to change | 'l' to lock"
                    else:
                        status_text = f"👥 {len(faces)} faces detected | ← → to select | 'l' to lock"
                elif len(faces) == 1:
                    status_text = f"👤 Press 'l' to lock the recognized face"
                else:
                    status_text = f"🔍 No faces detected"
                draw_text_box(vis, status_text, (12, y_offset), 0.75, (200, 255, 200), (0, 40, 0), 0.7, 6, cv2.FONT_HERSHEY_DUPLEX)
                y_offset += 40
            
            # Handle key presses
            key_raw = cv2.waitKey(1)
            key = key_raw & 0xFF
            
            # Handle arrow keys and navigation
            # Arrow keys in OpenCV: when key==0 or 224, the next byte contains the arrow key code
            # Left=75, Right=77, Up=72, Down=80
            # Also support 'a' for left, 'f' for right (to avoid conflict with 'd' for debug)
            arrow_key = None
            if key == 0 or key == 224:  # Extended key indicator
                # Get the actual arrow key code from the next byte
                arrow_key = (key_raw >> 8) & 0xFF
            
            if arrow_key == 75 or key == ord('a'):  # Left arrow or 'a' - previous face
                if not face_lock and len(faces) > 0:
                    if selected_face_index is None:
                        selected_face_index = len(faces) - 1
                    else:
                        selected_face_index = (selected_face_index - 1) % len(faces)
                    potential_face_to_lock = None  # Will be updated in next frame
                    print(f"[FaceSelect] Selected face {selected_face_index + 1}/{len(faces)}")
            elif arrow_key == 77 or key == ord('f'):  # Right arrow or 'f' - next face
                if not face_lock and len(faces) > 0:
                    if selected_face_index is None:
                        selected_face_index = 0
                    else:
                        selected_face_index = (selected_face_index + 1) % len(faces)
                    potential_face_to_lock = None  # Will be updated in next frame
                    print(f"[FaceSelect] Selected face {selected_face_index + 1}/{len(faces)}")
            
            if key == ord("q"):  # Quit
                break
            elif key == ord("r"):  # Reload DB
                matcher.reload_from(db_path)
                if target_speaker:
                    if target_speaker in matcher.db:
                        matcher.db = {target_speaker: matcher.db[target_speaker]}
                        matcher._rebuild()
                    else:
                        print(f"[recognize] target speaker '{target_speaker}' is missing after reload")
                print(f"[recognize] reloaded DB: {len(matcher._names)} identities")
            elif key in (ord("+"), ord("=")):  # Increase threshold
                matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
                print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
            elif key == ord("-"):  # Decrease threshold
                matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
                print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
            elif key == ord("d"):  # Toggle debug
                show_debug = not show_debug
                print(f"[recognize] debug overlay: {'ON' if show_debug else 'OFF'}")
            elif key == ord('l'):  # Lock/unlock face
                if face_lock:  # Unlock if already locked
                    save_action_history(face_lock.target_name, face_lock.history)
                    print(f"[FaceLock] Unlocked {face_lock.target_name}")
                    face_lock = None
                    selected_face_index = None
                    potential_face_to_lock = None
                    filtered_error_x = None
                    stable_track_command = MOVEMENT_CENTER
                    pending_track_command = None
                    pending_track_count = 0
                    face_missing_since = None
                    locks_found_streak = 0
                    locks_missing_streak = 0
                    search_cooldown_until = 0.0
                    search_active = False
                    out_of_frame_reported = False
                    last_locked_center = None
                    if mqtt_publisher is not None:
                        mqtt_publisher.publish(MOVEMENT_IDLE, force=True)
                # Only allow locking if we have a selected recognized face
                elif potential_face_to_lock and potential_face_to_lock[0] is not None:
                    name, emb, kps = potential_face_to_lock
                    face_lock = create_face_lock(name, emb, kps, current_time)
                    save_lock(name)
                    filtered_error_x = None
                    stable_track_command = MOVEMENT_CENTER
                    pending_track_command = None
                    pending_track_count = 0
                    face_missing_since = None
                    print(f"[FaceLock] Locked onto {name} (face {selected_face_index + 1 if selected_face_index is not None else '?'})")
                    # Keep selection for visual feedback, but locking is done
            
            cv2.imshow("Face Recognition - Press 'q' to quit", vis)
    finally:
        if mqtt_publisher is not None:
            mqtt_publisher.publish(MOVEMENT_IDLE, force=True)
            mqtt_publisher.publish_status(
                {
                    "timestamp": time.time(),
                    "movement": MOVEMENT_IDLE,
                    "error_x": 0.0,
                    "locked": False,
                    "target": None,
                    "locked_face_found": False,
                    "faces": 0,
                    "fps": None,
                    "threshold": round(float(matcher.dist_thresh), 3),
                    "provider": provider_name,
                    "shutdown": True,
                },
                force=True,
            )
            mqtt_publisher.close()
        det.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

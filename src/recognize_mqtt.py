# src/recognize.py
"""
Multi-face recognition (CPU-friendly) using your now-stable pipeline:
Haar (multi-face) -> FaceMesh 5pt (per-face ROI) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.
Run:
python addons/mqtt_servo_tracking/recognize_mqtt.py
Keys:
q : quit
r : reload DB from disk (data/db/face_db.npz)
+/- : adjust threshold (distance) live
d : toggle debug overlay
Notes:
- We run FaceMesh on EACH Haar face ROI (not the full frame). This avoids the
"FaceMesh points not consistent with Haar box" problem and enables multi-face.
- DB is expected from enroll: data/db/face_db.npz (name -> embedding vector)
- Distance definition: cosine_distance = 1 - cosine_similarity.
Since embeddings are L2-normalized, cosine_similarity = dot(a,b).
"""
from __future__ import annotations
import argparse
import csv
import json
import time
import os
import sys
import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort

# Suppress MediaPipe and other noisy warnings
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

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
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.haar_5pt import align_face_5pt
from src.onnx_providers import select_provider_interactive, get_provider_display_name

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

@dataclass
class ActionType(Enum):
    FACE_LOCKED = auto()
    FACE_LOST = auto()
    HEAD_LEFT = auto()
    HEAD_RIGHT = auto()
    EYE_BLINK = auto()
    SMILE = auto()

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
    last_position: Optional[Tuple[float, float]] = None
    last_eye_dist: Optional[float] = None
    last_mouth_size: Optional[float] = None
    history: List[Action] = field(default_factory=list)
    consecutive_frames: int = 0
    
    def update_position(self, kps: np.ndarray) -> List[Action]:
        actions = []
        current_time = time.time()
        
        # Calculate face center
        center_x = kps[:, 0].mean()
        center_y = kps[:, 1].mean()
        
        # Detect head movement
        if self.last_position is not None:
            dx = center_x - self.last_position[0]
            # Increased thresholds to reduce noise
            if dx > 20:  # Threshold for right movement (was 10)
                actions.append(Action(ActionType.HEAD_RIGHT, current_time, f"Moved right by {dx:.1f}px"))
            elif dx < -20:  # Threshold for left movement (was -10)
                actions.append(Action(ActionType.HEAD_LEFT, current_time, f"Moved left by {abs(dx):.1f}px"))
        
        # Detect eye blink (using vertical distance between eyes and nose)
        eye_level = (kps[0, 1] + kps[1, 1]) / 2  # Average y of both eyes
        nose_y = kps[2, 1]
        eye_dist = abs(eye_level - nose_y)
        
        if self.last_eye_dist is not None:
            if eye_dist < self.last_eye_dist * 0.7:  # Threshold for blink
                actions.append(Action(ActionType.EYE_BLINK, current_time, "Blink detected"))
        
        # Detect smile (using mouth width/height ratio) - WITH SANITY CHECKS
        mouth_width = abs(kps[3, 0] - kps[4, 0])
        mouth_height = abs(kps[3, 1] - kps[4, 1])
        # Clamp to prevent division issues
        mouth_height = max(mouth_height, 1.0)
        mouth_ratio = mouth_width / mouth_height
        
        # Sanity check: reasonable mouth ratio is between 1.0 and 15.0
        if 1.0 < mouth_ratio < 15.0:
            if self.last_mouth_size is not None and mouth_ratio > 1.8 * self.last_mouth_size:
                actions.append(Action(ActionType.SMILE, current_time, f"Smile detected (ratio: {mouth_ratio:.2f})"))
        else:
            # Reset if unrealistic - use last known good value or default
            mouth_ratio = self.last_mouth_size if self.last_mouth_size is not None else 3.0
        
        # Update state
        self.last_position = (center_x, center_y)
        self.last_eye_dist = eye_dist
        self.last_mouth_size = mouth_ratio
        self.last_seen = current_time
        self.consecutive_frames += 1
        
        return actions

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
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        if self.debug:
            print("[embed] model:", model_path)
            print("[embed] providers:", self.sess.get_providers())
            print("[embed] input:", self.sess.get_inputs()[0].name, self.sess.get_inputs()[0].shape, self.sess.get_inputs()[0].type)
            print("[embed] output:", self.sess.get_outputs()[0].name, self.sess.get_outputs()[0].shape, self.sess.get_outputs()[0].type)

    def _preprocess(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        img = aligned_bgr_112
        if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
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
            running_mode=vision.RunningMode.IMAGE,
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
        return faces.astype(np.int32)

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
        
        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(areas)[::-1]
        faces = faces[order][:max_faces]
        
        out: List[FaceDet] = []
        for (x, y, w, h) in faces:
            mx, my = 0.25 * w, 0.35 * h
            rx1, ry1, rx2, ry2 = _clip_xyxy(x - mx, y - my, x + w + mx, y + h + my, W, H)
            roi = frame_bgr[ry1:ry2, rx1:rx2]
            kps_roi = self._roi_facemesh_5pt(roi)
            if kps_roi is None:
                if self.debug:
                    print("[recognize] FaceMesh none for ROI -> skip")
                continue
            
            kps = kps_roi.copy()
            kps[:, 0] += float(rx1)
            kps[:, 1] += float(ry1)
            
            if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * float(w))):
                if self.debug:
                    print("[recognize] 5pt geometry failed -> skip")
                continue
            
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
        self.db = db
        self.dist_thresh = float(dist_thresh)
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
        e = emb.reshape(1, -1).astype(np.float32)
        sims = (self._mat @ e.T).reshape(-1)
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
    x, y = pos
    for dx, dy in [(shadow_offset, shadow_offset), (-shadow_offset, shadow_offset), 
                   (shadow_offset, -shadow_offset), (-shadow_offset, -shadow_offset)]:
        cv2.putText(img, text, (x + dx, y + dy), font, font_scale, shadow_color, thickness, cv2.LINE_AA)
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
    x, y = pos
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
    
    overlay = img.copy()
    box_x1 = x - padding
    box_y1 = y - text_height - padding
    box_x2 = x + text_width + padding
    box_y2 = y + baseline + padding
    
    h, w = img.shape[:2]
    box_x1 = max(0, box_x1)
    box_y1 = max(0, box_y1)
    box_x2 = min(w, box_x2)
    box_y2 = min(h, box_y2)
    
    if box_x2 > box_x1 and box_y2 > box_y1:
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    draw_text_with_shadow(img, text, (x, y), font_scale, text_color, 1, font=font)
    
    return (text_width + padding * 2, text_height + padding * 2 + baseline)

# -------------------------
# MQTT movement control
# -------------------------
MOVEMENT_LEFT = "LEFT"
MOVEMENT_RIGHT = "RIGHT"
MOVEMENT_CENTER = "CENTER"
MOVEMENT_SEARCH = "SEARCH"
MOVEMENT_IDLE = "IDLE"
DEFAULT_MQTT_BROKER = "157.173.101.159"
DEFAULT_MOVEMENT_TOPIC = "vision/corene/movement"
DEFAULT_STATUS_TOPIC = "vision/corene/status"


def compute_face_error_x(kps: np.ndarray, frame_width: int) -> float:
    face_center_x = float(np.mean(kps[:, 0]))
    return face_center_x - (float(frame_width) / 2.0)


def command_from_error_with_hysteresis(
    error_x: float,
    deadzone_px: float,
    center_exit_hysteresis_px: float,
    previous_command: str,
) -> str:
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
        "faces": int(faces_count),
        "fps": round(float(fps), 2) if fps is not None else None,
        "threshold": round(float(threshold), 3),
        "provider": provider_name,
    }


class MqttMovementPublisher:
    def __init__(
        self,
        broker_host: str,
        broker_port: int,
        topic: str,
        status_topic: str,
        client_id: str,
        min_publish_interval: float = 0.15,
        status_min_publish_interval: float = 0.25,
        heartbeat_interval: float = 0.2,
    ):
        self.topic = topic
        self.status_topic = status_topic
        self.min_publish_interval = float(max(0.0, min_publish_interval))
        self.status_min_publish_interval = float(max(0.0, status_min_publish_interval))
        self.heartbeat_interval = float(max(0.0, heartbeat_interval))
        self.last_command: Optional[str] = None
        self.last_publish_at = 0.0
        self.last_status_publish_at = 0.0
        self.connected = False
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        if mqtt is None:
            raise RuntimeError("paho-mqtt is not installed. Add it to requirements and run pip install -r requirements.txt")

        self.client = mqtt.Client(client_id=client_id, clean_session=True, protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=5)
        self.client.connect_async(broker_host, int(broker_port), keepalive=30)
        self.client.loop_start()

        if self.heartbeat_interval > 0:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

    def _on_connect(self, _client, _userdata, _flags, rc, _properties=None):
        self.connected = (rc == 0)
        if self.connected:
            print(f"[MQTT] Connected to broker")
            print(f"[MQTT] Movement topic: {self.topic}")
            print(f"[MQTT] Status topic: {self.status_topic}")
        else:
            print(f"[MQTT] Connect failed with code {rc}")

    def _on_disconnect(self, _client, _userdata, *args):
        rc = 0
        if len(args) == 1:
            rc = int(args[0])
        elif len(args) >= 2:
            rc = int(args[1])
        self.connected = False
        if rc != 0:
            print(f"[MQTT] Unexpected disconnect (rc={rc}), retrying...")

    def _heartbeat_loop(self):
        while not self._stop_event.wait(self.heartbeat_interval):
            if not self.connected:
                continue
            with self._lock:
                if self.last_command != MOVEMENT_SEARCH:
                    continue
            try:
                self.client.publish(self.topic, payload=MOVEMENT_SEARCH, qos=0, retain=False)
            except Exception:
                pass

    def publish(self, command: str, force: bool = False):
        now = time.time()
        with self._lock:
            if (
                not force
                and command == self.last_command
                and (now - self.last_publish_at) < self.min_publish_interval
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

    def close(self):
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face lock tracking with MQTT direction publishing for ESP32 servo control.",
    )
    parser.add_argument("--mqtt-broker", default=DEFAULT_MQTT_BROKER, help="MQTT broker host/IP.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument("--mqtt-topic", default=DEFAULT_MOVEMENT_TOPIC, help="MQTT topic to publish movement commands.")
    parser.add_argument("--mqtt-status-topic", default=DEFAULT_STATUS_TOPIC, help="MQTT topic to publish dashboard status JSON.")
    parser.add_argument("--mqtt-client-id", default=f"face-lock-{int(time.time())}", help="MQTT client id.")
    parser.add_argument("--deadzone-px", type=float, default=80.0, help="Horizontal pixel deadzone around frame center for CENTER command.")
    parser.add_argument("--center-exit-hysteresis-px", type=float, default=30.0, help="Extra pixels required to leave CENTER and start LEFT/RIGHT movement.")
    parser.add_argument("--error-smooth-alpha", type=float, default=0.35, help="EMA smoothing factor for horizontal error (0..1). Lower = smoother.")
    parser.add_argument("--command-confirm-frames", type=int, default=2, help="How many consecutive frames are needed before changing LEFT/RIGHT/CENTER.")
    parser.add_argument("--search-delay-sec", type=float, default=1.5, help="Delay before sending SEARCH after the locked face is temporarily lost.")
    parser.add_argument("--mqtt-min-interval", type=float, default=0.15, help="Minimum seconds between repeated identical MQTT commands.")
    parser.add_argument("--mqtt-status-min-interval", type=float, default=0.25, help="Minimum seconds between dashboard status MQTT messages.")
    parser.add_argument("--mqtt-search-heartbeat", type=float, default=0.2, help="Seconds between SEARCH heartbeat re-publishes.")
    parser.add_argument("--lock-timeout-sec", type=float, default=30.0, help="Seconds the locked target may be missing before auto-unlock. 0 keeps the lock forever.")
    parser.add_argument("--lost-confirm-frames", type=int, default=8, help="Consecutive frames face must be missing before declaring LOST.")
    parser.add_argument("--found-confirm-frames", type=int, default=4, help="Consecutive frames face must be found before declaring REACQUIRED.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--disable-mqtt", action="store_true", help="Run face lock tracking without MQTT publishing.")
    return parser.parse_args()

# -------------------------
# Session Logger
# -------------------------
class SessionLogger:
    FIELDS = [
        "timestamp_iso", "epoch", "locked_target", "recognized",
        "similarity", "distance", "accepted", "movement",
        "error_x", "faces", "fps",
    ]

    def __init__(self, path: Path, min_interval: float = 1.0):
        self.path = Path(path)
        self.min_interval = float(max(0.0, min_interval))
        self._last_at = 0.0
        self._last_key: Optional[Tuple] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)
        self._writer.writerow(self.FIELDS)
        self._f.flush()

    def log(self, *, locked_target, recognized, similarity, distance, accepted, movement, error_x, faces, fps, force=False):
        now = time.time()
        key = (locked_target, recognized, movement)
        if not force and key == self._last_key and (now - self._last_at) < self.min_interval:
            return
        self._last_at = now
        self._last_key = key
        iso = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._writer.writerow([
            iso, f"{now:.3f}", locked_target or "", recognized or "",
            f"{similarity:.4f}" if similarity is not None else "",
            f"{distance:.4f}" if distance is not None else "",
            "" if accepted is None else int(bool(accepted)),
            movement, f"{error_x:.1f}", int(faces),
            f"{fps:.1f}" if fps is not None else "",
        ])
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def save_action_history(face_name: str, actions: List[Action]):
    if not actions:
        return
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{face_name}_history_{timestamp}.txt"
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{filename}", "w") as f:
        for action in actions:
            time_str = datetime.fromtimestamp(action.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
            f.write(f"{time_str} - {action.type.name}: {action.details}\n")


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    args.error_smooth_alpha = float(max(0.01, min(1.0, args.error_smooth_alpha)))
    args.command_confirm_frames = int(max(1, args.command_confirm_frames))
    args.search_delay_sec = float(max(0.0, args.search_delay_sec))
    args.mqtt_min_interval = float(max(0.0, args.mqtt_min_interval))
    args.mqtt_status_min_interval = float(max(0.0, args.mqtt_status_min_interval))
    args.mqtt_search_heartbeat = float(max(0.0, args.mqtt_search_heartbeat))
    args.lock_timeout_sec = float(max(0.0, args.lock_timeout_sec))
    args.center_exit_hysteresis_px = float(max(0.0, args.center_exit_hysteresis_px))
    args.lost_confirm_frames = int(max(1, args.lost_confirm_frames))
    args.found_confirm_frames = int(max(1, args.found_confirm_frames))
    
    db_path = Path("data/db/face_db.npz")
    os.makedirs("logs", exist_ok=True)
    
    providers = select_provider_interactive()
    provider_name = get_provider_display_name(providers)
    print(f"\nUsing: {provider_name}")
    
    if "CUDAExecutionProvider" in providers and providers[0] == "CUDAExecutionProvider":
        print("\nNote: CUDA is selected for maximum performance.")
        print("      If you see CUDA errors about missing DLLs, try DirectML instead.\n")
    
    print("=" * 60 + "\n")
    
    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx", input_size=(112, 112), debug=False, providers=providers)
    db = load_db_npz(db_path)
    if not db:
        print("Warning: Database is empty. Please enroll identities first.")
        det.close()
        return
    
    matcher = FaceDBMatcher(db=db, dist_thresh=0.40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Camera not available (index 1). Try a different --camera-index.")
        det.close()
        return
    
    camera_width, camera_height = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    if actual_width != camera_width or actual_height != camera_height:
        print(f"  (Requested {camera_width}x{camera_height}, camera using {actual_width}x{actual_height})")
    
    print(f"\nRecognize (multi-face) - Using {provider_name}")
    print("Controls: q=quit, r=reload DB, +/- threshold, d=debug overlay")
    print("          LEFT/RIGHT arrows (or a/f keys) to select face, l=lock/unlock selected face")
    if not args.disable_mqtt:
        print(f"MQTT movement topic: {args.mqtt_topic} @ {args.mqtt_broker}:{args.mqtt_port}")
        print(f"MQTT dashboard status topic: {args.mqtt_status_topic}")

    session_logger = SessionLogger(Path("logs") / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    print(f"[LOG] Evidence log -> {session_logger.path}")
    
    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    show_debug = False
    movement_command = MOVEMENT_IDLE
    movement_error_x = 0.0
    filtered_error_x: Optional[float] = None
    stable_track_command = MOVEMENT_CENTER
    pending_track_command: Optional[str] = None
    pending_track_count = 0
    face_missing_since: Optional[float] = None
    mqtt_publisher: Optional[MqttMovementPublisher] = None

    if args.disable_mqtt:
        print("[MQTT] Disabled by flag (--disable-mqtt)")
    elif mqtt is None:
        print("[MQTT] paho-mqtt is not installed. Movement commands will not be published.")
    else:
        try:
            mqtt_publisher = MqttMovementPublisher(
                broker_host=args.mqtt_broker, broker_port=args.mqtt_port,
                topic=args.mqtt_topic, status_topic=args.mqtt_status_topic,
                client_id=args.mqtt_client_id,
                min_publish_interval=args.mqtt_min_interval,
                status_min_publish_interval=args.mqtt_status_min_interval,
                heartbeat_interval=args.mqtt_search_heartbeat,
            )
            mqtt_publisher.publish(MOVEMENT_IDLE, force=True)
        except Exception as e:
            print(f"[MQTT] Failed to initialize publisher: {e}")
            mqtt_publisher = None
    
    face_lock: Optional[FaceLock] = None
    lock_timeout = float(args.lock_timeout_sec)
    selected_face_index: Optional[int] = None
    potential_face_to_lock: Optional[Tuple[str, np.ndarray, np.ndarray]] = None
    
    face_embedding_history: Dict[int, List[np.ndarray]] = {}
    smoothing_window = 5
    face_label_history: Dict[int, List[str]] = {}
    label_smoothing_window = 7
    
    # --- Loss/Reacquire confirmation counters ---
    face_lost_counter = 0
    face_found_counter = 0
    lost_confirm_frames = args.lost_confirm_frames
    found_confirm_frames = args.found_confirm_frames
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            faces = det.detect(frame, max_faces=5)
            vis = frame.copy()
            
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()
            
            h, w = vis.shape[:2]
            thumb, pad = 112, 8
            x0, y0 = w - thumb - pad, 80
            shown = 0
            
            current_time = time.time()
            
            # Auto-unlock on timeout
            if face_lock and lock_timeout > 0 and (current_time - face_lock.last_seen) > lock_timeout:
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
                face_lost_counter = 0
                face_found_counter = 0
                if mqtt_publisher is not None:
                    mqtt_publisher.publish(MOVEMENT_IDLE, force=True)
            
            # Clean up histories
            active_face_ids = set(range(len(faces)))
            face_embedding_history = {k: v for k, v in face_embedding_history.items() if k in active_face_ids}
            face_label_history = {k: v for k, v in face_label_history.items() if k in active_face_ids}
            
            if selected_face_index is not None and selected_face_index >= len(faces):
                selected_face_index = None
                potential_face_to_lock = None

            locked_face_found = False
            locked_face_kps: Optional[np.ndarray] = None
            locked_face_match: Optional[MatchResult] = None

            for i, f in enumerate(faces):
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb_raw = embedder.embed(aligned)
                
                if i not in face_embedding_history:
                    face_embedding_history[i] = []
                face_embedding_history[i].append(emb_raw)
                if len(face_embedding_history[i]) > smoothing_window:
                    face_embedding_history[i].pop(0)
                
                if len(face_embedding_history[i]) > 0:
                    emb_stack = np.stack(face_embedding_history[i], axis=0)
                    emb_smooth = emb_stack.mean(axis=0)
                    emb_smooth = emb_smooth / (np.linalg.norm(emb_smooth) + 1e-12)
                    emb = emb_smooth.astype(np.float32)
                else:
                    emb = emb_raw
                
                mr = matcher.match(emb)

                raw_label = mr.name if mr.name is not None and mr.accepted else "Unknown"
                if i not in face_label_history:
                    face_label_history[i] = []
                face_label_history[i].append(raw_label)
                if len(face_label_history[i]) > label_smoothing_window:
                    face_label_history[i].pop(0)

                hist = face_label_history[i]
                if hist:
                    uniq = set(hist)
                    majority_label = max(uniq, key=hist.count)
                    unknown_count = hist.count("Unknown")
                    unknown_ratio = unknown_count / len(hist)
                    if majority_label != "Unknown" and unknown_ratio < 0.5:
                        stable_label = majority_label
                    else:
                        stable_label = "Unknown"
                else:
                    stable_label = raw_label
                
                is_locked_face = False
                if face_lock and mr.name == face_lock.target_name and mr.accepted:
                    actions = face_lock.update_position(f.kps)
                    face_lock.history.extend(actions)
                    for action in actions:
                        print(f"[Action] {action.type.name}: {action.details}")
                    is_locked_face = True
                    locked_face_found = True
                    locked_face_kps = f.kps.copy()
                    locked_face_match = mr
                
                label = stable_label
                status = " (LOCKED)" if is_locked_face else ""
                line1 = f"{label}{status}"
                line2 = f"dist={mr.distance:.3f} sim={mr.similarity:.3f}"
                
                is_selected = (selected_face_index == i) if selected_face_index is not None else False
                
                if selected_face_index is None and not face_lock and mr.name and mr.accepted:
                    selected_face_index = i
                    is_selected = True
                    potential_face_to_lock = (mr.name, emb, f.kps)
                
                if is_selected and not face_lock:
                    if mr.name and mr.accepted:
                        potential_face_to_lock = (mr.name, emb, f.kps)
                    else:
                        potential_face_to_lock = None
                
                if is_locked_face:
                    color = (255, 165, 0)
                    border_thickness = 4
                elif is_selected:
                    color = (255, 255, 0)
                    border_thickness = 4
                    cv2.circle(vis, (f.x1 + 15, f.y1 + 15), 8, color, -1)
                    draw_text_with_shadow(vis, "SELECTED", (f.x1, f.y2 + 25), 0.65, color, 1, font=cv2.FONT_HERSHEY_DUPLEX)
                else:
                    color = (0, 255, 0) if mr.accepted else (0, 0, 255)
                    border_thickness = 2
                
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, border_thickness)
                draw_text_with_shadow(vis, line1, (f.x1, max(0, f.y1 - 28)), 0.85, color, 1, font=cv2.FONT_HERSHEY_DUPLEX)
                draw_text_with_shadow(vis, line2, (f.x1, max(0, f.y1 - 6)), 0.65, color, 1, font=cv2.FONT_HERSHEY_DUPLEX)
                
                if is_selected and not face_lock and potential_face_to_lock:
                    hint_text = f"Press 'l' to lock {potential_face_to_lock[0]}"
                    draw_text_with_shadow(vis, hint_text, (f.x1, f.y2 + 45), 0.65, (255, 255, 0), 1, font=cv2.FONT_HERSHEY_DUPLEX)
                
                if y0 + thumb <= h and shown < 4:
                    vis[y0:y0 + thumb, x0:x0 + thumb] = aligned
                    draw_text_with_shadow(vis, f"{i+1}:{label}", (x0, y0 - 6), 0.6, color, 1, font=cv2.FONT_HERSHEY_DUPLEX)
                    y0 += thumb + pad
                    shown += 1
                
                if show_debug:
                    dbg = f"kpsLeye=({f.kps[0,0]:.0f},{f.kps[0,1]:.0f})"
                    draw_text_with_shadow(vis, dbg, (10, h - 20), 0.65, (255, 255, 255), 1, font=cv2.FONT_HERSHEY_DUPLEX)

            # ============================================================
            # MOVEMENT COMMAND WITH LOSS/REACQUIRE HYSTERESIS
            # ============================================================
            if face_lock and locked_face_found:
                # Face IS found this frame
                face_lost_counter = 0  # Reset lost counter
                
                if face_missing_since is not None:
                    # Was previously lost, now confirming reacquire — stop sweep immediately
                    face_found_counter += 1
                    if face_found_counter >= found_confirm_frames:
                        print(f"[FaceLock] {face_lock.target_name} REACQUIRED - tracking")
                        face_missing_since = None
                        face_found_counter = 0
                        filtered_error_x = None  # Reset filter on reacquire
                    movement_command = MOVEMENT_IDLE
                    movement_error_x = 0.0
                else:
                    # Normal tracking - face is visible
                    face_found_counter = 0
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
                # Face is NOT found this frame
                face_found_counter = 0  # Reset found counter
                face_lost_counter += 1
                
                if face_missing_since is None:
                    # Not yet declared lost - need confirmation
                    if face_lost_counter >= lost_confirm_frames:
                        face_missing_since = current_time
                        print(f"[FaceLock] {face_lock.target_name} LOST - holding lock, will SEARCH after {args.search_delay_sec}s")
                        face_lost_counter = 0
                    movement_command = MOVEMENT_CENTER
                else:
                    # Already declared lost
                    lost_for = current_time - face_missing_since
                    if lost_for < args.search_delay_sec:
                        movement_command = MOVEMENT_CENTER
                    else:
                        movement_command = MOVEMENT_SEARCH
                
                movement_error_x = 0.0
            else:
                # No lock at all
                face_lost_counter = 0
                face_found_counter = 0
                filtered_error_x = None
                stable_track_command = MOVEMENT_CENTER
                pending_track_command = None
                pending_track_count = 0
                face_missing_since = None
                movement_command = MOVEMENT_IDLE
                movement_error_x = 0.0

            if mqtt_publisher is not None:
                was_searching = mqtt_publisher.last_command == MOVEMENT_SEARCH
                force_publish = (
                    movement_command != MOVEMENT_SEARCH
                    and (was_searching or (face_lock and locked_face_found and face_missing_since is not None))
                )
                mqtt_publisher.publish(movement_command, force=force_publish)
                mqtt_publisher.publish_status(
                    build_dashboard_status(
                        movement_command=movement_command,
                        movement_error_x=movement_error_x,
                        face_lock=face_lock,
                        faces_count=len(faces),
                        locked_face_found=locked_face_found,
                        fps=fps,
                        threshold=matcher.dist_thresh,
                        provider_name=provider_name,
                    )
                )
            
            session_logger.log(
                locked_target=face_lock.target_name if face_lock else None,
                recognized=(locked_face_match.name if (locked_face_match and locked_face_match.accepted) else None),
                similarity=locked_face_match.similarity if locked_face_match else None,
                distance=locked_face_match.distance if locked_face_match else None,
                accepted=locked_face_match.accepted if locked_face_match else None,
                movement=movement_command,
                error_x=movement_error_x,
                faces=len(faces),
                fps=fps,
            )

            # Draw UI
            y_offset = 35
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
            
            if face_lock:
                status_text = f"LOCKED ON: {face_lock.target_name}"
                if face_missing_since is not None:
                    status_text += f" | LOST {current_time - face_missing_since:.1f}s ago"
                else:
                    status_text += f" | Frames: {face_lock.consecutive_frames}"
                draw_text_box(vis, status_text, (12, y_offset), 0.85, (255, 200, 100), (40, 20, 0), 0.8, 8, cv2.FONT_HERSHEY_DUPLEX)
                y_offset += 40
            else:
                if len(faces) > 1:
                    if selected_face_index is not None and selected_face_index < len(faces):
                        status_text = f"Selected face {selected_face_index + 1}/{len(faces)} | arrows to change | 'l' to lock"
                    else:
                        status_text = f"{len(faces)} faces | arrows to select | 'l' to lock"
                elif len(faces) == 1:
                    status_text = "Press 'l' to lock face"
                else:
                    status_text = "No faces detected"
                draw_text_box(vis, status_text, (12, y_offset), 0.75, (200, 255, 200), (0, 40, 0), 0.7, 6, cv2.FONT_HERSHEY_DUPLEX)
            
            # Key handling
            key_raw = cv2.waitKey(1)
            key = key_raw & 0xFF
            arrow_key = None
            if key == 0 or key == 224:
                arrow_key = (key_raw >> 8) & 0xFF
            
            if arrow_key == 75 or key == ord('a'):
                if not face_lock and len(faces) > 0:
                    if selected_face_index is None:
                        selected_face_index = len(faces) - 1
                    else:
                        selected_face_index = (selected_face_index - 1) % len(faces)
                    potential_face_to_lock = None
                    print(f"[Select] Face {selected_face_index + 1}/{len(faces)}")
            elif arrow_key == 77 or key == ord('f'):
                if not face_lock and len(faces) > 0:
                    if selected_face_index is None:
                        selected_face_index = 0
                    else:
                        selected_face_index = (selected_face_index + 1) % len(faces)
                    potential_face_to_lock = None
                    print(f"[Select] Face {selected_face_index + 1}/{len(faces)}")
            
            if key == ord("q"):
                break
            elif key == ord("r"):
                matcher.reload_from(db_path)
                print(f"[DB] Reloaded: {len(matcher._names)} identities")
            elif key in (ord("+"), ord("=")):
                matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
                print(f"[Thresh] dist={matcher.dist_thresh:.3f}")
            elif key == ord("-"):
                matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
                print(f"[Thresh] dist={matcher.dist_thresh:.3f}")
            elif key == ord("d"):
                show_debug = not show_debug
                print(f"[Debug] {'ON' if show_debug else 'OFF'}")
            elif key == ord('l'):
                if face_lock:
                    save_action_history(face_lock.target_name, face_lock.history)
                    print(f"[Lock] Unlocked {face_lock.target_name}")
                    face_lock = None
                    selected_face_index = None
                    potential_face_to_lock = None
                    filtered_error_x = None
                    stable_track_command = MOVEMENT_CENTER
                    pending_track_command = None
                    pending_track_count = 0
                    face_missing_since = None
                    face_lost_counter = 0
                    face_found_counter = 0
                    if mqtt_publisher is not None:
                        mqtt_publisher.publish(MOVEMENT_IDLE, force=True)
                elif potential_face_to_lock and potential_face_to_lock[0] is not None:
                    name, emb, kps = potential_face_to_lock
                    face_lock = FaceLock(target_name=name, target_emb=emb, last_seen=current_time)
                    face_lock.update_position(kps)
                    face_lock.history.append(Action(ActionType.FACE_LOCKED, current_time, f"Locked: {name}"))
                    filtered_error_x = None
                    stable_track_command = MOVEMENT_CENTER
                    pending_track_command = None
                    pending_track_count = 0
                    face_missing_since = None
                    face_lost_counter = 0
                    face_found_counter = 0
                    print(f"[Lock] Locked onto {name}")
            
            cv2.imshow("Face Recognition - Press 'q' to quit", vis)
    finally:
        if mqtt_publisher is not None:
            mqtt_publisher.publish(MOVEMENT_IDLE, force=True)
            mqtt_publisher.publish_status({
                "timestamp": time.time(), "movement": MOVEMENT_IDLE,
                "error_x": 0.0, "locked": False, "target": None,
                "locked_face_found": False, "faces": 0, "fps": None,
                "threshold": round(float(matcher.dist_thresh), 3),
                "provider": provider_name, "shutdown": True,
            }, force=True)
            mqtt_publisher.close()
        session_logger.close()
        det.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
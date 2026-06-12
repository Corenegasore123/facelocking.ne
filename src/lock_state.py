"""Persist the locked speaker identity between runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from config.corene import cfg

LOCK_FILE = cfg.lock_file


def save_lock(identity: str) -> dict:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": identity.strip(),
        "locked_at": int(time.time()),
    }
    LOCK_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_lock() -> Optional[str]:
    if not LOCK_FILE.exists():
        return None
    payload = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    identity = str(payload.get("identity", "")).strip()
    return identity or None


def clear_lock() -> bool:
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()
        return True
    return False

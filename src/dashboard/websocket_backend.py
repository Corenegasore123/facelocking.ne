"""
MQTT → WebSocket relay for the Corene dashboard.

Run alongside face locking when the broker has no built-in WebSocket listener:
  python -m src.dashboard.websocket_backend

Serve UI on http://localhost:5500 · run this relay on ws://localhost:5501 · set that URL in the dashboard
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Set

import paho.mqtt.client as mqtt
import websockets

from config.corene import cfg

WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", str(cfg.websocket_relay_port)))
MQTT_BROKER = os.getenv("MQTT_BROKER", cfg.mqtt_broker)
MQTT_PORT = int(os.getenv("MQTT_PORT", str(cfg.mqtt_port)))

clients: Set[websockets.WebSocketServerProtocol] = set()
loop: asyncio.AbstractEventLoop | None = None


def _broadcast(message: str) -> None:
    if loop is None or not clients:
        return
    asyncio.run_coroutine_threadsafe(_async_broadcast(message), loop)


async def _async_broadcast(message: str) -> None:
    dead = []
    for client in list(clients):
        try:
            await client.send(message)
        except Exception:
            dead.append(client)
    for client in dead:
        clients.discard(client)


def on_connect(client, _userdata, _flags, rc, _properties=None):
    if rc == 0:
        client.subscribe([
            (cfg.movement_topic, 0),
            (cfg.status_topic, 0),
            (cfg.heartbeat_topic, 0),
        ])
        print(
            f"MQTT connected — subscribed to {cfg.movement_topic}, "
            f"{cfg.status_topic}, {cfg.heartbeat_topic}"
        )
    else:
        print(f"MQTT connect failed: rc={rc}")


def on_message(_client, _userdata, msg):
    topic = msg.topic
    raw = msg.payload.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"movement": raw.strip(), "status": raw.strip()}
    envelope = {
        "type": "mqtt_message",
        "topic": topic,
        "data": data,
        "timestamp": int(time.time()),
    }
    _broadcast(json.dumps(envelope))


async def ws_handler(websocket):
    clients.add(websocket)
    welcome = json.dumps({
        "type": "relay_connected",
        "message": "Corene MQTT relay ready",
        "movement_topic": cfg.movement_topic,
        "status_topic": cfg.status_topic,
        "timestamp": int(time.time()),
    })
    try:
        await websocket.send(welcome)
    except Exception:
        pass
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)


async def run_ws_server():
    global loop
    loop = asyncio.get_running_loop()
    try:
        async with websockets.serve(ws_handler, "0.0.0.0", WEBSOCKET_PORT):
            print(f"WebSocket relay listening on ws://0.0.0.0:{WEBSOCKET_PORT}")
            print(f"Dashboard UI is usually http://localhost:{cfg.dashboard_http_port} (separate port)")
            await asyncio.Future()
    except OSError as exc:
        if getattr(exc, "winerror", None) == 10048 or exc.errno in (98, 10048):
            raise SystemExit(
                f"Port {WEBSOCKET_PORT} is already in use. "
                f"If Live Server uses {cfg.dashboard_http_port}, the relay must use a different port "
                f"(default {cfg.websocket_relay_port}). "
                f"Set WEBSOCKET_PORT=5501 or stop the process holding the port."
            ) from exc
        raise


def run_mqtt():
    client = mqtt.Client(client_id=f"corene-ws-relay-{int(time.time())}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()


def main() -> None:
    threading.Thread(target=run_mqtt, daemon=True).start()
    asyncio.run(run_ws_server())


if __name__ == "__main__":
    main()

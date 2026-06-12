"""
MQTT → WebSocket relay for the Corene dashboard.

Run alongside face locking when the broker has no built-in WebSocket listener:
  python -m src.dashboard.websocket_backend

Open dashboard/index.html and set WebSocket URL to ws://localhost:9002
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
        client.subscribe([(cfg.movement_topic, 0), (cfg.status_topic, 0)])
        print(f"MQTT connected — subscribed to {cfg.movement_topic} and {cfg.status_topic}")
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
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)


async def run_ws_server():
    global loop
    loop = asyncio.get_running_loop()
    async with websockets.serve(ws_handler, "0.0.0.0", WEBSOCKET_PORT):
        print(f"WebSocket relay listening on ws://0.0.0.0:{WEBSOCKET_PORT}")
        await asyncio.Future()


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

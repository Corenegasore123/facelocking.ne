"""Manual MQTT search test — sweeps SEARCHING / STOPPED for servo bring-up."""

from __future__ import annotations

import argparse
import os
import time

import paho.mqtt.client as mqtt

from config.corene import cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish SEARCHING then STOPPED over MQTT for servo tests.")
    parser.add_argument("--broker", default=os.getenv("MQTT_BROKER", cfg.mqtt_broker))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", str(cfg.mqtt_port))))
    parser.add_argument("--topic", default=cfg.movement_topic)
    parser.add_argument("--cycles", type=int, default=1, help="Search/stop cycles (0 = forever).")
    parser.add_argument("--search-sec", type=float, default=3.0, help="Seconds to publish SEARCHING per cycle.")
    args = parser.parse_args()

    client = mqtt.Client(client_id=f"corene-search-test-{int(time.time())}")
    print(f"Connecting to {args.broker}:{args.port}")
    client.connect(args.broker, args.port, 60)
    client.loop_start()

    cycle = 0
    try:
        while args.cycles == 0 or cycle < args.cycles:
            print(f"Publishing {cfg.cmd_search} for {args.search_sec:.1f}s")
            client.publish(args.topic, cfg.cmd_search, qos=0)
            time.sleep(args.search_sec)
            print(f"Publishing {cfg.cmd_stop}")
            client.publish(args.topic, cfg.cmd_stop, qos=0)
            time.sleep(1.0)
            cycle += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.publish(args.topic, cfg.cmd_stop, qos=0)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()

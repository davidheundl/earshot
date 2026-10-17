"""Process-wide settings: shared HMAC key and local ZeroMQ endpoints."""

import os

KEY = os.environ.get("EARSHOT_KEY", "earshot-dev-key-change-me").encode()

HOST = "127.0.0.1"
WORLD_PUB = f"tcp://{HOST}:5550"  # world -> agents ("tick.<id>"), radio + dashboard ("state")
WORLD_PULL = f"tcp://{HOST}:5551"  # agents -> world: command + telemetry
RADIO_PULL = f"tcp://{HOST}:5552"  # agents -> radio: frames to transmit
RADIO_PUB = f"tcp://{HOST}:5553"  # radio -> agents ("rx.<id>"), dashboard ("radio")
DASHBOARD_PORT = int(os.environ.get("EARSHOT_DASHBOARD_PORT", "8000"))
# our own Webots instance, so an already-open Webots does not clash
WEBOTS_PORT = int(os.environ.get("EARSHOT_WEBOTS_PORT", "1240"))
# Gazebo: the bridge process that embeds the Gazebo server (ADR 0008), and gz-transport kept
# on loopback in its own partition, so another Gazebo on the network never joins our bus
GAZEBO_BRIDGE = f"tcp://{HOST}:5554"
GZ_PARTITION = os.environ.get("EARSHOT_GZ_PARTITION", "earshot")

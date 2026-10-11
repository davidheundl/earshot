"""(De)serialisation of process-to-process payloads (not radio frames: those are raw bytes).

Everything on ZeroMQ is a multipart [topic, msgpack(payload)]. Topics end in '.' so that
'tick.1.' does not prefix-match 'tick.10.'.
"""

from __future__ import annotations

import msgpack

from earshot.messages import Mode, Sensor
from earshot.state import Command, DroneState, Sensed, SensorReading


def topic(kind: str, drone: int | None = None) -> bytes:
    return f"{kind}.{drone}.".encode() if drone is not None else f"{kind}.".encode()


def pack(obj) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def unpack(b: bytes):
    return msgpack.unpackb(b, raw=False, strict_map_key=False)


def sensed_to_dict(s: Sensed) -> dict:
    st = s.state
    return {
        "state": {
            "id": st.id, "t": st.t, "x": st.x, "y": st.y, "z": st.z, "heading": st.heading,
            "mode": int(st.mode), "perched_on": st.perched_on, "battery": st.battery,
            "motor_rpm": st.motor_rpm, "camera_on": st.camera_on,
        },
        "readings": [
            [int(r.sensor), r.bearing, r.bearing_sigma, list(r.class_probs), r.level]
            for r in s.readings
        ],
        "camera_frame": s.camera_frame,
    }


def sensed_from_dict(d: dict) -> Sensed:
    st = dict(d["state"])
    st["mode"] = Mode(st["mode"])
    return Sensed(
        state=DroneState(**st),
        readings=[
            SensorReading(Sensor(s), b, bs, tuple(cp), lvl) for s, b, bs, cp, lvl in d["readings"]
        ],
        camera_frame=d.get("camera_frame"),
    )


def command_to_dict(c: Command) -> dict:
    return {"mode": int(c.mode), "goal": c.goal, "perch_site": c.perch_site,
            "camera_on": c.camera_on}


def command_from_dict(d: dict) -> Command:
    return Command(Mode(d["mode"]), tuple(d["goal"]) if d["goal"] else None,
                   d["perch_site"], d["camera_on"])

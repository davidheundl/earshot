"""Radio messages: fixed binary layouts, each frame <= 40 bytes including its auth tag.

This module is the single source of truth for what goes over the (simulated or real LoRa)
radio. Every frame is:

    header (5 B) | payload (type-specific) | tag (10 B, truncated HMAC-SHA256)

Header: type u8 · origin u8 · seq u16 · hops_left u8.
`origin` + `seq` identify an Observation/Track/Claim exactly once across the swarm; that key
is what keeps fusion from double counting (CONTEXT.md: Evidence vs. Track).
Only `hops_left` changes when a frame is forwarded, so it is excluded from the tag.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
from dataclasses import dataclass
from enum import IntEnum

MAX_FRAME = 40
TAG_LEN = 10
_HEADER = struct.Struct("<BBHB")


class MsgType(IntEnum):
    OBSERVATION = 1
    TRACK = 2
    CLAIM = 3
    DIGEST = 4
    REQUEST = 5


class Sensor(IntEnum):
    """Which sensor made an Observation. RF carries its band: the ring radius depends on it."""

    SOUND = 0
    RF_2G4 = 1
    CAMERA = 2
    RF_5G8 = 3
    RF_915 = 4
    RF_UHF = 5  # 400-470 MHz handheld radios (poachers, rangers): sensors/rf.py
    SHOT = 6  # a rifle Shot: one impulse with a sharp bearing (sensors/acoustics.py SHOT)


RF_SENSORS = frozenset({Sensor.RF_2G4, Sensor.RF_5G8, Sensor.RF_915, Sensor.RF_UHF})


class Mode(IntEnum):
    PATROL = 0
    LISTEN = 1
    VERIFY = 2
    FILL_GAP = 3


class TargetClass(IntEnum):
    DRONE = 0
    VEHICLE = 1
    TRAIN = 2
    NOTHING = 3


class TrackStatus(IntEnum):
    SUSPECTED = 0
    EXPLAINED = 1
    VERIFYING = 2
    CONFIRMED = 3
    DISMISSED = 4


# ---------------------------------------------------------------------------------------------
# Payloads. Fields are in SI units / degrees; quantisation happens only in pack/unpack.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Header:
    origin: int
    seq: int
    hops_left: int = 1


@dataclass(frozen=True)
class SensorReading:
    """One sensor's measurement in one 1 s window: one Evidence item."""

    sensor: Sensor
    bearing: float  # degrees, 0 = north, clockwise; NaN if the sensor has no bearing (RF)
    bearing_sigma: float  # degrees
    class_probs: tuple[float, float, float, float]  # indexed by TargetClass
    level: float = 0.0  # sound: received dB(A); RF: received power dBm (-> ring radius);
    #                     camera hit: range in m to a ground target it geolocated (0: bearing
    #                     only); camera "nothing": radius cleared (0: belief.CAMERA_RANGE); < 327
    range: float = 0.0  # m to what the camera saw (geolocated, or a drone's box size); 0: none.
    #                     Stays on the drone: a Follower steers by it, the radio never carries it


@dataclass(frozen=True)
class Observation:
    """One drone's readings of one sensor window, sent together: a drone hearing two Sources
    pays for one frame. Evidence: each reading fused additively, exactly once per (origin, seq)."""

    header: Header
    t: float  # sim seconds
    x: float  # metres, local map frame (east)
    y: float  # metres, local map frame (north)
    mode: Mode
    readings: tuple[SensorReading, ...]  # 1..MAX_READINGS

    TYPE = MsgType.OBSERVATION
    MAX_READINGS = 2  # 5 + 9 + 2 * 8 + 10 = 40 B; more readings -> several Observations
    _S = struct.Struct("<IhhB")  # t, x, y, mode
    # per reading: sensor (3 bits) | bearing (13 bits), sigma, P(drone, vehicle, train), level;
    # P(nothing) is what the three leave (all producers' class_probs sum to 1)
    _R = struct.Struct("<HB3Bh")


@dataclass(frozen=True)
class Track:
    """A fused suspicion. State: replaces the previous Track with the same (origin, track_id)."""

    header: Header
    t: float
    track_id: int
    x: float
    y: float
    semi_major: float  # metres
    semi_minor: float  # metres
    angle: float  # degrees
    cls: TargetClass
    confidence: float  # 0..1
    status: TrackStatus
    strength: float = 0.0  # threat log-odds at the peak, 0..25.5 (probability saturates)

    TYPE = MsgType.TRACK
    _S = struct.Struct("<IHhh BBB BBB B")


@dataclass(frozen=True)
class Claim:
    """'I take task X with bid S until lease expires, going to (x, y).' lease 0 = release."""

    header: Header
    t: float
    task_id: int
    bid: float  # 0..655.35
    lease: float  # seconds
    x: float = 0.0  # target point: the Perch Site or the Suspicion to verify
    y: float = 0.0

    TYPE = MsgType.CLAIM
    _S = struct.Struct("<IHHHhh")


@dataclass(frozen=True)
class Digest:
    """'From origin o I hold everything up to seq s' for up to 6 origins."""

    header: Header
    t: float
    entries: tuple[tuple[int, int], ...]  # (origin, max_seq)

    TYPE = MsgType.DIGEST
    MAX_ENTRIES = 6  # 5 + 5 + 6*3 + 10 = 38 B; more origins -> several Digests


@dataclass(frozen=True)
class Request:
    """'Holder h: I am missing origin o, seq from..to.' Only h answers (agent/store.py)."""

    header: Header
    want_origin: int
    holder: int  # the neighbour whose Digest showed it has them
    from_seq: int
    to_seq: int

    TYPE = MsgType.REQUEST
    _S = struct.Struct("<BBHH")


Message = Observation | Track | Claim | Digest | Request


# ---------------------------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------------------------


def _clamp(v: float, lo: int, hi: int) -> int:
    if not math.isfinite(v):  # round() raises on these; a bad reading must not kill the agent
        return hi if v == math.inf else lo
    return max(lo, min(hi, round(v)))


def _ms(t: float) -> int:
    return _clamp(t * 1000.0, 0, 2**32 - 1)


def _q_prob(p: float) -> int:
    return _clamp(p * 255.0, 0, 255)


BEARING_STEPS = 8190  # 13-bit bearing: 0.044 deg; NO_BEARING marks NaN (RF, camera "nothing")
NO_BEARING = 0x1FFF


def _q_bearing(b: float) -> int:
    if math.isnan(b):
        return NO_BEARING
    return _clamp((b % 360.0) / 360.0 * BEARING_STEPS, 0, BEARING_STEPS) % BEARING_STEPS


# ---------------------------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------------------------


def _tag(key: bytes, signed: bytes) -> bytes:
    return hmac.new(key, signed, hashlib.sha256).digest()[:TAG_LEN]


def _payload(msg: Message) -> bytes:
    if isinstance(msg, Observation):
        if not 1 <= len(msg.readings) <= Observation.MAX_READINGS:
            raise ValueError(f"an Observation holds 1..{Observation.MAX_READINGS} readings")
        out = Observation._S.pack(
            _ms(msg.t),
            _clamp(msg.x, -32768, 32767),
            _clamp(msg.y, -32768, 32767),
            int(msg.mode),
        )
        for r in msg.readings:
            out += Observation._R.pack(
                int(r.sensor) << 13 | _q_bearing(r.bearing),
                _clamp(r.bearing_sigma * 2.0, 0, 255),
                *(_q_prob(p) for p in r.class_probs[:3]),
                _clamp(r.level * 100.0, -32768, 32767),
            )
        return out
    if isinstance(msg, Track):
        return Track._S.pack(
            _ms(msg.t),
            msg.track_id & 0xFFFF,
            _clamp(msg.x, -32768, 32767),
            _clamp(msg.y, -32768, 32767),
            _clamp(msg.semi_major / 5.0, 0, 255),
            _clamp(msg.semi_minor / 5.0, 0, 255),
            _clamp((msg.angle % 180.0) / 180.0 * 255.0, 0, 255),
            int(msg.cls),
            _q_prob(msg.confidence),
            int(msg.status),
            _clamp(msg.strength * 10.0, 0, 255),
        )
    if isinstance(msg, Claim):
        return Claim._S.pack(
            _ms(msg.t),
            msg.task_id & 0xFFFF,
            _clamp(msg.bid * 100.0, 0, 65535),
            _clamp(msg.lease * 1000.0 / 10.0, 0, 65535),  # 10 ms units, max ~655 s
            _clamp(msg.x, -32768, 32767),
            _clamp(msg.y, -32768, 32767),
        )
    if isinstance(msg, Digest):
        if len(msg.entries) > Digest.MAX_ENTRIES:
            raise ValueError(f"Digest holds at most {Digest.MAX_ENTRIES} entries")
        out = struct.pack("<IB", _ms(msg.t), len(msg.entries))
        for origin, seq in msg.entries:
            out += struct.pack("<BH", origin, seq & 0xFFFF)
        return out
    if isinstance(msg, Request):
        return Request._S.pack(msg.want_origin, msg.holder, msg.from_seq & 0xFFFF,
                               msg.to_seq & 0xFFFF)
    raise TypeError(f"not a radio message: {type(msg).__name__}")


def encode(msg: Message, key: bytes) -> bytes:
    h = msg.header
    head = _HEADER.pack(int(msg.TYPE), h.origin, h.seq & 0xFFFF, h.hops_left)
    body = _payload(msg)
    frame_wo_tag = head + body
    signed = frame_wo_tag[:4] + frame_wo_tag[5:]  # everything except hops_left
    frame = frame_wo_tag + _tag(key, signed)
    if len(frame) > MAX_FRAME:
        raise ValueError(f"{type(msg).__name__} frame is {len(frame)} B > {MAX_FRAME} B")
    return frame


class BadFrame(ValueError):
    """Truncated, unknown type, or failed authentication. Drop it."""


def decode(frame: bytes, key: bytes) -> Message:
    if len(frame) < _HEADER.size + TAG_LEN:
        raise BadFrame("too short")
    body, tag = frame[:-TAG_LEN], frame[-TAG_LEN:]
    if not hmac.compare_digest(tag, _tag(key, body[:4] + body[5:])):
        raise BadFrame("bad tag")
    mtype, origin, seq, hops = _HEADER.unpack_from(body)
    header = Header(origin=origin, seq=seq, hops_left=hops)
    p = body[_HEADER.size :]
    try:
        return _unpack(MsgType(mtype), header, p)
    except (struct.error, ValueError) as e:
        raise BadFrame(str(e)) from e


def _unpack(mtype: MsgType, header: Header, p: bytes) -> Message:
    if mtype is MsgType.OBSERVATION:
        n, rest = divmod(len(p) - Observation._S.size, Observation._R.size)
        if n < 1 or rest:
            raise BadFrame(f"Observation payload of {len(p)} B")
        t, x, y, mode = Observation._S.unpack_from(p)
        readings = []
        for i in range(n):
            sb, bs, c0, c1, c2, lvl = Observation._R.unpack_from(
                p, Observation._S.size + i * Observation._R.size)
            b = sb & 0x1FFF
            probs = (c0 / 255.0, c1 / 255.0, c2 / 255.0)
            readings.append(SensorReading(
                Sensor(sb >> 13), float("nan") if b == NO_BEARING else b * 360.0 / BEARING_STEPS,
                bs / 2.0, (*probs, max(0.0, 1.0 - sum(probs))), lvl / 100.0,
            ))
        return Observation(header, t / 1000.0, float(x), float(y), Mode(mode), tuple(readings))
    if mtype is MsgType.TRACK:
        t, tid, x, y, a, b, ang, cls, conf, status, strength = Track._S.unpack(p)
        return Track(
            header, t / 1000.0, tid, float(x), float(y), a * 5.0, b * 5.0,
            ang / 255.0 * 180.0, TargetClass(cls), conf / 255.0, TrackStatus(status),
            strength / 10.0,
        )
    if mtype is MsgType.CLAIM:
        t, task, bid, lease, x, y = Claim._S.unpack(p)
        return Claim(header, t / 1000.0, task, bid / 100.0, lease * 10.0 / 1000.0, float(x),
                     float(y))
    if mtype is MsgType.DIGEST:
        t, n = struct.unpack_from("<IB", p)
        entries = tuple(struct.unpack_from("<BH", p, 5 + 3 * i) for i in range(n))
        return Digest(header, t / 1000.0, entries)
    if mtype is MsgType.REQUEST:
        o, h, a, b = Request._S.unpack(p)
        return Request(header, o, h, a, b)
    raise BadFrame(f"unknown type {mtype}")


def with_hops(frame: bytes, hops_left: int) -> bytes:
    """Re-emit a frame for forwarding. The tag stays valid because hops_left is unsigned."""
    return frame[:4] + bytes([hops_left]) + frame[5:]


def peek_key(frame: bytes) -> tuple[int, int, int]:
    """(type, origin, seq) without verifying the tag. For duplicate filtering only."""
    mtype, origin, seq, _ = _HEADER.unpack_from(frame)
    return mtype, origin, seq


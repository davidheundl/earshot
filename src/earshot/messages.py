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


RF_SENSORS = frozenset({Sensor.RF_2G4, Sensor.RF_5G8, Sensor.RF_915})


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
class Observation:
    """One raw measurement by one drone. Evidence: fused additively, exactly once."""

    header: Header
    t: float  # sim seconds
    x: float  # metres, local map frame (east)
    y: float  # metres, local map frame (north)
    pos_sigma: float  # metres
    bearing: float  # degrees, 0 = north, clockwise; NaN if the sensor has no bearing (RF)
    bearing_sigma: float  # degrees
    class_probs: tuple[float, float, float, float]  # indexed by TargetClass
    sensor: Sensor
    mode: Mode
    level: float = 0.0  # sound: received dB(A); RF: received power dBm (-> ring radius)

    TYPE = MsgType.OBSERVATION
    _S = struct.Struct("<Ihhb HB 4B BBh")


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

    TYPE = MsgType.TRACK
    _S = struct.Struct("<IHhh BBB BBB")


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
    """'I am missing origin o, seq from..to.'"""

    header: Header
    want_origin: int
    from_seq: int
    to_seq: int

    TYPE = MsgType.REQUEST
    _S = struct.Struct("<BHH")


Message = Observation | Track | Claim | Digest | Request


# ---------------------------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------------------------


def _clamp(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, round(v)))


def _ms(t: float) -> int:
    return _clamp(t * 1000.0, 0, 2**32 - 1)


def _q_prob(p: float) -> int:
    return _clamp(p * 255.0, 0, 255)


# ---------------------------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------------------------


def _tag(key: bytes, signed: bytes) -> bytes:
    return hmac.new(key, signed, hashlib.sha256).digest()[:TAG_LEN]


def _payload(msg: Message) -> bytes:
    if isinstance(msg, Observation):
        nan = msg.bearing != msg.bearing
        return Observation._S.pack(
            _ms(msg.t),
            _clamp(msg.x, -32768, 32767),
            _clamp(msg.y, -32768, 32767),
            _clamp(msg.pos_sigma, 0, 127),
            0xFFFF if nan else _clamp((msg.bearing % 360.0) * 100.0, 0, 35999),
            _clamp(msg.bearing_sigma * 2.0, 0, 255),
            *(_q_prob(p) for p in msg.class_probs),
            int(msg.sensor),
            int(msg.mode),
            _clamp(msg.level * 100.0, -32768, 32767),
        )
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
        return Request._S.pack(msg.want_origin, msg.from_seq & 0xFFFF, msg.to_seq & 0xFFFF)
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
        t, x, y, ps, b, bs, c0, c1, c2, c3, sensor, mode, lvl = Observation._S.unpack(p)
        return Observation(
            header, t / 1000.0, float(x), float(y), float(ps),
            float("nan") if b == 0xFFFF else b / 100.0, bs / 2.0,
            (c0 / 255.0, c1 / 255.0, c2 / 255.0, c3 / 255.0),
            Sensor(sensor), Mode(mode), lvl / 100.0,
        )
    if mtype is MsgType.TRACK:
        t, tid, x, y, a, b, ang, cls, conf, status = Track._S.unpack(p)
        return Track(
            header, t / 1000.0, tid, float(x), float(y), a * 5.0, b * 5.0,
            ang / 255.0 * 180.0, TargetClass(cls), conf / 255.0, TrackStatus(status),
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
        o, a, b = Request._S.unpack(p)
        return Request(header, o, a, b)
    raise BadFrame(f"unknown type {mtype}")


def with_hops(frame: bytes, hops_left: int) -> bytes:
    """Re-emit a frame for forwarding. The tag stays valid because hops_left is unsigned."""
    return frame[:4] + bytes([hops_left]) + frame[5:]


def peek_key(frame: bytes) -> tuple[int, int, int]:
    """(type, origin, seq) without verifying the tag. For duplicate filtering only."""
    mtype, origin, seq, _ = _HEADER.unpack_from(frame)
    return mtype, origin, seq


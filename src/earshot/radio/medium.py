"""Radio medium: a shared LoRa-like broadcast channel, simulated deterministically.

Interface:
    medium = RadioMedium(RadioConfig(...), rng)
    delivered = medium.step(t, positions, outgoing, jam_zones)
        -> {receiver_id: [(frame, margin_db), ...]}
    medium.stats() -> dict

`outgoing` are frames each drone handed to its radio since the last step; they queue per
sender by priority (Claim, Request, Track, Observation, Digest; FIFO within one), half duplex,
one frame on air at a time, airtime budget. A frame is ready when handed in plus one random
jitter drawn for it then; it never starts earlier. `margin_db` is the received power above
sensitivity, what a real radio reports as RSSI/SNR; forwarders use it to decide who relays. A frame is delivered at the
end of its airtime to every receiver that (a) is in range after shadowing, (b) was not itself
transmitting during the overlap, (c) is not inside an active jam zone, and (d) beats the summed
power of all overlapping frames (weak ones too) by the capture margin. Shadowing is drawn once
per (frame, receiver), so a link has the same power as wanted frame and as interferer. Same
seed + same inputs = same deliveries.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RadioConfig:
    sf: int = 7
    bw_khz: float = 500.0
    cr: int = 1  # coding rate 4/(4+cr)
    preamble: int = 8
    nominal_range_m: float = 700.0  # median link range; sets tx power vs. sensitivity
    path_loss_exp: float = 2.9  # 2 line of sight, 2.7-3.5 urban
    shadowing_db: float = 6.0
    capture_db: float = 6.0
    duty_cycle: float = 0.10  # max fraction of airtime per sender over the window
    duty_window_s: float = 10.0
    queue_len: int = 32  # per sender; beyond this the least important, newest frame is dropped
    jitter_s: float = 0.3  # random start offset: contention window, no listen-before-talk

    @classmethod
    def from_dict(cls, d: Mapping) -> RadioConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def lora_airtime(payload_bytes: int, sf: int, bw_khz: float, cr: int = 1, preamble: int = 8,
                 explicit_header: bool = True, crc: bool = True) -> float:
    """Seconds on air for one LoRa packet (Semtech AN1200.13)."""
    t_sym = (2**sf) / (bw_khz * 1000.0)
    de = 1 if t_sym > 0.016 else 0
    ih = 0 if explicit_header else 1
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * int(crc) - 20 * ih
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * t_sym + n_payload * t_sym


# send order per sender, by message type byte (messages.MsgType): Claim, Request, Track,
# Observation, Digest
_PRIORITY = {3: 0, 5: 1, 2: 2, 1: 3, 4: 4}


@dataclass
class _Queued:
    frame: bytes
    ready: float  # handed in + its own jitter: the earliest start


def _enqueue(q: deque, item: _Queued) -> None:
    prio = _PRIORITY.get(item.frame[0], 9)
    i = len(q)
    while i > 0 and _PRIORITY.get(q[i - 1].frame[0], 9) > prio:
        i -= 1
    q.insert(i, item)


@dataclass
class _Tx:
    sender: int
    frame: bytes
    start: float
    end: float
    pos: tuple[float, float, float]
    shadow: dict[int, float] = field(default_factory=dict)  # receiver -> dB, drawn once


@dataclass
class _Stats:
    sent: int = 0
    delivered: int = 0  # receiver-frame pairs
    collided: int = 0
    out_of_range: int = 0
    jammed: int = 0
    dropped_queue: int = 0
    airtime: dict = field(default_factory=dict)  # sender -> seconds


class RadioMedium:
    REF_LOSS_DB = 31.7  # free space at 1 m, 915 MHz

    def __init__(self, cfg: RadioConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        # tx power chosen so the median received power at nominal range equals sensitivity
        self._margin_db = self.REF_LOSS_DB + 10 * cfg.path_loss_exp * math.log10(
            cfg.nominal_range_m
        )
        self._queues: dict[int, deque[_Queued]] = {}
        self._busy_until: dict[int, float] = {}
        self._history: dict[int, deque[tuple[float, float]]] = {}  # sender -> (start, airtime)
        self._in_flight: list[_Tx] = []
        self._ended: list[_Tx] = []  # finished, but overlapping a frame still in flight
        self._t = 0.0
        self._s = _Stats()
        self.links: dict[tuple[int, int], float] = {}  # (sender, receiver) -> last delivery t

    # -- interface -----------------------------------------------------------------------

    def step(
        self,
        t: float,
        positions: Mapping[int, tuple[float, float, float]],
        outgoing: Mapping[int, Sequence[bytes]],
        jam_zones: Sequence = (),
    ) -> dict[int, list[tuple[bytes, float]]]:
        for sender in sorted(outgoing):
            q = self._queues.setdefault(sender, deque())
            for f in outgoing[sender]:
                _enqueue(q, _Queued(f, t + float(self.rng.uniform(0, self.cfg.jitter_s))))
                if len(q) > self.cfg.queue_len:
                    q.pop()  # drop the least important, newest frame
                    self._s.dropped_queue += 1
        self._start_transmissions(t, positions)
        delivered = self._finish_transmissions(t, positions, jam_zones)
        self._t = t
        return delivered

    def stats(self) -> dict:
        elapsed = max(self._t, 1e-9)
        total_air = sum(self._s.airtime.values())
        return {
            "t": self._t,
            "sent": self._s.sent,
            "delivered": self._s.delivered,
            "collided": self._s.collided,
            "out_of_range": self._s.out_of_range,
            "jammed": self._s.jammed,
            "dropped_queue": self._s.dropped_queue,
            "channel_load": total_air / elapsed,
            "airtime_per_sender": {k: v / elapsed for k, v in self._s.airtime.items()},
            "queued": {k: len(q) for k, q in self._queues.items()},
            "airtime_40B_s": lora_airtime(40, self.cfg.sf, self.cfg.bw_khz, self.cfg.cr),
        }

    # -- internals -----------------------------------------------------------------------

    def _airtime(self, frame: bytes) -> float:
        c = self.cfg
        return lora_airtime(len(frame), c.sf, c.bw_khz, c.cr, c.preamble)

    def _budget_ok(self, sender: int, start: float, air: float) -> bool:
        h = self._history.setdefault(sender, deque())
        while h and h[0][0] < start - self.cfg.duty_window_s:
            h.popleft()
        used = sum(a for _, a in h)
        return used + air <= self.cfg.duty_cycle * self.cfg.duty_window_s

    def _start_transmissions(self, t: float, positions) -> None:
        for sender in sorted(self._queues):
            q = self._queues[sender]
            while q:
                # not before this tick: a frame held back by the budget starts once it allows
                start = max(self._busy_until.get(sender, 0.0), q[0].ready, self._t)
                if start > t:
                    break
                air = self._airtime(q[0].frame)
                if not self._budget_ok(sender, start, air):
                    break  # wait for the window to slide
                frame = q.popleft().frame
                self._history[sender].append((start, air))
                self._busy_until[sender] = start + air
                pos = positions.get(sender, (0.0, 0.0, 0.0))
                self._in_flight.append(_Tx(sender, frame, start, start + air, pos))
                self._s.sent += 1
                self._s.airtime[sender] = self._s.airtime.get(sender, 0.0) + air

    def _rx_power(self, tx: _Tx, r: int, rpos) -> float:
        """Power of `tx` at receiver `r` above sensitivity, dB; the same whether `tx` is the
        wanted frame or an interferer there."""
        if r not in tx.shadow:
            tx.shadow[r] = float(self.rng.normal(0, self.cfg.shadowing_db))
        d = max(1.0, math.dist(tx.pos, rpos))
        loss = self.REF_LOSS_DB + 10 * self.cfg.path_loss_exp * math.log10(d)
        return self._margin_db - loss + tx.shadow[r]

    def _finish_transmissions(self, t, positions, jam_zones) -> dict[int, list[tuple]]:
        done = [x for x in self._in_flight if x.end <= t]
        if not done:
            return {}
        out: dict[int, list[bytes]] = {}
        jammed = {
            r for r, p in positions.items()
            if any(math.dist(p[:2], z.center) <= z.radius for z in jam_zones)
        }
        for tx in sorted(done, key=lambda x: x.end):
            overlapping = [o for o in self._in_flight + self._ended if o is not tx
                           and o.start < tx.end and tx.start < o.end]
            for r in sorted(positions):
                rpos = positions[r]
                if r == tx.sender:
                    continue
                if r in jammed or tx.sender in jammed:
                    self._s.jammed += 1
                    continue
                if any(o.sender == r for o in overlapping):  # half duplex
                    self._s.collided += 1
                    continue
                p = self._rx_power(tx, r, rpos)
                if p < 0:
                    self._s.out_of_range += 1
                    continue
                if overlapping:
                    i_mw = sum(10 ** (self._rx_power(o, r, rpos) / 10) for o in overlapping)
                    if p - 10 * math.log10(i_mw) < self.cfg.capture_db:
                        self._s.collided += 1
                        continue
                out.setdefault(r, []).append((tx.frame, round(p, 1)))
                self.links[(tx.sender, r)] = t
                self._s.delivered += 1
        self._in_flight = [x for x in self._in_flight if x.end > t]
        # a frame that ends in a later tick still collides with one that ended in this one
        self._ended = [x for x in self._ended + done
                       if any(o.start < x.end for o in self._in_flight)]
        return out

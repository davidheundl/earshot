"""Agent: the whole brain of one drone. Pure and tick-driven (ADR 0001).

Interface:
    agent = Agent(me, city, key, swarm)
    out = agent.step(t, sensed, inbox)     -> StepOut(command, outbox, telemetry, events)

`inbox` holds (frame, margin_db) pairs the radio delivered since the last step; `outbox` holds
raw frames to hand to the radio. No clocks, sockets, threads, randomness or Webots in here: the lockstep
runner and the live agent process drive the exact same object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from earshot.agent.belief import TRACK_SAME, Belief, TrackEst, merge_overlapping, mergeable
from earshot.agent.policy import VERIFY, Policy, task_kind
from earshot.agent.store import MessageStore
from earshot.city import City
from earshot.messages import (
    Claim,
    Digest,
    Header,
    Message,
    Observation,
    Request,
    Track,
    TrackStatus,
    encode,
)
from earshot.state import Command, Sensed, SwarmConfig

OBS_MIN_INTERVAL = 1.0  # s, per sensor (concept 8.2: <= 1/s while listening)
OBS_SLACK = 0.15  # s; 1 s windows on a 0.1 s float clock arrive 0.9-1.1 s apart, never drop one
DIGEST_INTERVAL = 10.0
TRACK_INTERVAL = 1.0  # s between track extractions
TRACK_RESEND = 10.0  # unchanged Threat Class Tracks are re-broadcast this often
EXPLAINED_RESEND = 30.0  # Explained ones (train, vehicle) only this often: nobody acts on them
EXPLAINED_NEAR = 300.0  # m; and one per this radius: a moving train or patrol leaves a trail of
#                         Explained peaks, each in a new 100 m cell (a new track id)
MAX_TRACKS_SENT = 3  # most confident Threat Class Tracks per extraction
TRACK_MOVED = 50.0  # m; a Track that moved more is re-broadcast at once
OBS_HOPS = 1
TRACK_HOPS = 3
EXPLAINED_HOPS = 1  # an Explained Track only informs the neighbours; nobody acts on it
CLAIM_HOPS = 3


@dataclass(frozen=True)
class StepOut:
    command: Command
    outbox: list[bytes]
    telemetry: dict
    events: list[dict] = field(default_factory=list)


class Agent:
    def __init__(self, me: int, city: City, key: bytes, swarm: SwarmConfig):
        self.me = me
        self.key = key
        self.city = city
        self.scheme = swarm.radio_scheme
        self.store = MessageStore(me, key, swarm.radio_scheme, swarm.relay_radius)
        self.threat = swarm.threat
        self.belief = Belief(city, cell=swarm.map_cell, threat=swarm.threat)
        self.policy = Policy(me, city, swarm)
        self._seq = 0
        self._last_obs: dict[int, float] = {}
        self._last_digest = -DIGEST_INTERVAL + 1.7 * me  # stagger across drones
        self._digests = 0
        self._claims: dict[tuple[int, int], Claim] = {}
        self._tracks: list[TrackEst] = []
        self._last_tracks = -1e9
        self._sent_tracks: dict[int, tuple[float, float, int, float]] = {}  # id -> x, y, status, t
        self._sent_explained: list[tuple[float, float, float]] = []  # x, y, t of Explained sent
        self._heatmap: dict | None = None
        self._tracks_json: list[dict] = []
        self._merged: tuple[list[TrackEst], list[Track], list[TrackEst]] | None = None
        self._task: dict | None = None

    def _next_header(self, hops: int) -> Header:
        self._seq = (self._seq + 1) & 0xFFFF
        return Header(origin=self.me, seq=self._seq, hops_left=hops)

    def _emit(self, msg: Message, outbox: list[bytes]) -> None:
        frame = encode(msg, self.key)
        self.store.add_own(frame)
        outbox.extend([frame] * (3 if self.scheme == "unicast3" else 1))

    def step(self, t: float, sensed: Sensed, inbox: list[tuple[bytes, float]]) -> StepOut:
        outbox: list[bytes] = []
        s = sensed.state

        # 1. radio in
        for frame, margin in inbox:
            self._absorb(self.store.receive(frame, margin, t, (s.x, s.y)), t, outbox)

        # 2. own sensors -> Observations (Evidence)
        for r in sensed.readings:
            last = self._last_obs.get(r.sensor, -1e9)
            if last != t and t - last < OBS_MIN_INTERVAL - OBS_SLACK:
                continue  # same-tick readings are different sources; across ticks: <= 1 Hz
            self._last_obs[r.sensor] = t
            hops = TRACK_HOPS if self.scheme != "ours" else OBS_HOPS  # baselines flood Evidence
            obs = Observation(
                self._next_header(hops), t, s.x, s.y, 2.0, r.bearing, r.bearing_sigma,
                r.class_probs, r.sensor, s.mode, r.level,
            )
            self.belief.add_evidence(obs, t)
            self._emit(obs, outbox)

        # 3. fuse: Tracks once per second, broadcast what changed
        fault_events = []
        if t - self._last_tracks >= TRACK_INTERVAL:
            self._last_tracks = t
            for o in sorted(self.belief.check_faults(t)):
                fault_events.append({"t": round(t, 2), "kind": "fault_flagged", "drone": self.me,
                                     "suspect": o})
            self._tracks = self.belief.tracks(t)
            self._tracks_json = [
                {"id": tr.track_id, "x": round(tr.x, 1), "y": round(tr.y, 1),
                 "a": round(tr.semi_major), "b": round(tr.semi_minor), "ang": round(tr.angle),
                 "cls": tr.cls.name, "conf": round(tr.confidence, 2), "status": tr.status.name,
                 "origins": tr.origins, "strength": round(tr.strength, 1),
                 "support": [list(e) for e in tr.support]}
                for tr in self._tracks
            ]
            self._heatmap = self.belief.heatmap()
            self._broadcast_tracks(t, outbox)

        # 4. decide
        decision = self.policy.decide(t, s, self._all_tracks(t), self.belief, self._claims)
        for tid, bid, lease, x, y in decision.claims:
            claim = Claim(self._next_header(CLAIM_HOPS), t, tid, bid, lease, x, y)
            self._claims[(self.me, tid)] = claim
            self._emit(claim, outbox)
        self._task = decision.task

        # 5. relays and healing resends whose time has come; Digest for healing
        outbox.extend(self.store.due(t))
        if t - self._last_digest >= DIGEST_INTERVAL:
            self._last_digest = t
            held = self.store.digest()
            start = (self._digests * Digest.MAX_ENTRIES) % max(1, len(held))
            self._digests += 1  # rotate: with more than 6 origins, successive Digests cover all
            entries = tuple((held[start:] + held[:start])[: Digest.MAX_ENTRIES])
            outbox.append(encode(Digest(Header(self.me, 0, 1), t, entries), self.key))

        telemetry = {
            "mode": decision.command.mode.name,
            "task": self._task,
            "store": len(self.store),
            "accepted": self.store.stats.accepted,
            "duplicates": self.store.stats.duplicates,
            "forwarded": self.store.stats.forwarded,
            "bad": self.store.stats.bad,
            "sent": len(outbox),
            "belief": self.belief.summary(),
            "tracks": self._tracks_json,
            "heatmap": self._heatmap,
        }
        return StepOut(decision.command, outbox, telemetry, decision.events + fault_events)

    # -- tracks ---------------------------------------------------------------------------

    def _verifying_near(self, x: float, y: float, t: float) -> bool:
        return any(task_kind(tid) == VERIFY and c.lease > 0 and t < c.t + c.lease
                   and math.hypot(c.x - x, c.y - y) <= 200.0
                   for (_, tid), c in self._claims.items())

    def _broadcast_tracks(self, t: float, outbox: list[bytes]) -> None:
        threats = sorted((tr for tr in self._tracks if tr.cls == self.threat
                          and tr.status != TrackStatus.EXPLAINED),
                         key=lambda tr: -tr.confidence)[:MAX_TRACKS_SENT]
        explained = [tr for tr in self._tracks if tr.status == TrackStatus.EXPLAINED]
        self._sent_explained = [e for e in self._sent_explained if t - e[2] < EXPLAINED_RESEND]
        for tr in threats + explained:
            if tr.status == TrackStatus.EXPLAINED and any(
                    math.hypot(x - tr.x, y - tr.y) < EXPLAINED_NEAR for x, y, _ in self._sent_explained):
                continue
            resend = EXPLAINED_RESEND if tr.status == TrackStatus.EXPLAINED else TRACK_RESEND
            status = tr.status
            if status == TrackStatus.SUSPECTED and self._verifying_near(tr.x, tr.y, t):
                status = TrackStatus.VERIFYING
            prev = self._sent_tracks.get(tr.track_id)
            if prev is not None and prev[2] == status and t - prev[3] < resend \
                    and math.hypot(prev[0] - tr.x, prev[1] - tr.y) < TRACK_MOVED:
                continue
            if self._reported_by_others(tr, status, t):
                continue  # Trickle-style suppression: a neighbour just said the same thing
            self._sent_tracks[tr.track_id] = (tr.x, tr.y, status, t)
            if status == TrackStatus.EXPLAINED:
                self._sent_explained.append((tr.x, tr.y, t))
            hops = EXPLAINED_HOPS if status == TrackStatus.EXPLAINED else TRACK_HOPS
            self._emit(Track(self._next_header(hops), t, tr.track_id, tr.x, tr.y,
                             tr.semi_major, tr.semi_minor, tr.angle, tr.cls, tr.confidence,
                             status, tr.strength), outbox)

    def _reported_by_others(self, tr: TrackEst, status: TrackStatus, t: float) -> bool:
        return any(rx.header.origin != self.me and t - rx.t < TRACK_RESEND
                   and rx.status == status and rx.cls == tr.cls
                   and math.hypot(rx.x - tr.x, rx.y - tr.y) <= TRACK_SAME
                   for rx in self.belief.received_tracks(t))

    def _all_tracks(self, t: float) -> list[TrackEst]:
        """Own Tracks plus received ones; overlapping drone Suspicions merge into one
        (belief.merge_overlapping). Other received Tracks count only where this drone's own
        map shows nothing. Cached: the inputs change about once a second, not every tick."""
        rx = self.belief.received_tracks(t)
        if self._merged is not None and self._merged[0] is self._tracks \
                and len(self._merged[1]) == len(rx) \
                and all(a is b for a, b in zip(self._merged[1], rx, strict=True)):
            return self._merged[2]
        out = list(self._tracks)
        for tr in rx:
            status = TrackStatus.SUSPECTED if tr.status == TrackStatus.VERIFYING else tr.status
            est = TrackEst(tr.track_id, tr.x, tr.y, tr.semi_major, tr.semi_minor, tr.angle,
                           tr.cls, tr.confidence, status, origins=2, strength=tr.strength)
            if not mergeable(est, self.threat) and any(math.hypot(o.x - tr.x, o.y - tr.y) <= 150.0
                                                        for o in out):
                continue
            out.append(est)
        merged = merge_overlapping(out, self.city.bounds, self.threat)
        self._merged = (self._tracks, rx, merged)
        return merged

    # -- radio ----------------------------------------------------------------------------

    def _absorb(self, msg: Message | None, t: float, outbox: list[bytes]) -> None:
        if msg is None:
            return
        if isinstance(msg, Observation):
            self.belief.add_evidence(msg, t)
        elif isinstance(msg, Track):
            self.belief.set_track(msg)
        elif isinstance(msg, Claim):
            key = (msg.header.origin, msg.task_id)
            prev = self._claims.get(key)
            if prev is None or msg.t >= prev.t:
                self._claims[key] = msg
        elif isinstance(msg, Digest):
            for origin, lo, hi in self.store.on_digest(msg, t):
                self._emit_unstored(Request(Header(self.me, 0, 1), origin, lo, hi), outbox)
        elif isinstance(msg, Request):
            self.store.on_request(msg, t)

    def _emit_unstored(self, msg: Message, outbox: list[bytes]) -> None:
        outbox.append(encode(msg, self.key))

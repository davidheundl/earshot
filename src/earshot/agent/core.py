"""Agent: the whole brain of one drone. Pure and tick-driven (ADR 0001).

Interface:
    agent = Agent(me, city, key, home)
    out = agent.step(t, sensed, inbox)     -> StepOut(command, outbox, telemetry)

`inbox` holds raw frames the radio delivered since the last step; `outbox` holds raw frames to
hand to the radio. No clocks, sockets, threads or Webots in here: the lockstep runner and the
live agent process drive the exact same object.
"""

from __future__ import annotations

from dataclasses import dataclass

from earshot.agent.belief import Belief
from earshot.agent.policy import Policy
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
    encode,
)
from earshot.state import Command, Sensed

OBS_MIN_INTERVAL = 1.0  # s, per sensor (concept 8.2: <= 1/s while listening)
DIGEST_INTERVAL = 10.0
OBS_HOPS = 1
TRACK_HOPS = 3
CLAIM_HOPS = 3


@dataclass(frozen=True)
class StepOut:
    command: Command
    outbox: list[bytes]
    telemetry: dict


class Agent:
    def __init__(self, me: int, city: City, key: bytes, home: tuple[float, float]):
        self.me = me
        self.key = key
        self.store = MessageStore(me, key)
        self.belief = Belief(city)
        self.policy = Policy(me, city, home)
        self._seq = 0
        self._last_obs: dict[int, float] = {}
        self._last_digest = -DIGEST_INTERVAL + 1.7 * me  # stagger across drones
        self._claims: dict[tuple[int, int], Claim] = {}

    def _next_header(self, hops: int) -> Header:
        self._seq = (self._seq + 1) & 0xFFFF
        return Header(origin=self.me, seq=self._seq, hops_left=hops)

    def _emit(self, msg: Message, outbox: list[bytes]) -> None:
        frame = encode(msg, self.key)
        self.store.add_own(frame)
        outbox.append(frame)

    def step(self, t: float, sensed: Sensed, inbox: list[bytes]) -> StepOut:
        outbox: list[bytes] = []
        s = sensed.state

        # 1. radio in
        for frame in inbox:
            msg, fwd = self.store.receive(frame)
            if fwd is not None:
                outbox.append(fwd)
            self._absorb(msg)

        # 2. own sensors -> Observations (Evidence)
        for r in sensed.readings:
            last = self._last_obs.get(r.sensor, -1e9)
            if last != t and t - last < OBS_MIN_INTERVAL:
                continue  # same-tick readings are different sources; across ticks: <= 1 Hz
            self._last_obs[r.sensor] = t
            obs = Observation(
                self._next_header(OBS_HOPS), t, s.x, s.y, 2.0, r.bearing, r.bearing_sigma,
                r.class_probs, r.sensor, s.mode, r.level,
            )
            self.belief.add_evidence(obs)
            self._emit(obs, outbox)

        # 3. think
        self.belief.decay(0.0)
        command, claims = self.policy.decide(t, s, self.belief, list(self._claims.values()))
        for c in claims:
            self._emit(c, outbox)

        # 4. housekeeping: Digest for healing
        if t - self._last_digest >= DIGEST_INTERVAL:
            self._last_digest = t
            entries = tuple(self.store.digest()[: Digest.MAX_ENTRIES])
            outbox.append(encode(Digest(Header(self.me, 0, 1), t, entries), self.key))

        telemetry = {
            "mode": command.mode.name,
            "store": len(self.store),
            "accepted": self.store.stats.accepted,
            "duplicates": self.store.stats.duplicates,
            "forwarded": self.store.stats.forwarded,
            "bad": self.store.stats.bad,
            "sent": len(outbox),
            "belief": self.belief.summary(),
        }
        return StepOut(command, outbox, telemetry)

    def _absorb(self, msg: Message | None) -> None:
        if msg is None:
            return
        if isinstance(msg, Observation):
            self.belief.add_evidence(msg)
        elif isinstance(msg, Track):
            self.belief.set_track(msg)
        elif isinstance(msg, Claim):
            self._claims[(msg.header.origin, msg.task_id)] = msg
        elif isinstance(msg, (Digest, Request)):
            pass  # ON THE DAY: answer with missing frames (store.missing)

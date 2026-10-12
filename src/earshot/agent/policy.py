"""Policy: choose this drone's Mode and Command from its own Belief and the Claims it knows.

Interface:
    policy = Policy(me, city, swarm)
    decision = policy.decide(t, state, tracks, belief, claims)   -> Decision

Concept section 7, as reworked on 2026-10-11:
- One drone per danger, the nearest; everyone else holds their post (perched drones stay on
  their roof, patrollers in their Sector) to catch further dangers. A Suspicion (a drone-class
  Track that is not Explained or Confirmed; overlapping ones are already merged, see
  belief.merge_overlapping) becomes a VERIFY task once it is likely and strong. Task ids are
  kind * 4096 + the Suspicion's 100 m cell, so every bidder can decode where a task is.
- Allocation without a central node: each free drone bids on its nearest task and broadcasts a
  Claim (task, bid, lease, target point). Bid = NEAREST_BID - metres / 10, so the highest bid
  is the nearest drone (ties to the lower id); it follows the Verifier's distance as it flies.
  The auction is by place: Claims within SAME_TASK_RADIUS + 100 m are one danger, whatever
  cell each drone's averaged Suspicion falls in. A drone flies only after holding the best bid
  for a settle window, renews its lease while working, and releases with lease 0. A silent
  drone's lease lapses and someone else bids; a nearer drone that bids later takes over.
- Sectors are Voronoi cells of the homes of available drones (no active Claim); when a drone
  leaves, its neighbours' cells grow and patrollers fly Fill Gap.

Baselines for the evaluation (SwarmConfig.policy):
- "nearest": every non-confirmed Track of any class is verified by the nearest drone; no
  explaining away, no strength gate.
- "no_perch": nobody ever perches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from earshot.agent.belief import CONFIRM_RADIUS, TrackEst, cell_center
from earshot.city import City
from earshot.messages import Claim, Mode, TargetClass, TrackStatus
from earshot.state import Command, DroneState, SwarmConfig

VERIFY = 0  # the only task kind (listening tasks were dropped: listeners hold their posts)
KIND_NAME = {VERIFY: "verify"}
LEASE = 10.0  # s
RENEW = 4.0
SETTLE = 3.0  # > typical 3-hop Claim latency, so competing bids arrive before anyone flies
VERIFY_MIN_CONFIDENCE = 0.5
TOP_SUSPICIONS = 2  # only the strongest Suspicions spawn tasks; ghosts rank below the truth
CONFIRMED_SHADOW = 200.0  # Suspicions this close to a confirmed drone are its side lobes
NEAREST_BID = 655.0  # verify bid = NEAREST_BID - metres / 10: highest bid = nearest drone
VERIFY_MIN_STRENGTH = 4.0  # log-odds; 8 waited ~4x as long to confirm (eval 2026-10-12), 0 floods
SAME_TASK_RADIUS = 200.0  # a Claim of the same kind this close covers a drifted Suspicion
ARRIVE = 60.0  # m; the Verifier holds a 40 m standoff (backend Kinematics.standoff)
DISMISS_AFTER = 8.0  # s at the target with no Suspicion left -> dismissed
SECTOR_CELL = 100.0


TASK_STRIDE = 4096  # 100 m cells per task kind: cities up to 6.4 x 6.4 km; ids stay 14 bits


def task_id(kind: int, cid: int) -> int:
    return kind * TASK_STRIDE + cid


def task_kind(tid: int) -> int:
    return tid // TASK_STRIDE


@dataclass
class Decision:
    command: Command
    claims: list[tuple] = field(default_factory=list)  # (task_id, bid, lease, x, y) to broadcast
    events: list[dict] = field(default_factory=list)  # decision log for metrics
    task: dict | None = None  # for telemetry


@dataclass
class _Task:
    tid: int
    kind: int
    x: float
    y: float
    track: TrackEst | None
    target: tuple[float, float]  # verify point


class Policy:
    def __init__(self, me: int, city: City, swarm: SwarmConfig):
        self.me = me
        self.city = city
        self.swarm = swarm
        self.home = swarm.homes[me]
        self.fixed = me in swarm.fixed
        self.default = Mode.PATROL if swarm.policy == "no_perch" and not self.fixed \
            else swarm.defaults[me]
        if self.fixed:
            self.default = Mode.LISTEN
        self._sites = {p.id: p for p in city.perch_sites}
        self._home_site = self._best_roof_near(self.home)
        x0, y0, x1, y1 = city.bounds
        if math.ceil((x1 - x0) / 100.0) * math.ceil((y1 - y0) / 100.0) > TASK_STRIDE:
            raise ValueError(f"city too large for task ids: more than {TASK_STRIDE} 100 m cells")
        self._task: _Task | None = None
        self._claimed_at = 0.0
        self._last_claim = -1e9
        self._settled = False
        self._my_bid = 0.0
        self._arrived_at: float | None = None
        xs = np.arange(city.bounds[0] + SECTOR_CELL / 2, city.bounds[2], SECTOR_CELL)
        ys = np.arange(city.bounds[1] + SECTOR_CELL / 2, city.bounds[3], SECTOR_CELL)
        self._SX, self._SY = np.meshgrid(xs, ys)
        self._sectors: dict[tuple[int, ...], tuple] = {}  # available ids -> _sector()

    # -- main -----------------------------------------------------------------------------

    def decide(self, t: float, s: DroneState, tracks: list[TrackEst], belief,
               claims: dict[tuple[int, int], Claim]) -> Decision:
        active = {k: c for k, c in claims.items() if c.lease > 0 and t < c.t + c.lease}
        tasks = self._tasks(tracks, belief, active, s)
        d = Decision(command=self._default_command(t, s, active))
        if self.fixed:
            return d  # a microphone array on a roof: it listens, it never flies

        if self._task is not None:
            self._continue(t, s, tasks, active, tracks, d)
        if self._task is None:
            self._bid_for_best(t, s, tasks, active, d)

        if self._task is not None and self._settled:
            d.command = self._task_command(s)
        if self._task is not None:
            d.task = {"kind": KIND_NAME[self._task.kind], "tid": self._task.tid,
                      "x": self._task.target[0], "y": self._task.target[1],
                      "settled": self._settled, "bid": round(self._my_bid, 2)}
        return d

    # -- tasks ----------------------------------------------------------------------------

    def _tasks(self, tracks, belief, active, s) -> dict[int, _Task]:
        out: dict[int, _Task] = {}
        confirmed = [tr for tr in tracks if tr.status == TrackStatus.CONFIRMED]
        tracks = sorted((tr for tr in tracks if tr.status != TrackStatus.CONFIRMED and not any(
            math.hypot(c.x - tr.x, c.y - tr.y) <= CONFIRMED_SHADOW for c in confirmed)),
            key=lambda tr: -tr.strength)
        if self.swarm.policy != "nearest":
            tracks = [tr for tr in tracks if tr.cls == TargetClass.DRONE
                      and tr.status == TrackStatus.SUSPECTED][:TOP_SUSPICIONS]
        for tr in tracks:
            if tr.status == TrackStatus.CONFIRMED:
                continue
            if self.swarm.policy == "nearest":
                if tr.confidence >= VERIFY_MIN_CONFIDENCE:
                    tid = task_id(VERIFY, tr.track_id)
                    out[tid] = _Task(tid, VERIFY, tr.x, tr.y, tr, (tr.x, tr.y))
                continue
            if tr.cls != TargetClass.DRONE or tr.status != TrackStatus.SUSPECTED:
                continue  # Explained: not every alarm deserves a drone
            # one Verifier per Suspicion and nobody else moves: the other drones hold their
            # posts to catch further dangers (no listening tasks)
            if tr.confidence >= VERIFY_MIN_CONFIDENCE and tr.strength >= VERIFY_MIN_STRENGTH:
                tid = task_id(VERIFY, tr.track_id)
                out[tid] = _Task(tid, VERIFY, tr.x, tr.y, tr, (tr.x, tr.y))
        return out

    def _rival(self, task: _Task, active) -> tuple[float, int] | None:
        """The best (bid, -origin) among other drones' Claims for the same danger: any task
        id of the same kind within SAME_TASK_RADIUS + 100 m. Each drone's (averaged)
        Suspicion can sit in a different 100 m cell, so the auction is by place, not by id."""
        best = None
        for (origin, tid), c in active.items():
            if origin == self.me or task_kind(tid) != task.kind:
                continue
            if math.hypot(c.x - task.x, c.y - task.y) <= SAME_TASK_RADIUS + 100.0:
                best = max(best or (c.bid, -origin), (c.bid, -origin))
        return best

    # -- bidding --------------------------------------------------------------------------

    def _bid_for_best(self, t, s, tasks, active, d: Decision) -> None:
        best: tuple[float, _Task] | None = None
        for task in tasks.values():
            bid = self._bid(task, s)
            rival = self._rival(task, active)
            if rival is not None and rival >= (bid, -self.me):
                continue  # a nearer drone has it
            if best is None or bid > best[0]:
                best = (bid, task)
        if best is None:
            return
        self._my_bid, self._task = best
        self._claimed_at, self._settled, self._arrived_at = t, False, None
        self._send_claim(t, d, LEASE)
        d.events.append(self._event(t, "claim"))

    def _bid(self, task: _Task, s: DroneState) -> float:
        """The nearest drone bids highest, whatever it is doing now; rounded as the Claim
        carries it, so every drone compares the same numbers."""
        dist = math.hypot(task.x - s.x, task.y - s.y)
        return round(max(0.01, NEAREST_BID - dist / 10.0) * 100.0) / 100.0

    # -- holding a task -------------------------------------------------------------------

    def _continue(self, t, s, tasks, active, tracks, d: Decision) -> None:
        task = self._task
        current = tasks.get(task.tid) or self._drifted(task, tasks) or self._held(task, tracks)
        self._verify_progress(t, s, tracks, current, d)
        if self._task is None:
            return
        if current is not None:
            task.x, task.y, task.track = current.x, current.y, current.track
            task.target = (current.x, current.y)
        self._my_bid = self._bid(task, s)  # it gets nearer as it flies
        rival = self._rival(task, active)
        if rival is not None and rival > (self._my_bid, -self.me):
            self._task = None  # a nearer drone has it: let my lease lapse
            d.events.append({"t": round(t, 2), "kind": "outbid", "drone": self.me})
            return
        if not self._settled and t - self._claimed_at >= SETTLE:
            self._settled = True
            d.events.append(self._event(t, "dispatch"))
        if t - self._last_claim >= RENEW:
            self._send_claim(t, d, LEASE)

    def _drifted(self, task: _Task, tasks) -> _Task | None:
        same = [x for x in tasks.values() if task_kind(x.tid) == task.kind
                and math.hypot(x.x - task.x, x.y - task.y) <= SAME_TASK_RADIUS]
        return min(same, key=lambda x: math.hypot(x.x - task.x, x.y - task.y), default=None)

    def _held(self, task: _Task, tracks) -> _Task | None:
        """A task once claimed lasts while its Suspicion does, below the strength gate too:
        strengths hover around the gate, and a claim dropped before it settles flies nobody."""
        same = [tr for tr in tracks if tr.cls == TargetClass.DRONE
                and tr.status == TrackStatus.SUSPECTED and tr.confidence >= VERIFY_MIN_CONFIDENCE
                and math.hypot(tr.x - task.x, tr.y - task.y) <= SAME_TASK_RADIUS]
        tr = min(same, key=lambda tr: math.hypot(tr.x - task.x, tr.y - task.y), default=None)
        return None if tr is None else _Task(task.tid, task.kind, tr.x, tr.y, tr, (tr.x, tr.y))

    def _verify_progress(self, t, s, tracks, current, d: Decision) -> None:
        task = self._task
        confirmed = next((tr for tr in tracks if tr.status == TrackStatus.CONFIRMED
                          and math.hypot(tr.x - task.x, tr.y - task.y) <= CONFIRM_RADIUS), None)
        if confirmed is not None and self._settled:
            self._release(t, d, "confirmed", confirmed)
            return
        dist = math.hypot(task.target[0] - s.x, task.target[1] - s.y)
        if self._settled and dist <= ARRIVE and self._arrived_at is None:
            self._arrived_at = t
        if current is None and self._arrived_at is not None and t - self._arrived_at >= DISMISS_AFTER:
            self._release(t, d, "dismissed")
        elif current is None and not self._settled:
            self._release(t, d, "obsolete")

    def _release(self, t, d: Decision, why: str, track: TrackEst | None = None) -> None:
        ev = self._event(t, why)
        if track is not None:
            ev.update(x=round(track.x, 1), y=round(track.y, 1))
        d.events.append(ev)
        self._send_claim(t, d, 0.0)
        self._task = None
        self._settled = False

    def _send_claim(self, t, d: Decision, lease: float) -> None:
        task = self._task
        d.claims.append((task.tid, max(0.0, self._my_bid), lease, task.target[0], task.target[1]))
        self._last_claim = t

    def _event(self, t, kind) -> dict:
        task = self._task
        ev = {"t": round(t, 2), "kind": kind, "drone": self.me, "task": KIND_NAME[task.kind],
              "tid": task.tid, "x": round(task.x, 1), "y": round(task.y, 1)}
        if task.track is not None:
            ev.update(conf=round(task.track.confidence, 2), ellipse=round(task.track.semi_major))
        return ev

    # -- commands -------------------------------------------------------------------------

    def _task_command(self, s: DroneState) -> Command:
        target = self._task.target
        near = math.hypot(target[0] - s.x, target[1] - s.y) <= 150.0
        return Command(Mode.VERIFY, goal=target, camera_on=near, look_at=target)

    def _default_command(self, t, s: DroneState, active) -> Command:
        busy = {o for (o, _tid) in active}  # drones holding a Claim have left their Sector
        available = [i for i in self.swarm.homes if i == self.me or i not in busy]
        cx, cy, radius, grew = self._sector(available)
        if self.default == Mode.LISTEN and self._home_site is not None:
            return Command(Mode.LISTEN, perch_site=self._home_site.id)
        a = t / 25.0 + self.me
        goal = (cx + radius * math.cos(a), cy + radius * math.sin(a))
        return Command(Mode.FILL_GAP if grew else Mode.PATROL, goal=goal)

    def _sector(self, available: list[int]):
        """Centroid, patrol radius and whether my Voronoi cell grew beyond its full-swarm size."""
        key = tuple(sorted(available))
        if key not in self._sectors:  # depends only on which drones are home
            self._sectors[key] = self._compute_sector(available)
        return self._sectors[key]

    def _compute_sector(self, available: list[int]):
        def cell(ids):
            homes = np.array([self.swarm.homes[i] for i in ids])
            d = np.hypot(self._SX[..., None] - homes[:, 0], self._SY[..., None] - homes[:, 1])
            mine = np.argmin(d, axis=-1) == ids.index(self.me)
            return mine
        mine = cell(sorted(available))
        full = mine if len(available) == len(self.swarm.homes) else cell(sorted(self.swarm.homes))
        area = mine.sum() * SECTOR_CELL**2
        cx, cy = self._SX[mine].mean(), self._SY[mine].mean()
        return float(cx), float(cy), 0.35 * math.sqrt(area / math.pi), mine.sum() > full.sum()

    def _best_roof_near(self, p) -> object | None:
        return min(self._sites.values(), key=lambda r: math.hypot(r.x - p[0], r.y - p[1]),
                   default=None)


def decode_task(tid: int, bounds) -> tuple[str, float, float]:
    x, y = cell_center(tid % TASK_STRIDE, bounds)
    return KIND_NAME[task_kind(tid)], x, y

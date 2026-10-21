"""Policy: choose this drone's Mode and Command from its own Belief and the Claims it knows.

Interface:
    policy = Policy(me, city, swarm)
    decision = policy.decide(t, state, tracks, belief, claims)   -> Decision

Concept section 7, as reworked on 2026-10-11:
- One drone per danger, the nearest; everyone else holds their post (perched drones stay on
  their roof, patrollers in their Sector) to catch further dangers. A Suspicion (a Track of
  the Threat Class, SwarmConfig.threat, that is not Explained or Confirmed; overlapping ones
  are already merged, see belief.merge_overlapping) becomes a VERIFY task once it is likely
  and strong. Task ids are kind * 16384 + the Suspicion's 100 m cell, so every bidder can
  decode where a task is.
- Allocation without a central node: each free drone bids on its nearest task and broadcasts a
  Claim (task, bid, lease, target point). Bid = NEAREST_BID - metres / 10, so the highest bid
  is the nearest drone (ties to the lower id); it follows the Verifier's distance as it flies.
  The auction is by place: Claims within same_task_radius + 100 m are one danger, whatever
  cell each drone's averaged Suspicion falls in. A drone flies only after holding the best bid
  for a settle window (its Claim is sent again halfway through, so one lost frame cannot let a
  rival settle too), renews its lease while working, and releases with lease 0, also when
  outbid. A silent drone's lease lapses and someone else bids; a nearer drone that bids later
  takes over. An outbid drone waits outbid_cooldown before bidding near that place again.
- A task follows its Suspicion as it drifts, but at most same_task_radius from where it was
  claimed; beyond that it is dropped and re-auctioned.
- Follow (follow_for > 0): a Verifier that confirmed its threat stays with it, camera on,
  flying to the nearest Confirmed Track of the Threat Class and renewing its Claim, until
  follow_for has passed ("handed_off") or no Confirmed Track is near for dismiss_after
  ("lost"). It reports where it is once a second ("following"); the runners log that as the
  Threat Log. follow_for 0 releases at confirmation, as before.
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
from earshot.messages import Claim, Mode, TrackStatus
from earshot.state import Command, DroneState, SwarmConfig

VERIFY = 0  # the only task kind (listening tasks were dropped: listeners hold their posts)
KIND_NAME = {VERIFY: "verify"}
LEASE = 10.0  # s
RENEW = 4.0
NEAREST_BID = 655.0  # verify bid = NEAREST_BID - metres / 10: highest bid = nearest drone
# the tunable thresholds (strength gate, settle window, ...) are state.PolicyParams: self.p
SECTOR_CELL = 100.0


TASK_STRIDE = 16384  # 100 m cells per task kind: cities up to 12.8 x 12.8 km; one kind fits u16


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
    claimed: tuple[float, float] | None = None  # where it was claimed: drift is capped from here


class Policy:
    def __init__(self, me: int, city: City, swarm: SwarmConfig):
        self.me = me
        self.city = city
        self.swarm = swarm
        self.p = swarm.params
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
        self._reclaimed = False  # the Claim was sent again halfway through settle
        self._my_bid = 0.0
        self._arrived_at: float | None = None
        self._lost: tuple[float, float, float] | None = None  # (t, x, y) of the last outbid
        self._follow_since: float | None = None  # Follow: when the task's threat was confirmed
        self._unseen_since: float | None = None  # Follow: no Confirmed Track near since then
        self._follow_said = -1e9  # Follow: last "following" event
        self._cooldown = self.p.outbid_cooldown
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
            if self._follow_since is not None:
                d.task["following"] = True
        return d

    # -- tasks ----------------------------------------------------------------------------

    def _tasks(self, tracks, belief, active, s) -> dict[int, _Task]:
        out: dict[int, _Task] = {}
        confirmed = [tr for tr in tracks if tr.status == TrackStatus.CONFIRMED]
        tracks = sorted((tr for tr in tracks if tr.status != TrackStatus.CONFIRMED and not any(
            math.hypot(c.x - tr.x, c.y - tr.y) <= self.p.confirmed_shadow for c in confirmed)),
            key=lambda tr: -tr.strength)
        if self.swarm.policy != "nearest":
            tracks = [tr for tr in tracks if tr.cls == self.swarm.threat
                      and tr.status == TrackStatus.SUSPECTED][:self.p.top_suspicions]
        for tr in tracks:
            if tr.status == TrackStatus.CONFIRMED:
                continue
            if self.swarm.policy == "nearest":
                if tr.confidence >= self.p.verify_min_confidence:
                    tid = task_id(VERIFY, tr.track_id)
                    out[tid] = _Task(tid, VERIFY, tr.x, tr.y, tr, (tr.x, tr.y))
                continue
            if tr.cls != self.swarm.threat or tr.status != TrackStatus.SUSPECTED:
                continue  # Explained: not every alarm deserves a drone
            if math.hypot(tr.x - s.x, tr.y - s.y) > self.p.max_verify_distance:
                continue  # a nearer drone is there, though its Claim may not reach me
            # one Verifier per Suspicion and nobody else moves: the other drones hold their
            # posts to catch further dangers (no listening tasks)
            if (tr.confidence >= self.p.verify_min_confidence
                    and tr.strength >= self.p.verify_min_strength):
                tid = task_id(VERIFY, tr.track_id)
                out[tid] = _Task(tid, VERIFY, tr.x, tr.y, tr, (tr.x, tr.y))
        return out

    def _rival(self, task: _Task, active) -> tuple[float, int] | None:
        """The best (bid, -origin) among other drones' Claims for the same danger: any task
        id of the same kind within same_task_radius + 100 m. Each drone's (averaged)
        Suspicion can sit in a different 100 m cell, so the auction is by place, not by id."""
        best = None
        for (origin, tid), c in active.items():
            if origin == self.me or task_kind(tid) != task.kind:
                continue
            if math.hypot(c.x - task.x, c.y - task.y) <= self.p.same_task_radius + 100.0:
                best = max(best or (c.bid, -origin), (c.bid, -origin))
        return best

    # -- bidding --------------------------------------------------------------------------

    def _bid_for_best(self, t, s, tasks, active, d: Decision) -> None:
        best: tuple[float, _Task] | None = None
        for task in tasks.values():
            if self._lost is not None and t - self._lost[0] < self._cooldown \
                    and math.hypot(task.x - self._lost[1], task.y - self._lost[2]) \
                    <= self.p.same_task_radius:
                continue  # just outbid here: the Ghost next door is the same danger
            bid = self._bid(task, s)
            rival = self._rival(task, active)
            if rival is not None and rival >= (bid, -self.me):
                continue  # a nearer drone has it
            if best is None or bid > best[0]:
                best = (bid, task)
        if best is None:
            return
        self._my_bid, task = best
        self._task = _Task(task.tid, task.kind, task.x, task.y, task.track, task.target,
                           claimed=(task.x, task.y))
        self._claimed_at, self._settled, self._arrived_at = t, False, None
        self._reclaimed = False
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
        if self._follow_since is not None:
            self._follow(t, s, tracks, active, d)
            return
        current = tasks.get(task.tid) or self._drifted(task, tasks) or self._held(task, tracks)
        if current is not None and task.claimed is not None and math.hypot(
                current.x - task.claimed[0], current.y - task.claimed[1]) > self.p.same_task_radius:
            self._release(t, d, "drifted")  # walked too far from its Claim: re-auction it
            return
        self._verify_progress(t, s, tracks, current, d)
        if self._task is None:
            return
        if current is not None:
            task.x, task.y, task.track = current.x, current.y, current.track
            task.target = (current.x, current.y)
        self._my_bid = self._bid(task, s)  # it gets nearer as it flies
        rival = self._rival(task, active)
        if rival is not None and rival > (self._my_bid, -self.me):
            self._send_claim(t, d, 0.0)  # a nearer drone has it: release, don't linger
            self._lost = (t, task.x, task.y)
            self._task = None
            self._settled = False
            d.events.append({"t": round(t, 2), "kind": "outbid", "drone": self.me})
            return
        if not self._settled and t - self._claimed_at >= self.p.settle:
            self._settled = True
            d.events.append(self._event(t, "dispatch"))
        if not self._settled and not self._reclaimed and t - self._claimed_at >= self.p.settle / 2:
            self._reclaimed = True  # once more before flying: the first Claim may be lost
            self._send_claim(t, d, LEASE)
        elif t - self._last_claim >= RENEW:
            self._send_claim(t, d, LEASE)

    def _drifted(self, task: _Task, tasks) -> _Task | None:
        same = [x for x in tasks.values() if task_kind(x.tid) == task.kind
                and math.hypot(x.x - task.x, x.y - task.y) <= self.p.same_task_radius]
        return min(same, key=lambda x: math.hypot(x.x - task.x, x.y - task.y), default=None)

    def _held(self, task: _Task, tracks) -> _Task | None:
        """A task once claimed lasts while its Suspicion does, below the strength gate too:
        strengths hover around the gate, and a claim dropped before it settles flies nobody."""
        same = [tr for tr in tracks if tr.cls == self.swarm.threat
                and tr.status == TrackStatus.SUSPECTED
                and tr.confidence >= self.p.verify_min_confidence
                and math.hypot(tr.x - task.x, tr.y - task.y) <= self.p.same_task_radius]
        tr = min(same, key=lambda tr: math.hypot(tr.x - task.x, tr.y - task.y), default=None)
        return None if tr is None else _Task(task.tid, task.kind, tr.x, tr.y, tr, (tr.x, tr.y),
                                             task.claimed)

    def _verify_progress(self, t, s, tracks, current, d: Decision) -> None:
        task = self._task
        confirmed = next((tr for tr in tracks if tr.status == TrackStatus.CONFIRMED
                          and math.hypot(tr.x - task.x, tr.y - task.y) <= CONFIRM_RADIUS), None)
        if confirmed is not None and self._settled:
            if self.p.follow_for > 0:  # stay with it: Follow
                d.events.append(self._event(t, "confirmed")
                                | {"x": round(confirmed.x, 1), "y": round(confirmed.y, 1)})
                self._follow_since, self._unseen_since = t, None
                self._follow(t, s, tracks, {}, d)
                return
            self._release(t, d, "confirmed", confirmed)
            return
        dist = math.hypot(task.target[0] - s.x, task.target[1] - s.y)
        if self._settled and dist <= self.p.arrive and self._arrived_at is None:
            self._arrived_at = t
        if (current is None and self._arrived_at is not None
                and t - self._arrived_at >= self.p.dismiss_after):
            self._release(t, d, "dismissed")
        elif self._arrived_at is not None and t - self._arrived_at >= self.p.verify_timeout:
            # listeners keep a Suspicion alive the camera can neither see nor clear: give it up
            # and leave the place alone a while, as when outbid
            self._lost = (t, task.x, task.y)
            self._release(t, d, "timeout")
        elif current is None and not self._settled:
            self._release(t, d, "obsolete")

    def _follow(self, t, s, tracks, active, d: Decision) -> None:
        """Stay with the confirmed threat: fly to the nearest Confirmed Track of the Threat
        Class within same_task_radius, keep the Claim alive, report once a second. Two
        Verifiers that confirmed the same threat (Claims crossed on the air) do not both
        follow it: the nearer keeps it, as in the auction."""
        task = self._task
        near = [tr for tr in tracks if tr.status == TrackStatus.CONFIRMED
                and tr.cls == self.swarm.threat
                and math.hypot(tr.x - task.x, tr.y - task.y) <= self.p.same_task_radius]
        conf = min(near, key=lambda tr: math.hypot(tr.x - task.x, tr.y - task.y), default=None)
        if conf is not None:
            task.x, task.y, task.track, task.target = conf.x, conf.y, conf, (conf.x, conf.y)
            self._unseen_since = None
        elif self._unseen_since is None:
            self._unseen_since = t
        if self._unseen_since is not None and t - self._unseen_since >= self.p.dismiss_after:
            self._release(t, d, "lost")
            return
        if t - self._follow_since >= self.p.follow_for:
            self._release(t, d, "handed_off")
            return
        self._my_bid = self._bid(task, s)
        rival = self._rival(task, active)
        if rival is not None and rival > (self._my_bid, -self.me):
            self._lost = (t, task.x, task.y)
            self._release(t, d, "outbid")
            return
        if t - self._follow_said >= 1.0:
            self._follow_said = t
            d.events.append(self._event(t, "following"))
        if t - self._last_claim >= RENEW:
            self._send_claim(t, d, LEASE)

    def _release(self, t, d: Decision, why: str, track: TrackEst | None = None) -> None:
        ev = self._event(t, why)
        if track is not None:
            ev.update(x=round(track.x, 1), y=round(track.y, 1))
        d.events.append(ev)
        self._send_claim(t, d, 0.0)
        self._task = None
        self._settled = False
        self._follow_since = None

    def _send_claim(self, t, d: Decision, lease: float) -> None:
        task = self._task
        d.claims.append((task.tid, max(0.0, self._my_bid), lease, task.target[0], task.target[1]))
        self._last_claim = t

    def _event(self, t, kind) -> dict:
        task = self._task
        ev = {"t": round(t, 2), "kind": kind, "drone": self.me, "task": KIND_NAME[task.kind],
              "tid": task.tid, "x": round(task.x, 1), "y": round(task.y, 1)}
        if self._follow_since is not None:
            ev["cls"] = self.swarm.threat.name
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

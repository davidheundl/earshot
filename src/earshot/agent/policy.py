"""Policy: choose this drone's Mode and Command from its own Belief and the Claims it knows.

Interface:
    policy = Policy(me, city, swarm)
    decision = policy.decide(t, state, tracks, belief, claims)   -> Decision

Concept section 7:
- Tasks come from Suspicions (drone-class Tracks that are not Explained or Confirmed):
  LISTEN (two slots per Suspicion: land on a roof whose bearing crosses the existing ones near
  a right angle) while it is poorly localised, VERIFY (fly there, camera on) once it is likely
  and localised. Task ids are kind * 4096 + the Suspicion's 100 m cell, so identical maps give
  identical ids, and every bidder can decode where a task is.
- Allocation without a central node: each drone bids on the best task for itself and
  broadcasts a Claim (task, bid, lease, target point). Highest bid wins, ties to the lower id.
  A drone acts only after holding the best bid for a settle window, renews its lease while
  working, and releases with lease 0. A silent drone's lease lapses and someone else bids.
- Sectors are Voronoi cells of the homes of available drones (no active Claim); when a drone
  leaves, its neighbours' cells grow and patrollers fly Fill Gap.

Baselines for the evaluation (SwarmConfig.policy):
- "nearest": every non-confirmed Track of any class is verified by whoever bids highest
  (that is the nearest drone); no listening tasks, no explaining away.
- "no_perch": nobody ever perches; listening tasks are disabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from earshot.agent.belief import CONFIRM_RADIUS, TrackEst, cell_center
from earshot.city import City
from earshot.messages import Claim, Mode, TargetClass, TrackStatus
from earshot.sensors import acoustics as ac
from earshot.state import Command, DroneState, SwarmConfig

VERIFY, LISTEN_A, LISTEN_B = 0, 1, 2
KIND_NAME = {VERIFY: "verify", LISTEN_A: "listen", LISTEN_B: "listen"}
LEASE = 10.0  # s
RENEW = 4.0
SETTLE = 3.0  # > typical 3-hop Claim latency, so competing bids arrive before anyone flies
LOCALISED = 100.0  # m, semi-major axis below which more bearings add little
VERIFY_MIN_CONFIDENCE = 0.5
TOP_SUSPICIONS = 2  # only the strongest Suspicions spawn tasks; ghosts rank below the truth
CONFIRMED_SHADOW = 200.0  # Suspicions this close to a confirmed drone are its side lobes
VERIFY_MAX_ELLIPSE = 150.0  # semi-major (2 sigma); the strength gate separates truth from ghosts
VERIFY_MIN_STRENGTH = 8.0  # log-odds; crossing bearings reach 10-20, RF-ring overlaps 3-7
SAME_TASK_RADIUS = 200.0  # a Claim of the same kind this close covers a drifted Suspicion
ROOF_SEARCH_RADIUS = 300.0
ARRIVE = 60.0  # m; the Verifier holds a 40 m standoff (backend Kinematics.standoff)
DISMISS_AFTER = 8.0  # s at the target with no Suspicion left -> dismissed
ASSUMED_BACKGROUND_DB = 40.0
INTRUDER_LW = 89.0  # plan for the larger drone class
V_VERIFY = 15.0
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
    target: tuple[float, float]  # roof or verify point
    perch_site: str | None = None


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
        self._p_hear: dict[tuple, float] = {}  # (site, x, y) -> P(hear an intruder there)

    # -- main -----------------------------------------------------------------------------

    def decide(self, t: float, s: DroneState, tracks: list[TrackEst], belief,
               claims: dict[tuple[int, int], Claim]) -> Decision:
        self._belief, self._belief_t = belief, t
        active = {k: c for k, c in claims.items() if c.lease > 0 and t < c.t + c.lease}
        winners = self._winners(active)
        tasks = self._tasks(tracks, belief, active, s)
        d = Decision(command=self._default_command(t, s, active))
        if self.fixed:
            return d  # a microphone array on a roof: it listens, it never flies

        if self._task is not None:
            self._continue(t, s, tasks, winners, tracks, d)
        if self._task is not None and self._task.kind != VERIFY \
                and self._maybe_switch_to_verify(t, s, tasks, winners, active, d):
            tasks = {tid: x for tid, x in tasks.items() if x.kind == VERIFY}  # what it left for
        if self._task is None:
            self._bid_for_best(t, s, tasks, winners, active, d)

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
            if self.swarm.policy != "no_perch" and (tr.semi_major > LOCALISED or tr.origins < 2):
                for kind in (LISTEN_A, LISTEN_B):
                    tid = task_id(kind, tr.track_id)
                    out[tid] = _Task(tid, kind, tr.x, tr.y, tr, (tr.x, tr.y))
            if (tr.confidence >= VERIFY_MIN_CONFIDENCE and tr.semi_major <= VERIFY_MAX_ELLIPSE
                    and tr.strength >= VERIFY_MIN_STRENGTH):
                tid = task_id(VERIFY, tr.track_id)
                out[tid] = _Task(tid, VERIFY, tr.x, tr.y, tr, (tr.x, tr.y))
        return out

    @staticmethod
    def _winners(active) -> dict[int, tuple[float, int]]:
        best: dict[int, tuple[float, int]] = {}
        for (origin, tid), c in active.items():
            cur = best.get(tid)
            if cur is None or (c.bid, -origin) > (cur[0], -cur[1]):
                best[tid] = (c.bid, origin)
        return best

    def _covered(self, task: _Task, active) -> bool:
        """Someone else holds a Claim of the same kind for this Suspicion under an older id
        (the Suspicion drifted into another 100 m cell). Same id: the auction decides."""
        for (origin, tid), c in active.items():
            if origin == self.me or tid == task.tid or task_kind(tid) != task.kind:
                continue
            if math.hypot(c.x - task.x, c.y - task.y) <= SAME_TASK_RADIUS + 100.0:
                return True
        return False

    # -- bidding --------------------------------------------------------------------------

    def _bid_for_best(self, t, s, tasks, winners, active, d: Decision) -> None:
        best: tuple[float, _Task] | None = None
        for task in tasks.values():
            if self._covered(task, active):
                continue
            bid, target, site = self._bid(task, s, active)
            if bid <= 0:
                continue
            w = winners.get(task.tid)
            if w is not None and w[1] != self.me and (w[0], -w[1]) >= (bid, -self.me):
                continue
            task.target, task.perch_site = target, site
            if best is None or bid > best[0]:
                best = (bid, task)
        if best is None:
            return
        self._my_bid, self._task = best
        self._claimed_at, self._settled, self._arrived_at = t, False, None
        self._send_claim(t, d, LEASE)
        d.events.append(self._event(t, "claim"))

    def _maybe_switch_to_verify(self, t, s, tasks, winners, active, d: Decision) -> bool:
        """Verification outranks listening: a listener that would win a verify task drops its
        roof for it (otherwise a swarm that is all listening never verifies). True if it did;
        the caller then bids on verify tasks only, or it would take its roof straight back."""
        for task in tasks.values():
            if task.kind != VERIFY or self._covered(task, active):
                continue
            bid, _, _ = self._bid(task, s, active)
            w = winners.get(task.tid)
            if bid > 0 and (w is None or (bid, -self.me) > (w[0], -w[1])):
                self._release(t, d, "switch")
                return True
        return False

    def _bid(self, task: _Task, s: DroneState, active) -> tuple[float, tuple, str | None]:
        if task.kind == VERIFY:
            tr = task.track
            dist = math.hypot(task.x - s.x, task.y - s.y)
            value = tr.confidence if tr else 0.5
            cost = dist / V_VERIFY / 120.0  # minutes of flight, scaled
            if self.default == Mode.LISTEN and self.swarm.policy == "earshot":
                cost += 0.25  # leaving a listening post opens a Gap
            if s.perched_on is not None and self._hears(s, task) and self.swarm.policy == "earshot":
                cost += 0.3  # it is one of the bearings localising this Suspicion
            return 10.0 * (value - cost), (task.x, task.y), None
        if s.perched_on is not None and self._hears(s, task):
            return 0.0, (0, 0), None  # already listening to it
        taken = [(c.x, c.y) for (o, _), c in active.items() if o != self.me]
        site, score = self._best_listening_roof(s, task, taken)
        if site is None:
            return 0.0, (0, 0), None
        return 10.0 * score, (site.x, site.y), site.id

    def _hears(self, s: DroneState, task: _Task) -> bool:
        return math.hypot(task.x - s.x, task.y - s.y) <= 250.0

    def _best_listening_roof(self, s, task: _Task, taken) -> tuple:
        listeners = self._listeners(task)
        best, best_score = None, 0.0
        if len(self._p_hear) > 4096:
            self._p_hear.clear()
        for site in self._sites.values():
            d = math.hypot(site.x - task.x, site.y - task.y)
            if d > ROOF_SEARCH_RADIUS:
                continue
            if any(math.hypot(site.x - x, site.y - y) < 20 for x, y in taken + listeners):
                continue
            key = (site.id, task.x, task.y)  # a Track moves at most once per second
            if key not in self._p_hear:
                nb = self.city.obstacles_between((site.x, site.y), (task.x, task.y), site.z + 0.5,
                                                 float(self.city.ground(task.x, task.y)) + 30.0)
                lp = ac.received_level(INTRUDER_LW, TargetClass.DRONE, d, nb)
                self._p_hear[key] = ac.detection_probability(lp - ASSUMED_BACKGROUND_DB,
                                                             TargetClass.DRONE)
            p_hear = self._p_hear[key]
            ang = math.atan2(task.x - site.x, task.y - site.y)
            q = min((abs(math.sin(ang - math.atan2(task.x - lx, task.y - ly)))
                     for lx, ly in listeners), default=1.0)
            travel = math.hypot(site.x - s.x, site.y - s.y)
            score = q * p_hear - travel / 3000.0
            if score > best_score:
                best, best_score = site, score
        return best, best_score

    def _listeners(self, task: _Task) -> list[tuple[float, float]]:
        """Positions of perched drones whose latest bearing points at this Suspicion."""
        out = []
        belief = self._belief
        for o in belief.last_sound.values():
            if o.mode != Mode.LISTEN or self._belief_t - o.t > 10.0:
                continue
            ang = math.degrees(math.atan2(task.x - o.x, task.y - o.y)) % 360.0
            if abs((ang - o.bearing + 180) % 360 - 180) <= max(15.0, 2 * o.bearing_sigma):
                out.append((o.x, o.y))
        return out

    # -- holding a task -------------------------------------------------------------------

    def _continue(self, t, s, tasks, winners, tracks, d: Decision) -> None:
        task = self._task
        current = tasks.get(task.tid) or self._drifted(task, tasks)
        if task.kind == VERIFY:
            self._verify_progress(t, s, tracks, current, d)
            if self._task is None:
                return
        elif current is None and not self._suspicion_near(task, tracks):
            self._release(t, d, "obsolete")  # Suspicion confirmed or gone: back to default
            return
        if current is not None:
            task.x, task.y, task.track = current.x, current.y, current.track
            if task.kind == VERIFY:
                task.target = (current.x, current.y)
        w = winners.get(task.tid)
        if w is not None and w[1] != self.me and (w[0], -w[1]) > (self._my_bid, -self.me):
            self._task = None  # outbid: let my lease lapse
            d.events.append({"t": round(t, 2), "kind": "outbid", "drone": self.me})
            return
        if not self._settled and t - self._claimed_at >= SETTLE:
            self._settled = True
            d.events.append(self._event(t, "dispatch"))
        if t - self._last_claim >= RENEW:
            self._send_claim(t, d, LEASE)

    @staticmethod
    def _suspicion_near(task: _Task, tracks) -> bool:
        return any(tr.cls == TargetClass.DRONE and tr.status == TrackStatus.SUSPECTED
                   and math.hypot(tr.x - task.x, tr.y - task.y) <= SAME_TASK_RADIUS
                   for tr in tracks)

    def _drifted(self, task: _Task, tasks) -> _Task | None:
        same = [x for x in tasks.values() if task_kind(x.tid) == task.kind
                and math.hypot(x.x - task.x, x.y - task.y) <= SAME_TASK_RADIUS]
        return min(same, key=lambda x: math.hypot(x.x - task.x, x.y - task.y), default=None)

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
        task = self._task
        if task.kind == VERIFY:
            near = math.hypot(task.target[0] - s.x, task.target[1] - s.y) <= 150.0
            return Command(Mode.VERIFY, goal=task.target, camera_on=near, look_at=task.target)
        return Command(Mode.LISTEN, perch_site=task.perch_site)

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

    _belief = None  # read-only view for _listeners, set at the start of decide()
    _belief_t = 0.0


def decode_task(tid: int, bounds) -> tuple[str, float, float]:
    x, y = cell_center(tid % TASK_STRIDE, bounds)
    return KIND_NAME[task_kind(tid)], x, y

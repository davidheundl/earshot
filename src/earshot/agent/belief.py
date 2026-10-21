"""Belief: one drone's Probability Map and the Tracks extracted from it.

Interface:
    belief = Belief(city, cell=20.0, threat=TargetClass.DRONE)
    belief.add_evidence(obs, t_now)   Observation from anyone (own or received), each exactly once
    belief.set_track(track)           a received Track; replaces that origin's older one there
    belief.advance(t_now)             decay to t_now (call before reading)
    belief.posterior()                (4, ny, nx) class probabilities, indexed by TargetClass
    belief.tracks(t_now)              fused Suspicions as Track-like records (TrackEst)
    belief.faulty                     origins currently believed to have a Faulty Sensor
    belief.summary()                  small JSON for the dashboard

Model (concept section 6):
- Per cell and threat class k (drone, vehicle, train) a log-odds against "nothing here":
      logit_k = prior_k(cell) + sum_(o,f) G_o,f,k(cell) * min(1, N_EFF * D_o,f / W_o,f(cell)) + C_k(cell)
  G_o,f is the summed log-likelihood ratio from origin o (one Drone) through sensor family f
  (sound, RF) and W_o,f the summed weight (how many readings covered the cell). Repeated
  readings from one static sensor have correlated errors, so each contribution counts as at
  most N_EFF independent readings: beyond that it is their average, which keeps the wedge's
  shape (a capped sum would flatten it). G and W stay plain sums, so arrival order never
  matters (concept 8.4). D_o,f = exp(-(t - newest reading of o,f) / TAU): G and W decay
  alike, so without it a capped average would never fade once its sensor falls silent.
  C holds camera evidence, strong but capped the same way per camera (at most N_EFF frames'
  worth, fading with TAU_CAMERA from its newest frame): a Verifier hovering 20 s must not
  bury a spot for ten minutes, nor saturate the probability into a flat plateau.
- Evidence is a bounded log-likelihood ratio. A sound bearing raises cells along its wedge and
  mildly lowers the other cells within hearing range (mildly: the reading may come from a
  different source or be an outlier). An RF reading raises the ring at the distance its band
  implies and lowers cells inside and beyond it. A camera detection adds a blob, a camera
  "nothing here" removes one within the radius it could check (the reading's level when > 0,
  else CAMERA_RANGE), and drops a Confirmed point there that the camera no longer sees.
- Everything decays with time constant TAU (moving sources); late evidence enters already
  decayed by its own age.
- Priors come from the City: rail lines make trains likely, roads make vehicles likely. That
  is what explains rumble away (concept 6, "explaining away").
- The Threat Class (`threat`: drone at the airport, vehicle in a Reserve) is what a camera
  confirms, what a confirmed point follows and what merges: a Suspicion is a Track of that
  class. Off the places the City explains it, its prior is PRIOR_THREAT, the drone's.
- An RF band's ring goes to the class its emitters are (sensors.rf.Band.emitter).
- Faulty Sensor: an origin whose bearings keep missing what the others' map shows is excluded
  (step: `_check_faults`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import combinations, pairwise

import numpy as np

from earshot.city import City
from earshot.messages import RF_SENSORS, Mode, Observation, Sensor, TargetClass, Track, TrackStatus
from earshot.sensors import acoustics as ac
from earshot.sensors import rf

THREATS = (TargetClass.DRONE, TargetClass.VEHICLE, TargetClass.TRAIN)
TAU = 30.0  # s, evidence half-life ~21 s
TAU_CAMERA = 120.0
N_EFF = 1.5  # independent readings one static sensor is worth, however many it sends
SOUND_OFF_LINE = 0.6  # likelihood off the wedge relative to uniform: "maybe another source"
RF_ON_RING, RF_OFF_RING = 2.2, -0.7  # log-likelihood ratio on / away from the ring
A_CAMERA = 6.0
A_CAMERA_NONE = -8.0
RF_SIGMA_DB = 7.0  # shadowing + unknown line of sight
CAMERA_RANGE = 100.0  # wide-angle camera from 40-60 m altitude
ASSUMED_BACKGROUND_DB = 40.0  # drones assume night; the gate only says "could not have heard"

PRIOR_THREAT = -5.0  # the Threat Class wherever the City does not explain it
PRIOR_DRONE = PRIOR_THREAT
PRIOR_VEHICLE_ROAD, PRIOR_VEHICLE_ELSEWHERE, ROAD_WIDTH = -2.0, -8.0, 25.0
PRIOR_TRAIN_RAIL, PRIOR_TRAIN_ELSEWHERE, RAIL_WIDTH = -2.0, -10.0, 30.0

TRACK_THRESHOLD = 0.3  # threat probability at the peak; one capped sensor alone reaches 0.27
TRACK_RADIUS = 150.0
TRACK_SUPPRESS = 200.0
MAX_TRACKS = 6
EXPLAIN_DISTANCE = 60.0  # a vehicle/train blob this close to its road/rail is Explained
CONFIRM_RADIUS = 80.0  # a Threat Class Track this close to a camera confirmation is Confirmed
CONFIRM_FOLLOW = 150.0  # m a confirmed threat may move between Track updates and stay Confirmed
CONFIRM_SPEED = 15.0  # m/s a confirmed point moves at most: it follows the threat, not ghosts
CONFIRM_GRACE = 2.0  # s without a sighting before a camera "nothing" drops a confirmed point
CONFIRM_HOLD = 120.0  # s a confirmed point lives after the camera last saw it
TRACK_SAME = 60.0  # m; a Track this close with the same class is "the same" (and replaces it)
RX_MAX_AGE = 15.0  # s a received Track counts
FAULT_WINDOW = 30.0  # s of an origin's bearings that are checked
FAULT_MIN_CHECKS = 8  # bearings needed in the window before judging
FAULT_STRONG = 8.0  # threat log-odds the others reach on their own; ghosts stay at 3-7
FAULT_LOCALISED = 100.0  # m; only judge against tight Tracks (ghosts are ~140 m wide)
FAULT_LAG_M = 30.0  # m a moving source is ahead of the fused Track
FAULT_AGREE_SHARE = 0.1  # share of bearings pointing at a strong Track that counts as hearing it
FAULT_PERSIST = 5  # consecutive failed seconds before flagging


def _hearing_range(cls: TargetClass, lw: float, perched: bool) -> float:
    """Distance beyond which a source of this class is heard in < 5 % of windows."""
    ego = -math.inf if perched else ac.ego_noise(ac.EGO_RPM_REF, True, 8.0)
    noise = ac.db_sum(ASSUMED_BACKGROUND_DB, ego)
    lo, hi = 1.0, 20000.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        p = ac.detection_probability(ac.received_level(lw, cls, mid, 0) - noise, cls)
        lo, hi = (mid, hi) if p > 0.05 else (lo, mid)
    return lo


_LOUDEST = {TargetClass.DRONE: 89.0, TargetClass.VEHICLE: 108.0, TargetClass.TRAIN: 120.0}
HEARING_RANGE = {
    (k, perched): _hearing_range(k, _LOUDEST[k], perched) for k in THREATS for perched in (True, False)
}


@dataclass(frozen=True)
class TrackEst:
    """A fused Suspicion from this drone's own map."""

    track_id: int  # 100 m cell index of the position: identical maps -> identical ids
    x: float
    y: float
    semi_major: float
    semi_minor: float
    angle: float  # degrees, major axis from east, counter-clockwise
    cls: TargetClass
    confidence: float
    status: TrackStatus
    origins: int  # how many drones' evidence supports it
    strength: float = 0.0  # threat log-odds at the peak: ranks Suspicions (P saturates)
    support: tuple = ()  # ((origin, family, log-odds), ...) at the peak, for the class: who built it


def cell_id(x: float, y: float, bounds) -> int:
    """100 m cell index over the City bounds; also the basis of task ids (policy.TASK_STRIDE)."""
    x0, y0, x1, _ = bounds
    nx = math.ceil((x1 - x0) / 100.0)
    return int((y - y0) // 100) * nx + int((x - x0) // 100)


def cell_center(cid: int, bounds) -> tuple[float, float]:
    x0, y0, x1, _ = bounds
    nx = math.ceil((x1 - x0) / 100.0)
    return x0 + (cid % nx) * 100.0 + 50.0, y0 + (cid // nx) * 100.0 + 50.0


def merge_overlapping(tracks: list[TrackEst], bounds,
                      threat: TargetClass = TargetClass.DRONE) -> list[TrackEst]:
    """Overlapping Suspicions (Tracks of the Threat Class) are one danger seen several times (Ghosts beside the
    truth, neighbours' reports of it): they become one Suspicion at the likelihood-weighted
    mean of their positions (weight exp(log-odds)), the most likely place to find it: zones
    of similar strength average evenly, a clearly weaker Ghost barely pulls (the peak
    probability saturates near 1 for the truth and Ghosts alike, so it cannot rank them).
    The merged ellipse is the strongest member's, moved to the mean. A Confirmed Track merges
    only with Confirmed ones: a Suspicion beside it may be a second drone, and the policy's
    confirmed_shadow decides whether it is a side lobe. Other Tracks pass unchanged; a merged
    one takes its first member's place."""
    idx = [i for i, tr in enumerate(tracks) if mergeable(tr, threat)]
    root = {i: i for i in idx}

    def find(i: int) -> int:
        while root[i] != i:
            i = root[i]
        return i

    for a, b in combinations(idx, 2):
        if _confirmed(tracks[a]) == _confirmed(tracks[b]) and _overlap(tracks[a], tracks[b]):
            ra, rb = find(a), find(b)
            root[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[TrackEst]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(tracks[i])
    out = []
    for i, tr in enumerate(tracks):
        if i not in root:
            out.append(tr)
        elif i in groups:  # the group's first member
            out.append(_merged(groups[i], bounds))
    return out


def mergeable(tr: TrackEst, threat: TargetClass = TargetClass.DRONE) -> bool:
    return tr.cls == threat and tr.status != TrackStatus.EXPLAINED


def _toward(x: float, y: float, gx: float, gy: float, t0: float, t1: float):
    """(x, y) moved toward (gx, gy) by at most CONFIRM_SPEED over t1 - t0 (at least 1 s)."""
    d = math.hypot(gx - x, gy - y)
    step = min(d, CONFIRM_SPEED * max(1.0, t1 - t0))
    return (x, y) if d == 0 else (x + (gx - x) * step / d, y + (gy - y) * step / d)


def _confirmed(tr: TrackEst) -> bool:
    return tr.status == TrackStatus.CONFIRMED


def _reach(tr: TrackEst, theta: float) -> float:
    """Distance from the ellipse's centre to its edge in direction theta (radians from east)."""
    a, b = max(tr.semi_major, 1.0), max(tr.semi_minor, 1.0)
    phi = theta - math.radians(tr.angle)
    return a * b / math.hypot(b * math.cos(phi), a * math.sin(phi))


def _overlap(p: TrackEst, q: TrackEst) -> bool:
    d = math.hypot(q.x - p.x, q.y - p.y)
    # an ellipse reaches no further than its longer semi-axis: far pairs skip the trig
    if d > 1.001 * (max(p.semi_major, p.semi_minor, 1.0) + max(q.semi_major, q.semi_minor, 1.0)):
        return False
    theta = math.atan2(q.y - p.y, q.x - p.x)
    return d <= _reach(p, theta) + _reach(q, theta)


def _merged(group: list[TrackEst], bounds) -> TrackEst:
    if len(group) == 1:
        return group[0]
    top = max(tr.strength for tr in group)
    w = [math.exp(tr.strength - top) for tr in group]  # likelihood, relative to the strongest
    x = sum(wi * tr.x for wi, tr in zip(w, group, strict=True)) / sum(w)
    y = sum(wi * tr.y for wi, tr in zip(w, group, strict=True)) / sum(w)
    best = max(group, key=lambda tr: tr.strength)
    return replace(best, track_id=cell_id(x, y, bounds), x=x, y=y,
                   confidence=max(tr.confidence for tr in group),
                   origins=max(tr.origins for tr in group))


_PRIORS: tuple[City, float, TargetClass, np.ndarray] | None = None  # the last (City, cell, threat)'s


class Belief:
    def __init__(self, city: City, cell: float = 20.0, threat: TargetClass = TargetClass.DRONE):
        if threat not in THREATS:
            raise ValueError(f"the Threat Class must be one of {[c.name for c in THREATS]}")
        self.city = city
        self.cell = cell
        self.threat = threat
        xs, ys = city.grid(cell)
        self.X, self.Y = np.meshgrid(xs, ys)
        self.shape = self.X.shape
        self.prior = self._priors()
        self._G: dict[tuple[int, str], np.ndarray] = {}  # (origin, family) -> (3, ny, nx) LLR
        self._W: dict[tuple[int, str], np.ndarray] = {}  # (origin, family) -> (ny, nx) weight
        self._newest: dict[tuple[int, str], float] = {}  # (origin, family) -> newest reading t
        self._CG: dict[int, np.ndarray] = {}  # camera origin -> (3, ny, nx) LLR
        self._CW: dict[int, np.ndarray] = {}  # camera origin -> (ny, nx) frames that looked here
        self._c_newest: dict[int, float] = {}  # camera origin -> newest frame t
        self._cam: tuple[tuple, np.ndarray] | None = None  # (stamp, capped camera evidence)
        self._t = 0.0
        self._tracks_rx: dict[int, list[Track]] = {}  # origin -> its current Tracks
        # (x, y, t_seen, t_moved): last camera sighting (the hold) and last move (the speed cap)
        self._confirmed: list[tuple[float, float, float, float]] = []
        self.faulty: set[int] = set()
        self.last_sound: dict[int, Observation] = {}  # origin -> newest sound bearing
        self._fused: list[tuple[float, int, str]] = []  # (t, origin, family) for the dashboard
        self._recent_sound: dict[int, list[Observation]] = {}  # origin -> bearings in window
        self._fault_streak: dict[int, int] = {}
        self._track_cache: list[TrackEst] = []  # last extracted Tracks, for check_faults
        self._fault_stats: dict[int, list[int]] = {}  # origin -> [checked, unconfirmed]
        self._post: tuple[tuple, np.ndarray] | None = None  # (stamp, posterior) cache
        self.n_evidence = 0
        self._summary: tuple[tuple, dict] | None = None

    # -- priors ---------------------------------------------------------------------------

    def _priors(self) -> np.ndarray:
        """Read-only; the drones of one process share them (lockstep runs ten Beliefs)."""
        global _PRIORS
        if (_PRIORS is not None and _PRIORS[0] is self.city and _PRIORS[1] == self.cell
                and _PRIORS[2] == self.threat):
            return _PRIORS[3]

        def elsewhere(cls: TargetClass, p: float) -> float:
            return max(p, PRIOR_THREAT) if cls == self.threat else p

        prior = np.empty((3, *self.shape))
        prior[0] = PRIOR_DRONE
        d_road = self._distance_to(self.city.roads)
        d_rail = self._distance_to(self.city.rails)
        prior[1] = np.where(d_road <= ROAD_WIDTH, PRIOR_VEHICLE_ROAD,
                            elsewhere(TargetClass.VEHICLE, PRIOR_VEHICLE_ELSEWHERE))
        prior[2] = np.where(d_rail <= RAIL_WIDTH, PRIOR_TRAIN_RAIL,
                            elsewhere(TargetClass.TRAIN, PRIOR_TRAIN_ELSEWHERE))
        prior.flags.writeable = False
        _PRIORS = (self.city, self.cell, self.threat, prior)
        return prior

    def _distance_to(self, lines) -> np.ndarray:
        d = np.full(self.shape, np.inf)
        for line in lines:
            pts = np.asarray(line.coords)
            for (ax, ay), (bx, by) in pairwise(pts):
                vx, vy = bx - ax, by - ay
                L2 = vx * vx + vy * vy or 1.0
                u = np.clip(((self.X - ax) * vx + (self.Y - ay) * vy) / L2, 0, 1)
                d = np.minimum(d, np.hypot(self.X - (ax + u * vx), self.Y - (ay + u * vy)))
        return d

    # -- evidence -------------------------------------------------------------------------

    def advance(self, t_now: float) -> None:
        dt = t_now - self._t
        if dt <= 0:
            return
        f = math.exp(-dt / TAU)
        for g in self._G.values():
            g *= f
        for wgt in self._W.values():
            wgt *= f
        fc = math.exp(-dt / TAU_CAMERA)
        for g in self._CG.values():
            g *= fc
        for wgt in self._CW.values():
            wgt *= fc
        self._t = t_now

    def add_evidence(self, obs: Observation, t_now: float | None = None) -> None:
        """Fold one Observation into the map. Caller guarantees each (origin, seq) once."""
        t_now = self._t if t_now is None else t_now
        self.advance(t_now)
        age = max(0.0, self._t - obs.t)
        self.n_evidence += 1
        fam = "camera" if obs.sensor == Sensor.CAMERA else "sound" if obs.sensor == Sensor.SOUND else "rf"
        self._fused.append((obs.t, obs.header.origin, fam))
        if obs.sensor == Sensor.CAMERA:
            w = math.exp(-age / TAU_CAMERA)
            self._camera(obs, w)
            return
        w = math.exp(-age / TAU)
        family = "sound" if obs.sensor == Sensor.SOUND else "rf"
        key = (obs.header.origin, family)
        g = self._G.setdefault(key, np.zeros((3, *self.shape)))
        wgt = self._W.setdefault(key, np.zeros(self.shape))
        self._newest[key] = max(self._newest.get(key, obs.t), obs.t)
        if obs.sensor == Sensor.SOUND:
            self._sound(obs, g, wgt, w)
            last = self.last_sound.get(obs.header.origin)
            if last is None or obs.t >= last.t:
                self.last_sound[obs.header.origin] = obs
            self._recent_sound.setdefault(obs.header.origin, []).append(obs)
        elif obs.sensor in RF_SENSORS:
            self._rf(obs, g, wgt, w)

    def _geometry(self, obs: Observation):
        dx, dy = self.X - obs.x, self.Y - obs.y
        return np.hypot(dx, dy), np.degrees(np.arctan2(dx, dy)) % 360.0

    def _sound(self, obs: Observation, g: np.ndarray, wgt: np.ndarray, w: float) -> None:
        d, theta = self._geometry(obs)
        delta = (theta - obs.bearing + 180.0) % 360.0 - 180.0
        sigma = max(obs.bearing_sigma, 2.0)
        # bearing density relative to uniform: von-Mises-like wedge mixed with "other source"
        peak = 360.0 / (sigma * math.sqrt(2 * math.pi))
        dens = (1 - SOUND_OFF_LINE) * peak * np.exp(-0.5 * (delta / sigma) ** 2) + SOUND_OFF_LINE
        llr = np.log(dens)
        perched = obs.mode == Mode.LISTEN
        gate_any = np.zeros(self.shape, dtype=bool)
        for i, k in enumerate(THREATS):
            gate = d <= HEARING_RANGE[(k, perched)]
            gate_any |= gate
            g[i] += w * obs.class_probs[k] * llr * gate
        wgt += w * gate_any

    def _rf(self, obs: Observation, g: np.ndarray, wgt: np.ndarray, w: float) -> None:
        band = rf.BAND_OF_SENSOR[obs.sensor]
        d, _ = self._geometry(obs)
        at_1m = rf.mean_rx_power(band, 1.0, False)
        logd = np.log10(np.maximum(d, 1))
        # line of sight unknown: accept either path-loss exponent (two concentric rings)
        ring = np.maximum(
            np.exp(-0.5 * ((obs.level - (at_1m - 10 * rf.EXP_NLOS * logd)) / RF_SIGMA_DB) ** 2),
            np.exp(-0.5 * ((obs.level - (at_1m - 10 * rf.EXP_LOS * logd)) / RF_SIGMA_DB) ** 2),
        )
        llr = (RF_ON_RING - RF_OFF_RING) * ring + RF_OFF_RING
        # the drone links come from drones, handheld radios from people in vehicles
        g[THREATS.index(band.emitter)] += w * obs.class_probs[band.emitter] * llr
        wgt += w

    def _camera(self, obs: Observation, w: float) -> None:
        d, theta = self._geometry(obs)
        o = obs.header.origin
        g = self._CG.setdefault(o, np.zeros((3, *self.shape)))
        wgt = self._CW.setdefault(o, np.zeros(self.shape))
        self._c_newest[o] = max(self._c_newest.get(o, obs.t), obs.t)
        nothing = obs.class_probs[TargetClass.NOTHING] >= 0.5 or math.isnan(obs.bearing)
        if nothing:
            # a detector may clear less than the full field of view: it says how far (level)
            radius = obs.level if obs.level > 0 else CAMERA_RANGE
            disc = d <= radius
            g += w * A_CAMERA_NONE * disc  # looked here: no threat within range
            wgt += w * disc
            self._unconfirm(obs.x, obs.y, obs.t, radius)
            return
        disc = d <= CAMERA_RANGE
        delta = (theta - obs.bearing + 180.0) % 360.0 - 180.0
        blob = np.exp(-0.5 * (delta / 15.0) ** 2) * disc
        for i, k in enumerate(THREATS):
            g[i] += w * A_CAMERA * obs.class_probs[k] * blob
        wgt += w * disc
        if obs.class_probs[self.threat] >= 0.5:
            self._confirm(obs.x, obs.y, obs.t)

    def set_track(self, track: Track) -> None:
        """A received Track replaces the same origin's older one in its place (same id, or
        same class within TRACK_SAME): a moving Track leaves no stale copies behind."""
        mine = self._tracks_rx.setdefault(track.header.origin, [])
        same = [tr for tr in mine if tr.track_id == track.track_id
                or (tr.cls == track.cls
                    and math.hypot(tr.x - track.x, tr.y - track.y) <= TRACK_SAME)]
        if any(tr.t > track.t for tr in same):
            return  # an older copy (relayed or healed late): the newer one stands
        mine[:] = [tr for tr in mine if not any(tr is x for x in same)] + [track]
        if track.status == TrackStatus.CONFIRMED:
            self._confirm(track.x, track.y, track.t, seen=False)

    def _confirm(self, x: float, y: float, t: float, seen: bool = True) -> None:
        """Remember a camera-confirmed threat. A report near a known one moves it along (it
        moves on); it never seeds a second point, so 'confirmed' cannot spread. A
        neighbour's Confirmed Track (not `seen` by a camera) only moves an existing point, no
        faster than CONFIRM_SPEED (its Track is a map peak, which can sit on a Ghost), and
        never extends its hold: only a camera frame adds a point or keeps it alive."""
        for i, (cx, cy, seen_t, moved_t) in enumerate(self._confirmed):
            if math.hypot(cx - x, cy - y) <= CONFIRM_FOLLOW:
                if seen:
                    self._confirmed[i] = (x, y, t, t) if t >= moved_t \
                        else (cx, cy, max(seen_t, t), moved_t)
                elif t >= moved_t:
                    self._confirmed[i] = (*_toward(cx, cy, x, y, moved_t, t), seen_t, t)
                return
        if seen:
            self._confirmed.append((x, y, t, t))

    def _unconfirm(self, x: float, y: float, t: float, radius: float) -> None:
        """The camera looked within `radius` of a confirmed point and saw nothing: unless it
        saw the threat within CONFIRM_GRACE (one missed frame), it is no longer there."""
        self._confirmed = [c for c in self._confirmed
                           if math.hypot(c[0] - x, c[1] - y) > radius
                           or t - c[2] < CONFIRM_GRACE]

    # -- reading --------------------------------------------------------------------------

    def _contributions(self) -> dict[tuple[int, str], np.ndarray]:
        """Each trusted (origin, family)'s capped evidence, in insertion order."""
        return {key: self._contribution(key, g) for key, g in self._G.items()
                if key[0] not in self.faulty}

    def _logits(self, contribs: dict | None = None) -> np.ndarray:
        total = self.prior + self._camera_evidence()
        for c in (self._contributions() if contribs is None else contribs).values():
            total = total + c
        return total

    def _contribution(self, key, g: np.ndarray) -> np.ndarray:
        """At most N_EFF readings' worth: the sum while few, the average times N_EFF after
        (fading with TAU from the newest reading)."""
        return g * np.minimum(1.0, self._cap(key) / np.maximum(self._W[key], 1e-9))

    def _cap(self, key) -> float:
        return N_EFF * math.exp(-max(0.0, self._t - self._newest[key]) / TAU)

    def _camera_cap(self, o: int) -> float:
        return N_EFF * math.exp(-max(0.0, self._t - self._c_newest[o]) / TAU_CAMERA)

    def _camera_evidence(self) -> np.ndarray:
        """(3, ny, nx): each camera's frames, at most N_EFF frames' worth (as _contribution)."""
        stamp = self._stamp()
        if self._cam is None or self._cam[0] != stamp:
            total = np.zeros((3, *self.shape))
            for o, g in self._CG.items():
                total += g * np.minimum(1.0, self._camera_cap(o) / np.maximum(self._CW[o], 1e-9))
            self._cam = (stamp, total)
        return self._cam[1]

    def _threat_logodds_at(self, iy: int, ix: int, exclude: int) -> float:
        """Threat log-odds of one cell without `exclude`'s evidence (same sums as `_logits`)."""
        total = self.prior[:, iy, ix] + self._camera_evidence()[:, iy, ix]
        for key, g in self._G.items():
            if key[0] in self.faulty or key[0] == exclude:
                continue
            total = total + g[:, iy, ix] * min(1.0, self._cap(key) / max(self._W[key][iy, ix], 1e-9))
        return float(np.log(np.exp(total).sum()))

    def _stamp(self) -> tuple:
        """Changes whenever the map can change: decay, new evidence, a Faulty Sensor."""
        return self._t, self.n_evidence, tuple(sorted(self.faulty))

    def posterior(self) -> np.ndarray:
        """(4, ny, nx): P(drone), P(vehicle), P(train), P(nothing) per cell. Read-only."""
        stamp = self._stamp()
        if self._post is None or self._post[0] != stamp:
            self._post = (stamp, self._posterior_of(self._logits()))
        return self._post[1]

    @staticmethod
    def _posterior_of(z: np.ndarray) -> np.ndarray:
        m = np.maximum(z.max(axis=0), 0.0)
        e = np.exp(z - m)
        nothing = np.exp(-m)
        denom = e.sum(axis=0) + nothing
        return np.concatenate([e / denom, (nothing / denom)[None]], axis=0)

    def confirmed_near(self, x: float, y: float, t_now: float, radius=None,
                       hold=CONFIRM_HOLD) -> bool:
        """Is a confirmed point within `radius`? Points the camera has not seen for `hold`
        seconds expire first."""
        radius = CONFIRM_RADIUS if radius is None else radius
        self._confirmed = [c for c in self._confirmed if t_now - c[2] < hold]
        return any(math.hypot(c[0] - x, c[1] - y) <= radius for c in self._confirmed)

    # -- Faulty Sensor --------------------------------------------------------------------

    def check_faults(self, t_now: float) -> set[int]:
        """Judge each origin against what the *other* drones strongly agree on.

        For every Track the others support on their own (threat log-odds >= FAULT_STRONG
        without this origin) and that lies within this origin's hearing range, some of its
        recent bearings should point at it. If for FAULT_PERSIST consecutive seconds none
        does, it is a Faulty Sensor and its evidence is excluded until a bearing agrees
        again. An origin that hears extra things (a fan next to its roof) is fine as long as
        it also hears what everyone hears. Returns the origins newly flagged."""
        newly = set()
        for o, hist in self._recent_sound.items():
            hist[:] = [h for h in hist if t_now - h.t <= FAULT_WINDOW]
            if len(hist) < FAULT_MIN_CHECKS:
                continue
            last = hist[-1]
            perched = last.mode == Mode.LISTEN
            verdicts = []
            for tr in self._track_cache:
                d = math.hypot(tr.x - last.x, tr.y - last.y)
                if d < 30.0 or d > HEARING_RANGE[(tr.cls, perched)]:
                    continue
                if tr.semi_major > FAULT_LOCALISED:
                    continue
                if not self._should_hear(last, tr, d):
                    continue  # behind buildings or near the edge: silence proves nothing
                if self._threat_logodds_at(*self._cell(tr.x, tr.y), exclude=o) < FAULT_STRONG:
                    continue
                expected = math.degrees(math.atan2(tr.x - last.x, tr.y - last.y)) % 360.0
                # the Track's own uncertainty plus motion lag, seen from this distance
                slack = math.degrees(math.atan2(tr.semi_major + FAULT_LAG_M, d))
                agree = sum(1 for h in hist
                            if abs((h.bearing - expected + 180) % 360 - 180)
                            <= 2 * h.bearing_sigma + slack)
                verdicts.append(agree / len(hist) >= FAULT_AGREE_SHARE)
            if not verdicts:
                continue
            if any(verdicts):
                self._fault_streak[o] = 0
                self.faulty.discard(o)
                continue
            self._fault_streak[o] = self._fault_streak.get(o, 0) + 1
            if self._fault_streak[o] >= FAULT_PERSIST and o not in self.faulty:
                self.faulty.add(o)
                newly.add(o)
        return newly

    def _should_hear(self, at: Observation, tr: TrackEst, d: float) -> bool:
        """A source of the Track's class (planning level) would be heard more often than not."""
        if at.mode != Mode.LISTEN:
            return False  # flying drones are nearly deaf; never judge them on silence
        nb = self.city.obstacles_between((at.x, at.y), (tr.x, tr.y),
                                         float(self.city.ground(at.x, at.y)) + 12.0,
                                         float(self.city.ground(tr.x, tr.y)) + 20.0)
        lw = _LOUDEST[tr.cls]
        lp = ac.received_level(lw, tr.cls, d, nb)
        return ac.detection_probability(lp - ASSUMED_BACKGROUND_DB, tr.cls) >= 0.5

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        x0, y0 = self.X[0, 0], self.Y[0, 0]
        ix = int(np.clip(round((x - x0) / self.cell), 0, self.shape[1] - 1))
        iy = int(np.clip(round((y - y0) / self.cell), 0, self.shape[0] - 1))
        return iy, ix

    def tracks(self, t_now: float) -> list[TrackEst]:
        self.advance(t_now)
        contribs = self._contributions()
        z = self._logits(contribs)
        post = self._posterior_of(z)
        self._post = (self._stamp(), post)
        threat = post[:3].sum(axis=0)
        alive = threat.copy()
        out: list[TrackEst] = []
        support: dict[int, np.ndarray] = {}
        for key, c in contribs.items():
            support[key[0]] = support.get(key[0], 0) + c
        with np.errstate(divide="ignore"):  # -inf where every class is impossible: fine
            zt = np.log(np.exp(z).sum(axis=0))  # threat log-odds: sharp where bearings cross
        for _ in range(MAX_TRACKS):
            iy, ix = np.unravel_index(int(np.argmax(alive)), self.shape)
            peak = alive[iy, ix]
            if peak < TRACK_THRESHOLD:
                break
            px, py = self.X[iy, ix], self.Y[iy, ix]
            dist = np.hypot(self.X - px, self.Y - py)
            near = dist <= TRACK_RADIUS
            # relative likelihood, not probability: probability saturates into a plateau
            wts = np.where(near, np.exp(np.clip(zt - zt[iy, ix], -30, 0)), 0.0)
            wts[wts < 0.05] = 0.0
            wsum = wts.sum()
            mx, my = (wts * self.X).sum() / wsum, (wts * self.Y).sum() / wsum
            cov = np.cov(np.stack([self.X[wts > 0], self.Y[wts > 0]]), aweights=wts[wts > 0]) \
                if (wts > 0).sum() > 2 else np.eye(2) * (self.cell / 2) ** 2
            evals, evecs = np.linalg.eigh(cov)
            a, b = 2 * math.sqrt(max(evals[1], 1.0)), 2 * math.sqrt(max(evals[0], 1.0))
            ang = math.degrees(math.atan2(evecs[1, 1], evecs[0, 1])) % 180.0
            mass = [(post[i] * wts).sum() for i in range(3)]
            ci = int(np.argmax(mass))
            cls = THREATS[ci]
            origins = sum(1 for s in support.values() if s[ci][iy, ix] > 1.0)
            by_sensor = tuple(sorted(
                ((o, fam, round(float(c[ci, iy, ix]), 2)) for (o, fam), c in contribs.items()),
                key=lambda e: -abs(e[2])))
            by_sensor = tuple(e for e in by_sensor if abs(e[2]) >= 0.05)
            cam = float(self._camera_evidence()[ci][iy, ix])
            if abs(cam) >= 0.05:
                by_sensor += ((-1, "camera", round(cam, 2)),)
            prior = round(float(self.prior[ci][iy, ix]), 2)
            out.append(TrackEst(
                track_id=cell_id(mx, my, self.city.bounds), x=float(mx), y=float(my),
                semi_major=a, semi_minor=b, angle=ang, cls=cls,
                confidence=float(post[ci][iy, ix]),
                status=self._status(cls, mx, my), origins=origins,
                strength=float(zt[iy, ix]),
                support=by_sensor + ((-2, "prior", prior),),
            ))
            alive = np.where(dist <= TRACK_SUPPRESS, 0.0, alive)
        out = self._follow_confirmed(out, t_now)
        self._track_cache = out  # raw peaks: the Faulty Sensor check wants tight ellipses
        return merge_overlapping(out, self.city.bounds, self.threat)

    def _follow_confirmed(self, peaks: list[TrackEst], t_now: float) -> list[TrackEst]:
        """A confirmed threat stays confirmed as it moves on: each confirmed point takes the
        nearest peak of the Threat Class within CONFIRM_FOLLOW, once per extraction, and moves
        toward it (at most CONFIRM_SPEED). That peak is CONFIRMED, also where the City would
        explain it (a poacher on a track); other such peaks that close to a confirmed point
        are its side lobes, Explained. So the point cannot hop from peak to peak onto a
        second intruder, and side lobes are not broadcast as Confirmed. Following does not
        extend the hold: without a new camera sighting the point expires."""
        self.confirmed_near(0.0, 0.0, t_now)  # drop expired points
        out = list(peaks)
        taken: set[int] = set()
        for i, (cx, cy, seen_t, moved_t) in enumerate(list(self._confirmed)):
            near = [(math.hypot(tr.x - cx, tr.y - cy), j) for j, tr in enumerate(out)
                    if j not in taken and tr.cls == self.threat
                    and tr.status in (TrackStatus.SUSPECTED, TrackStatus.EXPLAINED)
                    and math.hypot(tr.x - cx, tr.y - cy) <= CONFIRM_FOLLOW]
            if not near:
                continue
            _, j = min(near)
            taken.add(j)
            out[j] = replace(out[j], status=TrackStatus.CONFIRMED)
            self._confirmed[i] = (*_toward(cx, cy, out[j].x, out[j].y, moved_t, t_now), seen_t,
                                  t_now)
        return [replace(tr, status=TrackStatus.EXPLAINED)
                if tr.cls == self.threat and tr.status == TrackStatus.SUSPECTED
                and self.confirmed_near(tr.x, tr.y, t_now, radius=CONFIRM_FOLLOW) else tr
                for tr in out]

    def _status(self, cls: TargetClass, x: float, y: float) -> TrackStatus:
        if cls == TargetClass.TRAIN and self.city.distance_to_rail(x, y) <= EXPLAIN_DISTANCE:
            return TrackStatus.EXPLAINED
        if cls == TargetClass.VEHICLE and self.city.distance_to_road(x, y) <= EXPLAIN_DISTANCE:
            return TrackStatus.EXPLAINED
        return TrackStatus.SUSPECTED

    def received_tracks(self, t_now: float, max_age: float = RX_MAX_AGE) -> list[Track]:
        """Each origin's current Tracks, by origin then arrival; older ones are pruned."""
        for o in list(self._tracks_rx):
            self._tracks_rx[o] = [tr for tr in self._tracks_rx[o] if t_now - tr.t <= RX_MAX_AGE]
            if not self._tracks_rx[o]:
                del self._tracks_rx[o]
        return [tr for o in sorted(self._tracks_rx) for tr in self._tracks_rx[o]
                if t_now - tr.t <= max_age]

    # -- dashboard ------------------------------------------------------------------------

    def heatmap(self, step: int = 2) -> dict:
        """Downsampled threat probability (uint8) and the argmax class, for the dashboard."""
        post = self.posterior()
        threat = post[:3].sum(axis=0)[::step, ::step]
        cls = post[:3].argmax(axis=0)[::step, ::step]
        return {
            "w": threat.shape[1], "h": threat.shape[0], "cell": self.cell * step,
            "p": (np.clip(threat, 0, 1) * 255).astype(np.uint8).tobytes(),
            "cls": cls.astype(np.uint8).tobytes(),
        }

    def summary(self) -> dict:
        """Small JSON for the dashboard, incl. readings fused per drone in the last 30 s.
        Cached until time, Evidence or the faulty set change (agents ask every tick)."""
        key = (self._t, len(self._fused), self.n_evidence, tuple(sorted(self.faulty)))
        if self._summary is not None and self._summary[0] == key:
            return self._summary[1]
        self._fused = [f for f in self._fused if self._t - f[0] <= 30.0]
        fused: dict[int, dict[str, int]] = {}
        for _, o, fam in self._fused:
            fused.setdefault(o, {}).setdefault(fam, 0)
            fused[o][fam] += 1
        out = {"n_evidence": self.n_evidence, "faulty": sorted(self.faulty), "fused": fused}
        key = (self._t, len(self._fused), self.n_evidence, tuple(sorted(self.faulty)))
        self._summary = (key, out)
        return out

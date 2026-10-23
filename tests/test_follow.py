"""Follow: the Quarry estimate and a Follower that stays with a moving threat, camera on it."""

from __future__ import annotations

import math
from pathlib import Path

from earshot.agent.belief import Belief, TrackEst, cell_id
from earshot.agent.follow import Quarry
from earshot.agent.policy import Policy
from earshot.backend.twin2d import Kinematics, Twin2D
from earshot.city import City
from earshot.messages import Mode, Sensor, TargetClass, TrackStatus
from earshot.runners.wire import command_from_dict, command_to_dict
from earshot.state import Command, DroneSpec, PolicyParams, SensorReading, SwarmConfig

ROOT = Path(__file__).resolve().parents[1]
CITY = City.load(ROOT / "data/cities/synthetic.json")
V = TargetClass.VEHICLE
V_PROBS = (0.0167, 0.95, 0.0167, 0.0167)


def _bearing(frm, to) -> float:
    return math.degrees(math.atan2(to[0] - frm[0], to[1] - frm[1])) % 360.0


def test_ranged_sightings_teach_the_quarry_its_velocity():
    q = Quarry(0.0, 0.0, 0.0)
    me = (-60.0, -60.0)
    for k in range(1, 15):
        car = (10.0 * k, 4.0 * k)
        assert q.sight(float(k), *me, _bearing(me, car), math.dist(me, car))
    assert abs(q.vx - 10.0) < 1.0 and abs(q.vy - 4.0) < 1.0
    x, y = q.at(16.5)  # coasts between sightings
    assert math.dist((x, y), (165.0, 66.0)) < 10.0


def test_a_bearing_takes_its_range_from_the_prediction():
    # a drone: the camera gives a direction only; the Confirmed Track gives the range
    q = Quarry(0.0, 100.0, 0.0)
    me = (0.0, -100.0)
    for k in range(1, 30):
        drone = (100.0 + 8.0 * k, 0.0)
        q.sight(float(k), *me, _bearing(me, drone), 0.0)
        q.track(k + 0.5, 100.0 + 8.0 * (k - 1.5), 0.0)  # the swarm's point lags 2 s
        me = (drone[0], drone[1] - 100.0)  # the Follower keeps up
    assert abs(q.speed - 8.0) < 1.5
    assert math.dist(q.at(29.0), (100.0 + 8.0 * 29, 0.0)) < 15.0


def test_a_fix_far_from_the_prediction_is_someone_else():
    q = Quarry(0.0, 0.0, 0.0)
    assert not q.sight(1.0, 0.0, -50.0, 90.0, 500.0)  # 500 m east: another vehicle
    assert (q.x, q.y) == (0.0, 0.0) and q.last_sighting is None


def test_command_speed_is_capped_by_the_airframe_and_crosses_the_wire():
    kin = Kinematics()
    assert kin.v(Mode.VERIFY) == 15.0 and kin.v(Mode.VERIFY, 7.0) == 7.0
    assert kin.v(Mode.VERIFY, 99.0) == kin.max_speed
    c = Command(Mode.VERIFY, goal=(1.0, 2.0), camera_on=True, look_at=(1.0, 2.0), speed=12.5)
    assert command_from_dict(command_to_dict(c)) == c


def _track(x, y, status) -> TrackEst:
    return TrackEst(cell_id(x, y, CITY.bounds), x, y, 60.0, 40.0, 0.0, V, 0.9, status,
                    origins=2, strength=9.0)


def test_a_follower_keeps_its_distance_and_its_camera_on_a_vehicle_that_speeds_up_and_stops():
    """Closed loop through the 2D twin: a pickup drives east at 6 m/s, speeds up to 20 m/s
    (faster than a Verifier cruises), then stops. Camera hits once a second, the swarm's
    Confirmed Track lags them."""
    spec = DroneSpec(0, -300.0, -300.0, home=(-300.0, -300.0))
    swarm = SwarmConfig.from_specs([spec], threat=V,
                                   params=PolicyParams.from_dict({"follow_for": 900.0,
                                                                  "same_task_radius": 400.0}))
    p = Policy(0, CITY, swarm)
    twin = Twin2D(CITY, [spec], Kinematics())
    s = twin.states()[0]

    def car(t):  # x along y = -150: 6 m/s, from t = 40 20 m/s, from t = 70 parked
        x = -150.0 + 6.0 * min(t, 40.0) + 20.0 * max(0.0, min(t, 70.0) - 40.0)
        return x, -150.0

    log, ended = [], []
    for k in range(900):
        t = round(k * 0.1, 1)
        truth = car(t)
        whole = abs(t - round(t)) < 1e-6
        status = TrackStatus.CONFIRMED if t >= 5.0 else TrackStatus.SUSPECTED
        lag = car(math.floor(t) - 1.0) if t >= 1.0 else truth
        tracks = [_track(*lag, status)]
        sightings = []
        if whole and s.camera_on and math.dist((s.x, s.y), truth) <= 300.0:
            rng = math.dist((s.x, s.y), truth)
            sightings = [SensorReading(Sensor.CAMERA, _bearing((s.x, s.y), truth), 2.0, V_PROBS,
                                       rng, rng)]
        d = p.decide(t, s, tracks, Belief(CITY, threat=V), {}, sightings)
        ended += [e["kind"] for e in d.events if e["kind"] in ("lost", "handed_off", "outbid")]
        s = twin.step(0.1, {0: d.command})[0]
        log.append((t, d.command, (s.x, s.y), truth))

    assert not ended  # followed all the way
    gap = {t: math.dist(me, truth) for t, _, me, truth in log}
    assert all(gap[t] < 65.0 for t in gap if 25.0 <= t < 40.0)  # cruising behind at 6 m/s
    assert all(gap[t] < 90.0 for t in gap if 55.0 <= t < 70.0)  # keeps up at 20 m/s
    # it brakes with the pickup: the Quarry coasts past the stop for a moment, so it parks
    # nearer than the gap (the twin never backs off), but not on top of it
    assert all(10.0 < gap[t] < 50.0 for t in gap if t >= 80.0)
    speeds = {t: c.speed for t, c, *_ in log if c.speed is not None}
    assert 4.0 < speeds[35.0] < 12.0 and speeds[65.0] > 17.0 and speeds[88.0] < 2.0
    looks = [math.dist(c.look_at, truth) for t, c, _, truth in log if t >= 25.0]
    assert max(looks) < 40.0  # the camera stays on the pickup
    assert all(c.camera_on and c.mode == Mode.VERIFY for t, c, *_ in log if t >= 6.0)


def test_the_camera_ranges_a_drone_by_its_size_for_the_follower_only():
    import numpy as np

    from earshot.scenario import Scenario, Source
    from earshot.sensors import SensorSuite
    from earshot.state import DroneState

    drone = Source("d", TargetClass.DRONE, 0, 100, [(0.0, 60.0)], 85.0)
    sc = Scenario("t", 1, "data/cities/synthetic.json", 100.0, 0.1, [], [drone])
    suite = SensorSuite(CITY, sc, np.random.default_rng(0))
    me = DroneState(0, 0.0, 0.0, 0.0, 40.0, 0.0, Mode.VERIFY, None, 1.0, 5700.0, True)
    hits = [d.reading for k in range(10) for d in suite.sense(me, float(k))
            if d.reading.sensor == Sensor.CAMERA and not math.isnan(d.reading.bearing)]
    assert hits and all(0.85 * 60.0 <= r.range <= 1.15 * 60.0 for r in hits)
    assert all(r.level == 0.0 for r in hits)  # the Belief and the radio still get a bearing


def test_a_guided_verifier_flies_to_the_intruder_confirms_and_stays_on_it():
    """Demo mock: the Suspicion sits 250 m off the real drone, the camera never fires; the
    guide alone brings the Verifier there, confirms it and keeps it within the gap."""
    D = TargetClass.DRONE
    spec = DroneSpec(0, -300.0, -300.0, home=(-300.0, -300.0))
    swarm = SwarmConfig.from_specs([spec], params=PolicyParams.from_dict({"follow_for": 600.0}))
    p = Policy(0, CITY, swarm)
    twin = Twin2D(CITY, [spec], Kinematics())
    s = twin.states()[0]
    truth = lambda t: (100.0 + 10.0 * t, 100.0)
    events, gaps = [], []
    for k in range(1200):
        t = round(k * 0.1, 1)
        x, y = truth(t)
        ghost = TrackEst(cell_id(x - 250, y, CITY.bounds), x - 250, y, 120.0, 60.0, 0.0, D,
                         0.9, TrackStatus.SUSPECTED, origins=2, strength=9.0)
        cmd = p._task_command(s) if p._task is not None and p._settled else None
        guide = (x, y) if cmd is not None and cmd.mode == Mode.VERIFY else None
        d = p.decide(t, s, [ghost], Belief(CITY), {}, (), guide)
        events += [(t, e["kind"]) for e in d.events if e["kind"] != "following"]
        s = twin.step(0.1, {0: d.command})[0]
        if p._follow_since is not None:
            gaps.append(math.dist((s.x, s.y), truth(t)))
    kinds = [k for _, k in events]
    assert kinds[:3] == ["claim", "dispatch", "confirmed"] and "lost" not in kinds
    assert gaps and max(gaps[100:]) < 50.0  # on the real drone, not on the Suspicion

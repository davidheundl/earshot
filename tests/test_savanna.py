"""The savanna Reserve (tools/make_savanna.py, scenarios/savanna.yaml) holds together: every
drone has a tower to perch on, the map fits the task ids, poachers stay slower than a
Verifier, and a short lockstep run goes through."""

import math
import sys
from pathlib import Path

import pytest

from earshot.agent.belief import CONFIRM_SPEED
from earshot.agent.policy import Policy
from earshot.backend.twin2d import Kinematics
from earshot.city import City
from earshot.messages import Mode, TargetClass
from earshot.scenario import Scenario
from earshot.state import SwarmConfig

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/savanna.yaml"


@pytest.fixture(scope="module")
def world():
    sc = Scenario.load(SCENARIO)
    return sc, City.load(ROOT / sc.city)


def test_every_drone_perches_on_a_tower_of_its_own(world):
    sc, city = world
    assert sc.threat == TargetClass.VEHICLE and len(sc.drones) == 20
    sites = {p.id: p for p in city.perch_sites}
    homes = set()
    for d in sc.drones:
        p = Policy(d.id, city, SwarmConfig.from_specs(sc.drones, threat=sc.threat))  # task ids fit
        site = p._home_site
        assert site.id.startswith("p_tower") and math.hypot(site.x - d.x, site.y - d.y) < 5.0
        assert site.id in sites and d.default == Mode.LISTEN
        homes.add(site.id)
    assert len(homes) == 20
    x0, y0, x1, y1 = city.bounds
    assert (x1 - x0) * (y1 - y0) == pytest.approx(100e6)  # 100 km2


def test_poachers_are_intruders_and_slower_than_a_verifier(world):
    sc, _ = world
    verifier = Kinematics.for_scenario(sc).v(Mode.VERIFY)
    by_id = {s.id: s for s in sc.sources}
    assert {s.id for s in sc.sources if s.intruder} == {"poacher1", "poacher2"}
    assert not by_id["ranger1"].intruder and "uhf" in by_id["poacher1"].rf_bands
    for s in sc.sources:
        length = sum(math.dist(a, b) for a, b in zip(s.path, s.path[1:], strict=False))
        speed = length / (s.end - s.start)
        assert speed < min(verifier, CONFIRM_SPEED) - 2.0, s.id
        if s.cls == TargetClass.NOTHING:
            assert s.kind and speed < 2.0  # game walks


def test_the_land_use_raster_speaks_the_builders_classes(world):
    _, city = world
    sys.path.insert(0, str(ROOT / "tools"))
    from build_webots_world import LANDUSE

    raw = city.to_json()
    assert raw["landuse"]["classes"] == LANDUSE and raw["biome"] == "savanna"
    assert len(raw["road_surfaces"]) == len(raw["roads"])


def test_a_short_run_goes_through():
    from earshot.runners.lockstep import run

    sc = Scenario.load(SCENARIO)
    sc.duration = 8.0
    r = run(sc, root=ROOT)
    assert r.steps == 80 and r.radio["sent"] > 0 and r.threats == []


def test_the_reserves_camera_is_coco_yolo(world):
    from earshot.sensors.camera import load_detector

    sc, _ = world
    det = load_detector(sc.camera)  # the keys the world and the kinematics read are not its own
    assert str(det.weights) == "yolo11n.pt" and det.verify_conf >= 0.25  # ByteTrack's new track
    assert det.class_of("truck") == TargetClass.VEHICLE
    assert det.class_of("elephant") == TargetClass.NOTHING
    assert det.class_of("quadcopter") == TargetClass.NOTHING  # not in this model's map
    # a Verifier holding its standoff counts as arrived, and its camera looks at that point
    assert sc.params["arrive"] > Kinematics.for_scenario(sc).standoff


def test_a_yolo_car_in_the_reserve_is_geolocated(world):
    # the camera 40 m up, aimed at the ground 120 m ahead, a car box standing on the image centre
    # three frames running: a vehicle reading 120 m along the bearing, as the stand-in's
    from types import SimpleNamespace

    import numpy as np

    from earshot.messages import Sensor
    from earshot.sensors import SensorSuite
    from earshot.sensors.camera import load_detector
    from earshot.state import DroneState

    class Car:
        def __init__(self):
            self.names = {0: "car"}

        def track(self, img, **kw):
            box = SimpleNamespace(cls=[0], conf=[0.4], id=[1],
                                  xyxy=[SimpleNamespace(tolist=lambda: [620, 370, 660, 384])])
            return [SimpleNamespace(boxes=[box], speed={})]

    sc, city = world
    det = load_detector(sc.camera)
    det.model, det.device = Car(), "cpu"
    suite = SensorSuite(city, sc, np.random.default_rng(0), camera=det)
    frame = {"bgra": bytes(1280 * 768 * 4), "w": 1280, "h": 768, "fov": 1.0, "yaw": 90.0,
             "pitch": math.atan2(40.0, 120.0), "height": 40.0}
    cams = []
    for k in range(12):
        t = 100.0 + k * 0.1
        me = DroneState(3, t, 0.0, 0.0, 400.0, 90.0, Mode.VERIFY, None, 1.0, 5700.0, True)
        cams += [d for d in suite.sense(me, t, frame) if d.reading.sensor == Sensor.CAMERA]
    assert cams and all(d.label == "car" for d in cams)
    r = cams[0].reading
    assert int(np.argmax(r.class_probs)) == TargetClass.VEHICLE
    assert r.bearing == pytest.approx(90.0, abs=1.0) and r.level == pytest.approx(120.0, rel=0.02)

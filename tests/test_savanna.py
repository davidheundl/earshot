"""The savanna Reserve (tools/make_savanna.py, scenarios/savanna.yaml) holds together: each
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
    assert sc.threat == TargetClass.VEHICLE and len(sc.drones) == 10
    sites = {p.id: p for p in city.perch_sites}
    homes = set()
    for d in sc.drones:
        p = Policy(d.id, city, SwarmConfig.from_specs(sc.drones, threat=sc.threat))  # task ids fit
        site = p._home_site
        assert site.id.startswith("p_tower") and math.hypot(site.x - d.x, site.y - d.y) < 5.0
        assert site.id in sites and d.default == Mode.LISTEN
        homes.add(site.id)
    assert len(homes) == 10
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
        speed = length / (s.end - s.start - sum(s.waits))  # driving, not stopped
        assert speed < min(verifier, CONFIRM_SPEED) - 2.0, s.id
        if s.cls == TargetClass.NOTHING:
            assert s.kind and speed < 2.0  # game walks


def test_the_shot_is_fired_standing_at_the_kill(world):
    sc, _ = world
    by_id = {s.id: s for s in sc.sources}
    shots = [e for e in sc.events if e["op"] == "shot"]
    assert shots
    for e in shots:
        src = by_id[e["source"]]
        at = src.position(e["t"])
        assert at in [p for p, w in zip(src.path, src.waits, strict=True) if w > 0]
        assert math.dist(at, by_id["elephants"].position(e["t"])) < 200.0  # rifle range


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


def test_the_reserves_coco_mapping_builds_a_detector(world):
    # scenarios/savanna.yaml keeps the stand-in; its commented YOLO block must still load
    from earshot.sensors.camera import load_detector

    sc, _ = world
    assert load_detector(sc.camera) is None  # detector: truth
    det = load_detector({**sc.camera, "detector": "yolo", "weights": "yolo11n.pt",
                         "classes": {"car": "vehicle", "truck": "vehicle", "elephant": "nothing"}})
    assert det.class_of("truck") == TargetClass.VEHICLE
    assert det.class_of("elephant") == TargetClass.NOTHING
    assert det.class_of("quadcopter") == TargetClass.NOTHING  # not in this model's map

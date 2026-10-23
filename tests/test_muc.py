"""The Munich Airport City (tools/make_muc.py) and its scenario."""

import json
import math
from pathlib import Path

from shapely import Point

from earshot.city import City
from earshot.runners.lockstep import run
from earshot.scenario import Scenario

ROOT = Path(__file__).resolve().parents[1]
CITY = ROOT / "data/cities/muc.json"


def test_city_fits_the_task_ids_and_its_assets_exist():
    c = City.load(CITY)
    raw = json.loads(CITY.read_text())
    x0, y0, x1, y1 = c.bounds
    assert math.ceil((x1 - x0) / 100) * math.ceil((y1 - y0) / 100) <= 4096  # policy.task_id
    files = [raw["landuse"]["image"], raw["basemap"]]
    files += [t["image"] for t in raw["ground_tiles"]]
    files += [f for m in raw["building_meshes"] for f in m.values()]
    assert all((CITY.parent / f).exists() for f in files)
    assert len(c.buildings) > 1000 and len(c.rails) >= 1 and len(c.roads) > 50


def test_perch_sites_sit_on_their_flat_roofs():
    c = City.load(CITY)
    by_id = {b.id: b for b in c.buildings}
    for p in c.perch_sites:
        b = by_id[p.building]
        assert b.footprint.buffer(1.0).contains(Point(p.x, p.y))
        assert abs(p.z - b.height) < 0.1


def test_listening_drones_start_on_a_roof():
    sc = Scenario.load(ROOT / "scenarios/muc.yaml")
    c = City.load(CITY)
    for d in sc.drones:
        if d.default.name == "LISTEN":
            assert min(math.hypot(p.x - d.x, p.y - d.y) for p in c.perch_sites) < 1.0


def test_intruder_over_the_apron_is_confirmed():
    sc = Scenario.load(ROOT / "scenarios/muc.yaml", seed=0)
    sc.duration = 200.0
    r = run(sc, root=ROOT)
    assert any(e["kind"] == "confirmed" for e in r.events)

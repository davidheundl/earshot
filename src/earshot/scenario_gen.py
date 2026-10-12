"""Random scenarios for the evaluation: same swarm, different world each seed.

    earshot eval --scenario random --runs 30

The swarm (drone homes, default modes, radio) comes from a base scenario; per seed the
generator places: one intruder (small or larger drone, 80 % with an RF link) loitering at a
random spot away from rail and roads, 1-2 trains, 2-3 vehicles, 1-2 Nuisances next to random
roofs, and in half the runs a Jam Zone over the intruder area.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from earshot.city import City
from earshot.messages import TargetClass
from earshot.scenario import JamZone, Scenario, Source


def generate(base: Scenario, seed: int, root: Path = Path(".")) -> Scenario:
    rng = np.random.default_rng(10_000 + seed)
    city = City.load(root / base.city)
    x0, y0, x1, y1 = city.bounds
    sc = Scenario(name=f"random-{seed}", seed=seed, city=base.city, duration=base.duration,
                  dt=base.dt, drones=base.drones, sources=[], radio=dict(base.radio),
                  policy=base.policy, environment=base.environment)

    # intruder: away from rail and roads, so it is not explained away by map knowledge
    for _ in range(10_000):
        cx, cy = rng.uniform(x0 + 200, x1 - 200), rng.uniform(y0 + 200, y1 - 400)
        if city.distance_to_rail(cx, cy) > 200 and city.distance_to_road(cx, cy) > 60:
            break
    else:
        raise ValueError(f"{base.city}: no spot 200 m from rail and 60 m from roads")
    start = float(rng.uniform(30, 120))
    loiter = [(cx + rng.uniform(-60, 60), cy + rng.uniform(-60, 60)) for _ in range(5)]
    sc.sources.append(Source(
        "intruder", TargetClass.DRONE, start, min(base.duration - 5, start + rng.uniform(200, 260)),
        loiter, float(rng.choice([82.0, 89.0], p=[0.3, 0.7])),
        ("2g4",) if rng.random() < 0.8 else (),
    ))

    rail = list(city.rails[0].coords) if city.rails else None
    for k in range(int(rng.integers(1, 3))):
        if rail is None:
            break
        t0 = float(rng.uniform(0, base.duration - 80))
        path = rail if rng.random() < 0.5 else rail[::-1]
        sc.sources.append(Source(f"train{k}", TargetClass.TRAIN, t0, t0 + 60, path, 120.0))

    for k in range(int(rng.integers(2, 4))):
        if not city.roads:
            break
        road = list(city.roads[int(rng.integers(len(city.roads)))].coords)
        t0 = float(rng.uniform(0, base.duration - 130))
        path = road if rng.random() < 0.5 else road[::-1]
        sc.sources.append(Source(f"truck{k}", TargetClass.VEHICLE, t0, t0 + 125, path, 105.0))

    roofs = city.perch_sites
    for k in range(int(rng.integers(1, 3))):
        if not roofs:
            break
        r = roofs[int(rng.integers(len(roofs)))]
        a = rng.uniform(0, 2 * math.pi)
        p = (r.x + 30 * math.cos(a), r.y + 30 * math.sin(a))
        sc.sources.append(Source(f"fan{k}", TargetClass.NOTHING, 0.0, base.duration, [p],
                                 float(rng.uniform(70, 78))))

    if rng.random() < 0.5:
        t0 = start + float(rng.uniform(40, 120))
        sc.jamming.append(JamZone(t0, t0 + 40.0, (cx, cy), 350.0))
    return sc

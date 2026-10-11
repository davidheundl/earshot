"""Generate a placeholder City JSON until the real LoD2 + OSM export exists.

2 x 2 km, a rail line across the north, two roads, a grid of blocks with flat roofs.
Deterministic (seeded) so tests and scenarios can rely on it.

    uv run python tools/make_synthetic_city.py data/cities/synthetic.json
"""

import json
import sys

import numpy as np

SIZE = 2000.0
BLOCK = 160.0
rng = np.random.default_rng(7)

buildings, perch = [], []
for i, cx in enumerate(np.arange(-SIZE / 2 + BLOCK, SIZE / 2, BLOCK * 1.6)):
    for j, cy in enumerate(np.arange(-SIZE / 2 + BLOCK, SIZE / 2 - 300, BLOCK * 1.6)):
        if rng.random() < 0.25:  # leave parks / plazas
            continue
        w, h = rng.uniform(40, 110, size=2)
        height = float(rng.choice([8, 12, 15, 20, 28]))
        bid = f"b{i}_{j}"
        fp = [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
              [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]
        buildings.append({"id": bid, "footprint": fp, "height": height})
        if rng.random() < 0.6:  # flat roof usable as a Perch Site
            perch.append({"id": f"p{i}_{j}", "building": bid,
                          "x": float(cx), "y": float(cy), "z": height})

city = {
    "name": "synthetic",
    "origin": {"lat": 48.3538, "lon": 11.7861, "epsg": 25832, "note": "placeholder"},
    "bounds": [-SIZE / 2, -SIZE / 2, SIZE / 2, SIZE / 2],
    "buildings": buildings,
    "perch_sites": perch,
    "rails": [[[-SIZE / 2, 820.0], [0.0, 760.0], [SIZE / 2, 840.0]]],
    "roads": [
        [[-SIZE / 2, -40.0], [SIZE / 2, -40.0]],
        [[300.0, -SIZE / 2], [300.0, SIZE / 2]],
    ],
}

out = sys.argv[1] if len(sys.argv) > 1 else "data/cities/synthetic.json"
with open(out, "w") as f:
    json.dump(city, f, indent=1)
print(f"{out}: {len(buildings)} buildings, {len(perch)} perch sites")

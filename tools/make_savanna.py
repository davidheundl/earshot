"""Generate the 'savanna' City: a 10 x 10 km private game reserve in the South African
lowveld (bushveld), the second world Earshot runs in. Deterministic (seeded).

    uv run python tools/make_savanna.py          -> data/cities/savanna.json + savanna_landuse.png

A Reserve is a City (earshot/city.py, JSON v2): instead of streets it has dirt tracks (the
Webots builder lays them as dirt, `road_surfaces`), instead of flat roofs the drones perch on
lookout towers, on the ranger camp and the outposts. The land is grass and open thornveld
with thickets, a meandering sand river with riverine forest, granite kopjes that block sound
and radio (terrain), and waterholes where the game comes to drink. A fence road runs round
the boundary; poachers come through the fence from outside.

`biome: savanna` tells the Webots builder and the dashboard to paint it as bushveld
(acacias, golden grass, red earth) and which sky to use.
"""

from __future__ import annotations

import json
import math
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
from shapely import LineString, Point, Polygon
from shapely.affinity import rotate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_webots_world import LANDUSE
from make_valley import distance_to_polyline, fractal, smoothstep

SEED = 24
HALF = 5000.0  # inner area: -5000 .. 5000 m (100 km2)
SPACING = 25.0  # 401 x 401 terrain grid
OUTER_HALF, OUTER_SPACING = 14000.0, 200.0
LANDUSE_PX = 2048  # ~4.9 m / px
FENCE_INSET = 120.0  # the boundary road runs this far inside the fence

GRASS, BUSH, THICKET, RIVERINE, SAND, ROCK, TRACK, CAMP = (
    LANDUSE.index(n) for n in ("grass", "bush", "thicket", "riverine", "sand", "rock", "track",
                               "camp"))
WATER = LANDUSE.index("water")

rng = np.random.default_rng(SEED)

# -- terrain ---------------------------------------------------------------------------------


def river_y(x):
    """The sand river meanders from the north-west to the south-east."""
    x = np.asarray(x, float)
    return 1500.0 - 0.32 * x + 520.0 * np.sin(x / 1300.0) + 160.0 * np.sin(x / 410.0 + 0.7)


KOPJES = [  # x, y, height, radius: granite outcrops
    (-3300.0, 2600.0, 38.0, 170.0), (2500.0, 3300.0, 30.0, 150.0), (-1200.0, -2900.0, 44.0, 190.0),
    (3600.0, -2300.0, 26.0, 130.0), (-4000.0, -1400.0, 22.0, 120.0), (600.0, -400.0, 18.0, 100.0),
    (4100.0, 1700.0, 34.0, 160.0),
]

WATERHOLES = [  # x, y, radius: dams and pans
    (-1700.0, 2150.0, 55.0), (1250.0, 900.0, 70.0), (-3200.0, -2600.0, 45.0), (2900.0, -1000.0, 60.0),
    (-500.0, -1700.0, 40.0), (3800.0, 3900.0, 45.0), (-4100.0, 600.0, 35.0),
]


def base_height(X, Y, big, small):
    """Lowveld plain around 350 m, falling gently to the east, broad swells, the river valley
    cut in, the kopjes standing out steeply."""
    h = 360.0 - 0.0028 * X + 9.0 * big + 1.5 * small
    d_river = np.abs(Y - river_y(X))
    h -= 11.0 * (1 - smoothstep(30.0, 420.0, d_river))
    for kx, ky, kh, kr in KOPJES:
        r = np.hypot(X - kx, Y - ky) / kr
        h += kh * np.exp(-(r ** 3))  # steep sides, rounded top
    return h


def make_terrain():
    n = int(2 * HALF / SPACING) + 1
    xs = -HALF + SPACING * np.arange(n)
    X, Y = np.meshgrid(xs, xs)
    big = fractal(X.shape, [80, 40, 16], [1.0, 0.5, 0.25], SEED)
    small = fractal(X.shape, [6, 3], [1.0, 0.5], SEED + 1)
    return X, Y, base_height(X, Y, big, small)


def outer_terrain():
    n = int(2 * OUTER_HALF / OUTER_SPACING) + 1
    xs = -OUTER_HALF + OUTER_SPACING * np.arange(n)
    X, Y = np.meshgrid(xs, xs)
    big = fractal(X.shape, [20, 10, 5], [1.0, 0.5, 0.25], SEED + 7)
    H = 360.0 - 0.0028 * X + 9.0 * big
    ridge = fractal(X.shape, [12, 6], [1.0, 0.4], SEED + 8)  # the escarpment far in the west
    H += (260.0 + 160.0 * ridge) * np.clip((-X - 9000.0) / 4000.0, 0, 1) ** 1.6
    H += (60.0 + 50.0 * ridge) * np.clip((np.hypot(X, Y) - 9000.0) / 5000.0, 0, 1) ** 2
    return xs, H


class Ground:
    def __init__(self, H):
        self.H = H

    def __call__(self, x, y):
        n = self.H.shape[0]
        i = np.clip((np.asarray(x) + HALF) / SPACING, 0, n - 1.001)
        j = np.clip((np.asarray(y) + HALF) / SPACING, 0, n - 1.001)
        i0, j0 = np.floor(i).astype(int), np.floor(j).astype(int)
        fi, fj = i - i0, j - j0
        H = self.H
        return ((1 - fi) * (1 - fj) * H[j0, i0] + fi * (1 - fj) * H[j0, i0 + 1]
                + (1 - fi) * fj * H[j0 + 1, i0] + fi * fj * H[j0 + 1, i0 + 1])


def flatten(X, Y, H, c, r, level=None):
    d = np.hypot(X - c[0], Y - c[1])
    level = float(np.median(H[d <= r])) if level is None else level
    w = 1 - smoothstep(r * 0.7, r * 1.3, d)
    return H * (1 - w) + level * w, level


# -- tracks -----------------------------------------------------------------------------------


def wiggle(pts, amp=25.0, step=40.0):
    """Densify a polyline and let it wander a little: bush tracks are never straight."""
    out = []
    for (ax, ay), (bx, by) in pairwise(pts):
        L = math.hypot(bx - ax, by - ay)
        k = max(2, int(L / step))
        nx, ny = -(by - ay) / L, (bx - ax) / L
        phase = rng.uniform(0, 6.28)
        for i in range(k):
            u = i / k
            w = amp * math.sin(u * math.pi) * math.sin(u * L / 230.0 + phase)
            out.append((ax + u * (bx - ax) + w * nx, ay + u * (by - ay) + w * ny))
    out.append(tuple(pts[-1]))
    return [(round(x, 1), round(y, 1)) for x, y in out]


def tracks():
    """(points, width, surface): the fence road, the gravel spine and the dirt tracks."""
    e = HALF - FENCE_INSET
    fence = [(-e, -e), (e, -e), (e, e), (-e, e), (-e, -e)]
    spine = wiggle([(-HALF, -300.0), (-2600.0, -500.0), (-600.0, 150.0), (1250.0, 650.0),
                    (2900.0, 250.0), (HALF, 500.0)], amp=40.0)
    dirt = [
        [(-2600.0, -500.0), (-3200.0, -2600.0), (-3600.0, -e)],  # HQ to the south fence
        [(-2600.0, -500.0), (-4100.0, 600.0), (-e, 1500.0)],  # HQ west
        [(-600.0, 150.0), (-1700.0, 2150.0), (-3300.0, 2950.0), (-2500.0, e)],  # north-west loop
        [(1250.0, 650.0), (1800.0, 2400.0), (2500.0, 3650.0), (3800.0, 3900.0), (e, 3200.0)],
        [(2900.0, 250.0), (2900.0, -1000.0), (3600.0, -2650.0), (2400.0, -e)],  # south-east
        [(-600.0, 150.0), (-500.0, -1700.0), (-1200.0, -3250.0), (0.0, -e)],  # south
        [(2900.0, 250.0), (4100.0, 1350.0), (e, 1900.0)],  # east
        [(-500.0, -1700.0), (1200.0, -1900.0), (2900.0, -1000.0)],  # south cross track
    ]
    out = [(wiggle(fence, amp=6.0, step=60.0), 5.0, "dirt"), (spine, 7.0, "gravel")]
    out += [(wiggle(p), 4.0, "dirt") for p in dirt]
    return out


# -- built things -------------------------------------------------------------------------------

CAMPS = [  # name, centre, radius (labels and cleared ground)
    ("Ranger HQ", (-2600.0, -500.0), 140.0, "camp"),
    ("Mhondoro Lodge", (1250.0, 650.0), 120.0, "lodge"),
    ("North outpost", (-600.0, 4600.0), 50.0, "outpost"),
    ("East outpost", (4620.0, -500.0), 50.0, "outpost"),
    ("South outpost", (1500.0, -4620.0), 50.0, "outpost"),
]


def rect(x, y, w, h, angle):
    return rotate(Polygon([(x - w / 2, y - h / 2), (x + w / 2, y - h / 2), (x + w / 2, y + h / 2),
                           (x - w / 2, y + h / 2)]), angle, origin=(x, y))


def put(out, taken, ground, bid, fp, height, floors, wall, roof, **look):
    if any(fp.buffer(3.0).intersects(t) for t in taken):
        return False
    taken.append(fp)
    corners = list(fp.exterior.coords)[:4]
    out.append({"id": bid, "footprint": [[round(x, 2), round(y, 2)] for x, y in corners],
                "base_z": round(float(min(ground(x, y) for x, y in corners)) - 0.4, 2),
                "height": height, "floors": floors, "floor_height": round(height / floors, 2),
                "wall": wall, "roof": roof, **look})
    return True


def camp_buildings(ground, taken, avoid):
    out = []
    for name, (cx, cy), r, kind in CAMPS:
        tag = "".join(w[0] for w in name.split()).lower()
        if kind == "camp":  # offices, workshop, staff houses, all under corrugated iron
            for k in range(40):
                if len([b for b in out if b["id"].startswith(tag)]) >= 9:
                    break
                a, d = rng.uniform(0, 6.28), rng.uniform(15, r * 0.8)
                w, h = rng.uniform(10, 22), rng.uniform(8, 14)
                fp = rect(cx + d * math.cos(a), cy + d * math.sin(a), w, h, rng.uniform(-8, 8))
                if fp.intersects(avoid):
                    continue
                put(out, taken, ground, f"{tag}{k}", fp, 4.0, 1,
                    str(rng.choice(["concrete building", "old house", "brick building"])),
                    "flat roof", roof_material="sheet metal")
            put(out, taken, ground, f"{tag}_water", rect(cx + 30, cy - 35, 5, 5, 0), 14.0, 4,
                "concrete building", "flat roof", roof_material="sheet metal", kind="tower")
        elif kind == "lodge":  # thatched chalets along the river, one flat-decked main lodge
            put(out, taken, ground, f"{tag}_main", rect(cx, cy, 26, 16, 12), 6.0, 2,
                "stone wall", "flat roof", roof_material="old tiles")
            for k in range(14):
                a = rng.uniform(0, 6.28)
                fp = rect(cx + r * 0.8 * math.cos(a), cy + r * 0.8 * math.sin(a), 8, 7, rng.uniform(0, 90))
                if fp.intersects(avoid):
                    continue
                put(out, taken, ground, f"{tag}{k}", fp, 3.0, 1, "stone wall", "hipped roof",
                    roof_material="old tiles")
        else:  # outpost: a block hut and a radio mast
            put(out, taken, ground, f"{tag}_hut", rect(cx, cy, 9, 7, 0), 3.5, 1,
                "concrete building", "flat roof", roof_material="sheet metal")
    return out


# drone homes: 20 lookout towers on a jittered 5 x 4 grid, each on the highest ground nearby
TOWER_GRID_X = [-4000.0, -2000.0, 0.0, 2000.0, 4000.0]
TOWER_GRID_Y = [-3750.0, -1250.0, 1250.0, 3750.0]


def towers(ground, taken, avoid):
    out = []
    for j, gy in enumerate(TOWER_GRID_Y):
        for i, gx in enumerate(TOWER_GRID_X):
            cx, cy = gx + rng.uniform(-350, 350), gy + rng.uniform(-350, 350)
            best = None
            for _ in range(60):  # highest open spot within 300 m: a tower sees and hears further
                x, y = cx + rng.uniform(-300, 300), cy + rng.uniform(-300, 300)
                if not (-HALF + 400 < x < HALF - 400 and -HALF + 400 < y < HALF - 400):
                    continue
                if Point(x, y).buffer(20).intersects(avoid):
                    continue
                z = float(ground(x, y))
                if best is None or z > best[2]:
                    best = (x, y, z)
            x, y, _ = best
            put(out, taken, ground, f"tower{j * 5 + i}", rect(x, y, 4, 4, rng.uniform(0, 90)), 12.0, 3,
                "concrete building", "flat roof", roof_material="sheet metal", kind="tower")
    return out


# -- land use ---------------------------------------------------------------------------------


def landuse_raster(H, roads, waters, buildings):
    n = LANDUSE_PX
    px = 2 * HALF / n
    xs = -HALF + px * (np.arange(n) + 0.5)
    X, Y = np.meshgrid(xs, xs)  # row 0 = south
    Hs = zoom(H, n / H.shape[0], order=1)[:n, :n]
    slope = np.hypot(*np.gradient(Hs, px))
    noise = fractal((n, n), [256, 64, 16], [1.0, 0.5, 0.25], SEED + 3)
    fine = fractal((n, n), [24, 6], [1.0, 0.5], SEED + 4)
    lu = np.full((n, n), GRASS, dtype=np.uint8)
    lu[noise + 0.35 * fine > -0.15] = BUSH  # open thornveld over half the reserve
    lu[(noise + 0.5 * fine > 0.55)] = THICKET
    d_river = np.abs(Y - river_y(X))
    lu[(d_river < 130.0 + 60.0 * fine)] = RIVERINE
    lu[d_river < 14.0 + 6.0 * fine] = SAND  # the dry river bed
    for kx, ky, _, kr in KOPJES:
        d = np.hypot(X - kx, Y - ky) * (1 + 0.25 * fine)  # ragged, not a circle
        foot = (d >= 0.75 * kr) & (d < 1.5 * kr) & (lu != SAND) & (noise + fine > -0.6)
        lu[foot] = THICKET  # bush clings to the foot
        lu[(d < 0.75 * kr) | ((d < 1.0 * kr) & (slope > 0.12))] = ROCK
    for _, (cx, cy), r, _ in CAMPS:
        lu[np.hypot(X - cx, Y - cy) <= r * 0.9] = CAMP
    built = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(built)
    for b in buildings:
        draw.polygon([((x + HALF) / px, n - (y + HALF) / px) for x, y in b["footprint"]], fill=255)
    lu[(np.array(built)[::-1] > 0)] = CAMP  # cleared ground under every hut and tower
    for w in waters:
        d = np.hypot(X - w["x"], Y - w["y"])
        lu[(d <= w["r"] + 25) & (lu != WATER)] = SAND  # trampled mud round the water
        lu[d <= w["r"]] = WATER
    for pts, width, _ in roads:
        d, _ = distance_to_polyline(X, Y, np.asarray(pts))
        lu[d <= width / 2 + 0.5] = TRACK
    return lu


# -- main -------------------------------------------------------------------------------------


def main(out_json: Path) -> None:
    X, Y, H = make_terrain()

    waters = []
    for wx, wy, wr in WATERHOLES:  # a basin, the water 1.5 m below its rim
        d = np.hypot(X - wx, Y - wy)
        rim = float(np.median(H[(d > wr) & (d < wr + 40)]))
        z = rim - 1.5
        H = np.where(d <= wr + 60, np.minimum(H, z - 2.0 + 3.5 * smoothstep(wr - 20, wr + 60, d)), H)
        waters.append({"x": wx, "y": wy, "r": wr, "z": round(z, 2)})
    for _, c, r, _ in CAMPS:
        H, _ = flatten(X, Y, H, c, r)

    roads = tracks()
    for pts, width, _ in roads:  # graded tracks: the 25 m grid must not poke through
        d, _ = distance_to_polyline(X, Y, np.asarray(pts))
        H = np.where(d < width / 2 + 30.0, H - 0.3 * (1 - smoothstep(width / 2, width / 2 + 30.0, d)), H)

    ground = Ground(H)
    avoid = Point(0, 0).buffer(0)
    for pts, width, _ in roads:
        avoid = avoid.union(LineString(pts).buffer(width / 2 + 2.0))
    for w in waters:
        avoid = avoid.union(Point(w["x"], w["y"]).buffer(w["r"] + 20))
    taken: list = []
    buildings = camp_buildings(ground, taken, avoid)
    buildings += towers(ground, taken, avoid)

    perch = []
    for b in buildings:
        fp = Polygon(b["footprint"])
        if b["roof"] == "flat roof" and (fp.area >= 60 or b.get("kind") == "tower"):
            c = fp.centroid
            perch.append({"id": f"p_{b['id']}", "building": b["id"], "x": round(c.x, 1),
                          "y": round(c.y, 1), "z": round(b["base_z"] + b["height"], 2)})

    lu = landuse_raster(H, roads, waters, buildings)
    lu_path = out_json.with_name(out_json.stem + "_landuse.png")
    Image.fromarray(lu[::-1]).save(lu_path)  # image row 0 = north

    _oxs, OH = outer_terrain()
    city = {
        "name": "savanna",
        "origin": {"lat": -24.85, "lon": 31.45, "note": "procedural bushveld reserve, lowveld-like"},
        "biome": "savanna",
        "bounds": [-HALF, -HALF, HALF, HALF],
        "terrain": {"origin": [-HALF, -HALF], "spacing": SPACING, "n": int(H.shape[0]),
                    "heights": [round(float(v), 2) for v in H.ravel()]},
        "terrain_outer": {"origin": [-OUTER_HALF, -OUTER_HALF], "spacing": OUTER_SPACING,
                          "n": int(OH.shape[0]), "heights": [round(float(v), 1) for v in OH.ravel()]},
        "landuse": {"image": lu_path.name, "classes": LANDUSE},
        "water": waters,
        "towns": [{"name": n, "x": c[0], "y": c[1], "r": r, "kind": k} for n, c, r, k in CAMPS],
        "kopjes": [{"x": x, "y": y, "r": r, "h": h} for x, y, h, r in KOPJES],
        "buildings": buildings,
        "perch_sites": perch,
        "rails": [],
        "roads": [[list(p) for p in pts] for pts, _, _ in roads],
        "road_widths": [w for _, w, _ in roads],
        "road_surfaces": [s for _, _, s in roads],
    }
    out_json.write_text(json.dumps(city))
    homes = [p for p in perch if p["id"].startswith("p_tower")]
    print(f"{out_json}: {len(buildings)} buildings, {len(perch)} perch sites ({len(homes)} towers), "
          f"{len(roads)} tracks, terrain {H.min():.0f}..{H.max():.0f} m, landuse {lu_path.name}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/cities/savanna.json"))

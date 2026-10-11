"""Generate the 'valley' City: a 4 x 4 km landscape with hills, a lake, forests, fields, a town,
an industrial park, villages, roads and a rail line. Deterministic (seeded).

    uv run python tools/make_valley.py            -> data/cities/valley.json + valley_landuse.png

Writes City JSON v2 (see earshot/city.py): the inner terrain grid (10 m) the simulation uses,
a coarse outer ring (100 m, visual only, out to the horizon), buildings with base heights and
Webots styles, Perch Sites on flat roofs, roads and rail as polylines, the lake, and a land-use
raster (PNG, one class id per pixel) that the Webots builder paints and the dashboard draws.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter, zoom
from shapely import LineString, Point, Polygon
from shapely.affinity import rotate, translate

SEED = 11
HALF = 2000.0  # inner area: -2000 .. 2000 m
SPACING = 10.0
OUTER_HALF, OUTER_SPACING = 6000.0, 100.0
LANDUSE_PX = 1024  # land-use raster over the inner area (~4 m / px)

# land-use classes (ids in the raster; names for the dashboard and the builder)
LANDUSE = ["meadow", "field", "forest", "urban", "water", "road", "rail", "industrial"]
MEADOW, FIELD, FOREST, URBAN, WATER, ROAD, RAIL, INDUSTRIAL = range(len(LANDUSE))

rng = np.random.default_rng(SEED)


# -- terrain ---------------------------------------------------------------------------------

def fractal(shape, scales_px, amps, seed):
    r = np.random.default_rng(seed)
    out = np.zeros(shape)
    for s, a in zip(scales_px, amps, strict=True):
        coarse = r.normal(size=(max(2, shape[0] // s + 2), max(2, shape[1] // s + 2)))
        up = zoom(coarse, s, order=3)[: shape[0], : shape[1]]
        out += a * up / (np.abs(up).max() or 1)
    return out


def valley_center(x):
    return -250.0 + 160.0 * np.sin(x / 900.0) + 60.0 * np.sin(x / 370.0 + 1.0)


PEAKS = [(-250.0, 750.0, 130.0, 260.0), (1450.0, 350.0, 110.0, 230.0),  # x, y, height, radius
         (-1650.0, -650.0, 95.0, 220.0), (600.0, 1500.0, 120.0, 300.0), (-200.0, -1250.0, 85.0, 230.0)]


def base_height(X, Y, noise_big, noise_small):
    """Valley floor around 30 m, hills rising to the north and south edges, a few distinct
    hilltops, fractal relief."""
    d = np.abs(Y - valley_center(X)) / 1300.0
    shape = 140.0 * np.clip(d, 0, None) ** 1.4
    peaks = sum(h * np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * r * r)) for px, py, h, r in PEAKS)
    return 30.0 + shape + peaks + 60.0 * noise_big + 8.0 * noise_small


def make_terrain():
    n = int(2 * HALF / SPACING) + 1
    xs = -HALF + SPACING * np.arange(n)
    X, Y = np.meshgrid(xs, xs)
    big = fractal(X.shape, [160, 80, 40], [1.0, 0.5, 0.25], SEED)
    small = fractal(X.shape, [12, 5], [1.0, 0.5], SEED + 1)
    return X, Y, base_height(X, Y, big, small)


def outer_terrain():
    n = int(2 * OUTER_HALF / OUTER_SPACING) + 1
    xs = -OUTER_HALF + OUTER_SPACING * np.arange(n)
    X, Y = np.meshgrid(xs, xs)
    big = fractal(X.shape, [16, 8, 4], [1.0, 0.5, 0.25], SEED + 7)
    H = base_height(X, Y, big, np.zeros_like(X))
    ridge = fractal(X.shape, [10, 5], [1.0, 0.4], SEED + 8)  # mountains frame the horizon
    H += (180.0 + 140.0 * ridge) * np.clip((np.hypot(X, Y) - 3200.0) / 2500.0, 0, 1) ** 1.5
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


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def flatten_disc(X, Y, H, c, r, level=None):
    d = np.hypot(X - c[0], Y - c[1])
    inside = d <= r
    level = float(np.median(H[inside])) if level is None else level
    w = 1 - smoothstep(r * 0.75, r * 1.25, d)
    return H * (1 - w) + level * w, level


def distance_to_polyline(X, Y, pts):
    d = np.full(X.shape, np.inf)
    t_best = np.zeros(X.shape)
    acc = 0.0
    for (ax, ay), (bx, by) in itertools.pairwise(pts):
        vx, vy = bx - ax, by - ay
        L = math.hypot(vx, vy) or 1.0
        u = np.clip(((X - ax) * vx + (Y - ay) * vy) / (L * L), 0, 1)
        dd = np.hypot(X - (ax + u * vx), Y - (ay + u * vy))
        better = dd < d
        d = np.where(better, dd, d)
        t_best = np.where(better, acc + u * L, t_best)
        acc += L
    return d, t_best


def corridor(X, Y, H, pts, half_width, shoulder, max_grade=0.06):
    """Cut and fill the terrain along a road or rail: a smooth profile with limited grade."""
    pts = np.asarray(pts, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    ground = Ground(H)
    z = gaussian_filter(np.asarray(ground(pts[:, 0], pts[:, 1])), sigma=6, mode="nearest")
    for k in range(1, len(z)):  # grade limit, both directions
        z[k] = np.clip(z[k], z[k - 1] - max_grade * seg[k - 1], z[k - 1] + max_grade * seg[k - 1])
    for k in range(len(z) - 2, -1, -1):
        z[k] = np.clip(z[k], z[k + 1] - max_grade * seg[k], z[k + 1] + max_grade * seg[k])
    d, t = distance_to_polyline(X, Y, pts)
    zt = np.interp(t, s, z)
    w = 1 - smoothstep(half_width, half_width + shoulder, d)
    return H * (1 - w) + zt * w


def polyline(x0, x1, f, step=20.0):
    xs = np.arange(x0, x1 + step, step)
    return [(float(x), float(f(x))) for x in xs]


def bezier(p0, p1, p2, step=20.0):
    L = math.dist(p0, p1) + math.dist(p1, p2)
    ts = np.linspace(0, 1, max(3, int(L / step)))
    return [(float((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]),
             float((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])) for t in ts]


# -- settlements ------------------------------------------------------------------------------

TOWNS = [  # name, centre, radius, kind, street orientation (deg)
    ("Altstadt", (-950.0, -230.0), 420.0, "city", 8.0),
    ("Gewerbepark", (420.0, -470.0), 340.0, "industrial", -12.0),
    ("Hochfeld", (1050.0, 880.0), 250.0, "village", 25.0),
    ("Unterau", (-1350.0, -1350.0), 240.0, "village", -20.0),
    ("Nordberg", (-560.0, 1180.0), 210.0, "village", 5.0),
]

STYLES = {  # lots: a block is split into lots x lots parcels, each built with probability fill
    "city": {"floors": (3, 9), "floor_h": 3.2, "lots": 3, "fill": 0.9, "flat": 0.65, "block": 66.0,
                 "walls": ["office building", "residential building", "brick building",
                        "concrete building", "gray glass building", "old office building",
                        "classic building", "windowed building"]},
    "industrial": {"floors": (2, 4), "floor_h": 4.5, "lots": 1, "fill": 0.95, "flat": 0.95, "block": 105.0,
                       "walls": ["factory building", "concrete building", "construction building"]},
    "village": {"floors": (2, 3), "floor_h": 3.0, "lots": 3, "fill": 0.6, "flat": 0.3, "block": 54.0,
                    "walls": ["old house", "residential building", "tall house", "brick building",
                           "stone brick"]},
}
ROOFS_PITCHED = ["gabled roof", "hipped roof"]


def town_streets(c, r, angle, block):
    """A rotated street grid clipped to the town disc: list of 2-point lines."""
    lines = []
    k = int(r // block) + 1
    disc = Point(c).buffer(r * 0.95)
    for i in range(-k, k + 1):
        for horizontal in (True, False):
            off = i * block
            raw = LineString([(-r, off), (r, off)]) if horizontal else LineString([(off, -r), (off, r)])
            ln = translate(rotate(raw, angle, origin=(0, 0)), c[0], c[1]).intersection(disc)
            if ln.length > 60 and ln.geom_type == "LineString":
                lines.append([tuple(map(float, p)) for p in ln.coords])
    return lines


def place_buildings(town, avoid, ground, taken):
    """Blocks between the streets are split into lots; each lot gets one building set back
    from its edges, so streets end up lined with buildings like in a real town."""
    name, c, r, kind, angle = town
    st = STYLES[kind]
    out = []
    block, n = st["block"], st["lots"]
    street = 9.0  # street + pavement band each side of a street centre line
    lot = (block - street) / n
    k = int(r // block) + 1
    for i in range(-k, k):
        for j in range(-k, k):
            for li in range(n):
                for lj in range(n):
                    if rng.random() > st["fill"]:
                        continue
                    lx = i * block + street / 2 + (li + 0.5) * lot
                    ly = j * block + street / 2 + (lj + 0.5) * lot
                    setback = rng.uniform(1.5, 4.0) if kind != "village" else rng.uniform(3, 7)
                    w = max(6.0, lot - 2 * setback - rng.uniform(0, lot * 0.25))
                    h = max(6.0, lot - 2 * setback - rng.uniform(0, lot * 0.25))
                    rect = Polygon([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
                    fp = translate(rotate(translate(rect, lx, ly), angle, origin=(0, 0)), *c)
                    edge = r * (0.75 + 0.25 * math.sin(3 * math.atan2(ly, lx) + i))  # ragged edge
                    if fp.centroid.distance(Point(c)) > edge:
                        continue
                    if fp.intersects(avoid) or any(fp.buffer(1.0).intersects(t) for t in taken):
                        continue
                    taken.append(fp)
                    corners = list(fp.exterior.coords)[:4]
                    base = float(min(ground(x, y) for x, y in corners)) - 0.5
                    floors = int(rng.integers(st["floors"][0], st["floors"][1] + 1))
                    if kind == "city":  # taller towards the centre
                        floors = max(2, floors - int(3 * fp.centroid.distance(Point(c)) / r))
                    flat = rng.random() < st["flat"]
                    out.append({
                        "id": f"{name[:3].lower()}{len(out)}",
                        "footprint": [list(map(float, p)) for p in corners],
                        "base_z": round(base, 2),
                        "height": round(floors * st["floor_h"], 2),
                        "floors": floors, "floor_height": st["floor_h"],
                        "wall": str(rng.choice(st["walls"])),
                        "roof": "flat roof" if flat else str(rng.choice(ROOFS_PITCHED)),
                        "town": name,
                    })
    return out


def village_halls(town, avoid, ground, taken, count=2):
    """A community hall / fire station / barn with a flat roof: where village drones perch."""
    name, c, r, _kind, angle = town
    out = []
    for _ in range(200):
        if len(out) >= count:
            break
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0, r * 0.6)
        x, y = c[0] + d * math.cos(a), c[1] + d * math.sin(a)
        w, h = rng.uniform(16, 24), rng.uniform(12, 18)
        fp = rotate(Polygon([(x - w / 2, y - h / 2), (x + w / 2, y - h / 2),
                             (x + w / 2, y + h / 2), (x - w / 2, y + h / 2)]), angle, origin=(x, y))
        if fp.intersects(avoid) or any(fp.buffer(2).intersects(t) for t in taken):
            continue
        taken.append(fp)
        corners = list(fp.exterior.coords)[:4]
        out.append({"id": f"{name[:3].lower()}hall{len(out)}",
                    "footprint": [list(map(float, p)) for p in corners],
                    "base_z": round(float(min(ground(x, y) for x, y in corners)) - 0.5, 2),
                    "height": 8.0, "floors": 2, "floor_height": 4.0,
                    "wall": str(rng.choice(["concrete building", "brick building", "factory building"])),
                    "roof": "flat roof", "town": name})
    return out


def farms(avoid, ground, taken, count=14):
    out = []
    while len(out) < count:
        x, y = rng.uniform(-1800, 1800), rng.uniform(-1800, 1800)
        if any(Point(x, y).distance(Point(t[1])) < t[2] + 150 for t in TOWNS):
            continue
        w, h = rng.uniform(10, 18), rng.uniform(16, 30)
        fp = rotate(Polygon([(x - w / 2, y - h / 2), (x + w / 2, y - h / 2),
                             (x + w / 2, y + h / 2), (x - w / 2, y + h / 2)]), rng.uniform(0, 90))
        if fp.buffer(15).intersects(avoid) or any(fp.buffer(10).intersects(t) for t in taken):
            continue
        taken.append(fp)
        corners = list(fp.exterior.coords)[:4]
        out.append({"id": f"farm{len(out)}", "footprint": [list(map(float, p)) for p in corners],
                    "base_z": round(float(min(ground(x, y) for x, y in corners)) - 0.5, 2),
                    "height": 6.0, "floors": 2, "floor_height": 3.0,
                    "wall": str(rng.choice(["old house", "stone wall", "old building"])),
                    "roof": "gabled roof", "town": "farm"})
    return out


# -- land use ---------------------------------------------------------------------------------

def landuse_raster(H, roads, rails, lake, towns, buildings):
    n = LANDUSE_PX
    px = 2 * HALF / n
    xs = -HALF + px * (np.arange(n) + 0.5)
    X, Y = np.meshgrid(xs, xs)  # row 0 = south
    Hs = zoom(H, n / H.shape[0], order=1)[:n, :n]
    slope = np.hypot(*np.gradient(Hs, px))
    lu = np.full((n, n), MEADOW, dtype=np.uint8)
    # fields on gentle lowland, forest on hills and steep ground, meadow in between
    noise = fractal((n, n), [128, 32], [1.0, 0.4], SEED + 3)
    near_town = np.zeros((n, n), dtype=bool)
    for _, c, r, _, _ in towns:
        near_town |= np.hypot(X - c[0], Y - c[1]) <= r + 180
    fieldness = (Hs < 110) & (slope < 0.12) & (noise > -0.35)
    lu[fieldness] = FIELD
    forest = (((Hs > 140) & (noise > -0.1)) | (slope > 0.22) | (noise > 0.7)) & ~near_town
    lu[forest] = FOREST
    # urban ground only where buildings and streets are: irregular, real town edges
    built = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(built)
    for b in buildings:
        draw.polygon([((x + HALF) / px, n - (y + HALF) / px) for x, y in b["footprint"]], fill=255)
    built = np.array(built)[::-1] > 0
    for _, c, r, kind, _ in towns:
        inside = (np.hypot(X - c[0], Y - c[1]) <= r) & binary_dilation(built, iterations=int(18 / px))
        lu[inside] = INDUSTRIAL if kind == "industrial" else URBAN
    if lake:
        lu[np.hypot(X - lake["x"], Y - lake["y"]) <= lake["r"]] = WATER
    for pts, width in roads:  # painted slightly narrower than the Road PROTO drawn on top
        d, _ = distance_to_polyline(X, Y, np.asarray(pts))
        lu[d <= width / 2 - 0.5] = ROAD
    for pts in rails:
        d, _ = distance_to_polyline(X, Y, np.asarray(pts))
        lu[d <= 3.5] = RAIL
    return lu


# -- main -------------------------------------------------------------------------------------

def main(out_json: Path) -> None:
    X, Y, H = make_terrain()

    # lake in a basin, south-east
    lake = {"x": 1250.0, "y": -1250.0, "r": 260.0}
    d = np.hypot(X - lake["x"], Y - lake["y"])
    rim = float(np.median(H[(d > lake["r"]) & (d < lake["r"] + 40)]))
    lake["z"] = round(rim - 4.0, 2)
    H = np.where(d <= lake["r"] + 60, np.minimum(H, lake["z"] - 6 + 10 * smoothstep(lake["r"] - 80, lake["r"] + 60, d)), H)

    # settlements sit on flattened ground
    for t in TOWNS:
        H, _ = flatten_disc(X, Y, H, t[1], t[2])

    # rail and main road along the valley, spurs to the villages and the lake
    rail = polyline(-HALF, HALF, lambda x: valley_center(x) + 120.0)
    main_road = polyline(-HALF, HALF, lambda x: valley_center(x) - 70.0)
    spurs = [
        bezier((-560.0, float(valley_center(-560.0)) - 70), (-700.0, 500.0), (-560.0, 1180.0)),
        bezier((1050.0, float(valley_center(1050.0)) - 70), (1300.0, 300.0), (1050.0, 880.0)),
        bezier((-1350.0, float(valley_center(-1350.0)) - 70), (-1500.0, -900.0), (-1350.0, -1350.0)),
        bezier((800.0, float(valley_center(800.0)) - 70), (1100.0, -800.0), (1180.0, -980.0)),
        bezier((-1350.0, -1350.0), (-300.0, -1500.0), (500.0, -800.0)),
    ]
    roads = [(main_road, 9.0)] + [(s, 7.0) for s in spurs]
    for name, c, r, kind, ang in TOWNS:
        blk = STYLES[kind]["block"]
        roads += [(ln, 7.0 if kind != "village" else 6.0) for ln in town_streets(c, r, ang, blk)]
    H = corridor(X, Y, H, rail, 10.0, 25.0, max_grade=0.02)  # rail first: roads win where they meet
    for pts, width in roads:  # flat well beyond the asphalt: the 10 m grid must not poke through
        H = corridor(X, Y, H, pts, width / 2 + 8, 25.0)

    ground = Ground(H)
    avoid = LineString(rail).buffer(12)
    for pts, width in roads:
        avoid = avoid.union(LineString(pts).buffer(width / 2 + 1.5))
    avoid = avoid.union(Point(lake["x"], lake["y"]).buffer(lake["r"] + 15))
    taken: list = []
    buildings = []
    for t in TOWNS:
        if t[3] == "village":
            buildings += village_halls(t, avoid, ground, taken)  # first, so they find room
        buildings += place_buildings(t, avoid, ground, taken)
    buildings += farms(avoid, ground, taken)

    perch = []
    for b in buildings:
        fp = Polygon(b["footprint"])
        if b["roof"] == "flat roof" and fp.area >= 100:  # ~10 x 10 m is enough to land
            c = fp.centroid
            perch.append({"id": f"p_{b['id']}", "building": b["id"], "x": round(c.x, 1),
                          "y": round(c.y, 1), "z": round(b["base_z"] + b["height"], 2)})

    lu = landuse_raster(H, roads, [rail], lake, TOWNS, buildings)
    lu_path = out_json.with_name(out_json.stem + "_landuse.png")
    Image.fromarray(lu[::-1]).save(lu_path)  # image row 0 = north

    _oxs, OH = outer_terrain()
    city = {
        "name": "valley",
        "origin": {"lat": 48.35, "lon": 11.78, "note": "procedural landscape"},
        "bounds": [-HALF, -HALF, HALF, HALF],
        "terrain": {"origin": [-HALF, -HALF], "spacing": SPACING, "n": int(H.shape[0]),
                    "heights": [round(float(v), 2) for v in H.ravel()]},
        "terrain_outer": {"origin": [-OUTER_HALF, -OUTER_HALF], "spacing": OUTER_SPACING,
                          "n": int(OH.shape[0]), "heights": [round(float(v), 1) for v in OH.ravel()]},
        "landuse": {"image": lu_path.name, "classes": LANDUSE},
        "water": [lake],
        "towns": [{"name": n, "x": c[0], "y": c[1], "r": r, "kind": k} for n, c, r, k, _ in TOWNS],
        "buildings": buildings,
        "perch_sites": perch,
        "rails": [[list(p) for p in rail]],
        "roads": [[list(p) for p in pts] for pts, _ in roads],
        "road_widths": [w for _, w in roads],
    }
    out_json.write_text(json.dumps(city))
    print(f"{out_json}: {len(buildings)} buildings, {len(perch)} perch sites, {len(roads)} roads, "
          f"terrain {H.min():.0f}..{H.max():.0f} m, landuse {lu_path.name}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/cities/valley.json"))

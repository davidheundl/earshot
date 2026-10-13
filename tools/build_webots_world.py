"""City JSON + scenario -> Webots world (.wbt) plus its assets, with the DEF names
WebotsBackend expects (DRONE_<id>, SRC_<id>[_w<k>], RIG).

    uv run python tools/build_webots_world.py scenarios/valley.yaml webots/worlds/valley.wbt

The look (ADR 0005 superseded: the venue has WLAN, Webots fetches PROTO assets on first open):
- terrain: the City's 10 m height grid as an ElevationGrid with a painted land-use texture
  (fields with crop rows, meadows, forest floor, streets), plus a coarse ring out to the horizon
- Webots SimpleBuilding per building (walls, roof shape and material from the City), Road
  PROTOs draped on the terrain, a rail bed with rails and sleepers, a lake
- forests and scattered trees as a few merged meshes (Webots' Forest cannot follow hills)
- sky, sun and haze; quadcopters from primitives (swarm in signal orange; intruders larger and
  dark, with a glowing red ring and beam, --no-markers to hide); trucks, a train of separate
  wagons, and a rooftop fan unit as Sources
Assets go to <world>_assets/ next to the world; a hash skips rebuilding when nothing changed.
A City without terrain (the flat test city) gets the same look on flat ground.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from shapely import Point, Polygon, STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import itertools

from earshot.backend.webots import LIVE_SLOTS, N_VIEWS, TRAIN_WAGONS, live_slot
from earshot.city import City
from earshot.scenario import Scenario

BUILDER_VERSION = "6"
PROTO = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects"
EXTERNPROTOS = [
    "objects/backgrounds/protos/TexturedBackground.proto",
    "objects/backgrounds/protos/TexturedBackgroundLight.proto",
    "objects/buildings/protos/SimpleBuilding.proto",
    "objects/road/protos/Road.proto",
    "vehicles/protos/generic/TruckSimple.proto",
]
LANDUSE = ["meadow", "field", "forest", "urban", "water", "road", "rail", "industrial",
           "runway", "taxiway", "apron"]  # the last three: airfields (tools/make_muc.py)
TEXTURE_PX = 2048
TREE_TILE = 500.0  # m; tree meshes are merged per tile so cameras can cull the rest
INTRUDER_RGB = (1.0, 0.15, 0.15)
rng = np.random.default_rng(5)


# -- helpers ----------------------------------------------------------------------------------

def fmt(v, nd=2):
    return " ".join(f"{x:.{nd}f}" for x in v)


def pbr(rgb, rough=0.9, emissive=None, intensity=1.0, transparency=0.0, texture=None):
    s = f"PBRAppearance {{ baseColor {fmt(rgb, 3)} roughness {rough} metalness 0"
    if texture:
        s += f' baseColorMap ImageTexture {{ url "{texture}" }}'
    if emissive:
        s += f" emissiveColor {fmt(emissive, 3)} emissiveIntensity {intensity}"
    if transparency:
        s += f" transparency {transparency}"
    return s + " }"


def photo_look(url: str) -> str:
    """An aerial photo as ground or roof: lit, but dimmed, since it carries its own sunlight."""
    return pbr((0.52, 0.52, 0.52), 1, texture=url)


def axis_angle_view(yaw_deg: float, pitch_down_deg: float) -> str:
    """Viewpoint orientation facing compass `yaw` (0 = north), tilted down by `pitch`."""
    yaw, pitch = math.radians(90.0 - yaw_deg), math.radians(pitch_down_deg)
    w1, z1 = math.cos(yaw / 2), math.sin(yaw / 2)
    w2, y2 = math.cos(pitch / 2), math.sin(pitch / 2)
    w, x, y, z = w1 * w2, -z1 * y2, w1 * y2, z1 * w2
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    return f"{x / n:.4f} {y / n:.4f} {z / n:.4f} {2 * math.acos(max(-1, min(1, w))):.4f}"


def densify(pts, step=20.0):
    out = [pts[0]]
    for a, b in itertools.pairwise(pts):
        n = max(1, int(math.dist(a, b) // step))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
    return out


# -- land use and textures --------------------------------------------------------------------

def landuse(raw: dict, city_dir: Path, city: City) -> np.ndarray:
    """Class raster, row 0 = north. From the City's land-use image, or drawn for flat cities."""
    if "landuse" in raw:
        return np.array(Image.open(city_dir / raw["landuse"]["image"]))
    x0, y0, x1, y1 = city.bounds
    n = 1024
    img = Image.new("L", (n, n), LANDUSE.index("meadow"))
    d = ImageDraw.Draw(img)

    def px(x, y):
        return ((x - x0) / (x1 - x0) * n, (1 - (y - y0) / (y1 - y0)) * n)

    for b in city.buildings:
        d.polygon([px(*p) for p in b.footprint.buffer(12).exterior.coords], fill=LANDUSE.index("urban"))
    for r in city.roads:
        d.line([px(*p) for p in r.coords], fill=LANDUSE.index("road"), width=max(2, int(9 / (x1 - x0) * n)))
    for r in city.rails:
        d.line([px(*p) for p in r.coords], fill=LANDUSE.index("rail"), width=max(2, int(5 / (x1 - x0) * n)))
    return np.array(img)


def paint_ground(lu: np.ndarray, size_m: float) -> Image.Image:
    """Land-use classes -> a natural-looking colour texture (TEXTURE_PX square)."""
    n = TEXTURE_PX
    cls = np.array(Image.fromarray(lu).resize((n, n), Image.NEAREST))
    m_per_px = size_m / n
    yy, xx = np.mgrid[0:n, 0:n] * m_per_px
    noise = np.array(Image.fromarray((rng.random((n // 8, n // 8)) * 255).astype(np.uint8))
                     .resize((n, n), Image.BICUBIC), dtype=float) / 255 - 0.5
    fine = rng.random((n, n)) - 0.5
    base = {
        "meadow": (108, 146, 72), "forest": (58, 78, 44), "urban": (138, 140, 128),
        "industrial": (150, 150, 146), "water": (52, 82, 100), "road": (72, 72, 74),
        "rail": (108, 98, 86), "field": (0, 0, 0), "runway": (96, 96, 98),
        "taxiway": (128, 128, 126), "apron": (150, 150, 146),
    }
    rgb = np.zeros((n, n, 3))
    for name, col in base.items():
        rgb[cls == LANDUSE.index(name)] = col
    # fields: rotated patchwork, each patch its own crop, rows along the patch
    crops = np.array([(196, 178, 102), (148, 168, 80), (126, 104, 72), (172, 160, 92),
                      (104, 140, 62), (182, 150, 90)], float)
    ang = math.radians(20)
    u = xx * math.cos(ang) + yy * math.sin(ang)
    v = -xx * math.sin(ang) + yy * math.cos(ang)
    pid = (np.floor(u / 140).astype(int) * 7919 + np.floor(v / 85).astype(int) * 104729) % len(crops)
    rows = 0.08 * np.sin(2 * np.pi * u / 4.0)
    field = cls == LANDUSE.index("field")
    rgb[field] = crops[pid[field]] * (1 + rows[field])[:, None]
    # variation: large-scale tint and fine grain, stronger on vegetation
    veg = np.isin(cls, [LANDUSE.index(c) for c in ("meadow", "forest", "field")])
    rgb *= (1 + np.where(veg, 0.22, 0.08)[..., None] * noise[..., None])
    rgb *= (1 + np.where(veg, 0.10, 0.05)[..., None] * fine[..., None])
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def paint_outer(n_px=2048) -> Image.Image:
    """Distant land: a fine forest / meadow / field mosaic, soft so the haze carries it."""
    def noise(cells):
        return np.array(Image.fromarray((rng.random((cells, cells)) * 255).astype(np.uint8))
                        .resize((n_px, n_px), Image.BICUBIC), dtype=float) / 255
    mix = 0.6 * noise(256) + 0.4 * noise(64)
    forest = np.array((70, 92, 58), float)
    meadow = np.array((122, 148, 88), float)
    field = np.array((160, 156, 104), float)
    t = np.clip((mix - 0.45) / 0.15, 0, 1)[..., None]
    rgb = meadow * (1 - t) + forest * t
    fields = (noise(128) > 0.62)[..., None] & (t < 0.5)
    rgb = np.where(fields, field, rgb)
    rgb = 0.6 * rgb + 0.4 * rgb.mean(axis=(0, 1))  # low contrast: it is seen through haze
    rgb *= 1 + 0.06 * (rng.random((n_px, n_px)) - 0.5)[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))


# -- meshes -----------------------------------------------------------------------------------

class Obj:
    def __init__(self):
        self.v: list[np.ndarray] = []
        self.f: list[np.ndarray] = []
        self.count = 0

    def add(self, verts: np.ndarray, faces: np.ndarray):
        self.v.append(np.asarray(verts, float))
        self.f.append(np.asarray(faces) + self.count + 1)
        self.count += len(verts)

    def write(self, path: Path):
        with open(path, "w") as fh:
            if self.v:
                np.savetxt(fh, np.vstack(self.v), fmt="v %.2f %.2f %.2f")
                np.savetxt(fh, np.vstack(self.f), fmt="f %d %d %d")


def cone(x, y, z0, z1, r, k=7):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(0, 1)
    ring = np.stack([x + r * np.cos(a), y + r * np.sin(a), np.full(k, z0)], 1)
    verts = np.vstack([ring, [[x, y, z1]]])
    faces = np.array([[i, (i + 1) % k, k] for i in range(k)])
    return verts, faces


_BLOB = None


def blob(x, y, zc, rx, rz):
    """Low-poly canopy: a squashed, slightly irregular octahedron subdivided once (32 faces)."""
    global _BLOB
    if _BLOB is None:
        t = [np.array(v, float) for v in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
        faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
        verts, mid, out = list(t), {}, []

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in mid:
                p = verts[i] + verts[j]
                verts.append(p / np.linalg.norm(p))
                mid[key] = len(verts) - 1
            return mid[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            out += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        _BLOB = (np.array(verts), np.array(out))
    v, f = _BLOB
    v = v * (1 + 0.12 * (rng.random((len(v), 1)) - 0.5))
    return v * [rx, rx, rz] + [x, y, zc], f


def trunk(x, y, z0, z1, r, k=3):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    lo = np.stack([x + r * np.cos(a), y + r * np.sin(a), np.full(k, z0)], 1)
    hi = lo.copy()
    hi[:, 2] = z1
    faces = [[i, (i + 1) % k, k + i] for i in range(k)] + [[(i + 1) % k, k + (i + 1) % k, k + i] for i in range(k)]
    return np.vstack([lo, hi]), np.array(faces)


def trees(lu: np.ndarray, city: City, assets: Path, forest_only: bool = False) -> list[tuple[str, str]]:
    """Forests and scattered trees on the ground, merged by species and shade within each
    TREE_TILE square, so a camera only draws the tiles in its view. Returns (species, file).
    forest_only: no scattered trees (an aerial photo already shows where they stand)."""
    x0, y0, x1, y1 = city.bounds
    nrow, ncol = lu.shape[:2]
    spacing = 9.5
    gx, gy = np.meshgrid(np.arange(x0 + 5, x1 - 5, spacing), np.arange(y0 + 5, y1 - 5, spacing))
    gx = gx + rng.uniform(-0.45, 0.45, gx.shape) * spacing
    gy = gy + rng.uniform(-0.45, 0.45, gy.shape) * spacing
    col = np.clip(((gx - x0) / (x1 - x0) * ncol).astype(int), 0, ncol - 1)
    row = np.clip(((1 - (gy - y0) / (y1 - y0)) * nrow).astype(int), 0, nrow - 1)
    cls = lu[row, col]
    keep_p = np.zeros(len(LANDUSE))
    for name, p in (("forest", 0.92), ("meadow", 0.015), ("field", 0.002), ("urban", 0.05),
                    ("industrial", 0.01)):
        keep_p[LANDUSE.index(name)] = p if name == "forest" or not forest_only else 0.0
    keep = rng.random(cls.shape) < keep_p[cls]
    xs, ys = gx[keep], gy[keep]
    if city.buildings and len(xs):  # no trees inside buildings
        tree = STRtree([b.footprint.buffer(2) for b in city.buildings])
        idx = tree.query([Point(x, y) for x, y in zip(xs, ys, strict=True)], predicate="intersects")
        inside = np.zeros(len(xs), bool)
        inside[idx[0]] = True
        xs, ys = xs[~inside], ys[~inside]
    zs = np.atleast_1d(city.ground(xs, ys)) if len(xs) else np.array([])
    zlo, zhi = (float(zs.min()), float(zs.max())) if len(zs) else (0.0, 1.0)
    meshes: dict[tuple[str, int, int], Obj] = {}
    for x, y, z in zip(xs, ys, zs, strict=True):
        tile = (int((x - x0) // TREE_TILE), int((y - y0) // TREE_TILE))
        mesh = lambda name, tile=tile: meshes.setdefault((name, *tile), Obj())
        conifer = rng.random() < 0.35 + 0.45 * (z - zlo) / max(zhi - zlo, 1.0)
        h = rng.uniform(11, 22)
        if conifer:
            name = "conifer_dark" if rng.random() < 0.5 else "conifer"
            r = h * rng.uniform(0.18, 0.24)
            mesh(name).add(*cone(x, y, z + 0.15 * h, z + 0.62 * h, r * 1.15))
            mesh(name).add(*cone(x, y, z + 0.45 * h, z + h, r * 0.8))
            mesh("trunk").add(*trunk(x, y, z - 0.5, z + 0.2 * h, 0.25))
        else:
            name = "broadleaf" if rng.random() < 0.55 else "broadleaf_light"
            mesh(name).add(*blob(x, y, z + 0.62 * h, h * rng.uniform(0.26, 0.34), h * 0.3))
            mesh("trunk").add(*trunk(x, y, z - 0.5, z + 0.45 * h, 0.3))
    for old in assets.glob("trees_*.obj"):  # earlier builds, other tilings
        old.unlink()
    out = []
    for (name, i, j), m in sorted(meshes.items()):
        path = assets / f"trees_{name}_{i}_{j}.obj"
        m.write(path)
        out.append((name, path.name))
    return out


def rail_meshes(city: City, assets: Path) -> None:
    ballast, steel, sleepers = Obj(), Obj(), Obj()
    for line in city.rails:
        pts = densify(list(line.coords), 4.0)
        for (ax, ay), (bx, by) in itertools.pairwise(pts):
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / L, dx / L
            za, zb = float(city.ground(ax, ay)), float(city.ground(bx, by))
            ring = []  # ballast: trapezoid strip (top 3 m, bottom 4.4 m, 0.45 m high)
            for x, y, z in ((ax, ay, za), (bx, by, zb)):
                ring += [(x - 2.2 * nx, y - 2.2 * ny, z - 0.1), (x - 1.5 * nx, y - 1.5 * ny, z + 0.35),
                         (x + 1.5 * nx, y + 1.5 * ny, z + 0.35), (x + 2.2 * nx, y + 2.2 * ny, z - 0.1)]
            ballast.add(ring, [[0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 6], [2, 6, 3], [3, 6, 7]])
            for side in (-0.72, 0.72):  # rails on top of the sleepers
                ox, oy, hw = side * nx, side * ny, 0.035
                v = [(ax + ox - hw * nx, ay + oy - hw * ny, za + 0.5), (ax + ox + hw * nx, ay + oy + hw * ny, za + 0.5),
                     (bx + ox - hw * nx, by + oy - hw * ny, zb + 0.5), (bx + ox + hw * nx, by + oy + hw * ny, zb + 0.5),
                     (ax + ox - hw * nx, ay + oy - hw * ny, za + 0.35), (bx + ox - hw * nx, by + oy - hw * ny, zb + 0.35)]
                steel.add(v, [[0, 2, 1], [1, 2, 3], [0, 4, 2], [2, 4, 5]])
            for k in range(int(L // 1.2)):  # sleepers every 1.2 m
                t = (k + 0.5) * 1.2 / L
                cx, cy, cz = ax + t * dx, ay + t * dy, za + t * (zb - za) + 0.42
                ux, uy = dx / L * 0.12, dy / L * 0.12
                v = [(cx - 1.25 * nx - ux, cy - 1.25 * ny - uy, cz), (cx + 1.25 * nx - ux, cy + 1.25 * ny - uy, cz),
                     (cx + 1.25 * nx + ux, cy + 1.25 * ny + uy, cz), (cx - 1.25 * nx + ux, cy - 1.25 * ny + uy, cz)]
                sleepers.add(v, [[0, 1, 2], [0, 2, 3]])
    ballast.write(assets / "rail_ballast.obj")
    steel.write(assets / "rail_steel.obj")
    sleepers.write(assets / "rail_sleepers.obj")


# -- aircraft at the stands -------------------------------------------------------------------

AIRLINERS = {  # length, span, fuselage radius (m): an A320-like and an A350-like twin
    "narrow": (37.6, 35.8, 2.0),
    "wide": (66.8, 64.8, 3.0),
}
TAIL_RGB = [(0.07, 0.12, 0.32), (0.72, 0.07, 0.08), (0.0, 0.38, 0.44), (0.96, 0.7, 0.08),
            (0.1, 0.32, 0.18), (0.55, 0.1, 0.45)]  # generic liveries, no real airline's
AIRCRAFT_LOOK = {"body": ((0.92, 0.92, 0.93), 0.6), "wing": ((0.74, 0.76, 0.78), 0.55),
                 "engine": ((0.62, 0.63, 0.66), 0.3), "dark": ((0.06, 0.07, 0.09), 0.2)}


def _slab(poly, z, thick, out: list, tilt=None):
    """A thin plate: polygon `poly` [(x, y)] at height z (or z per vertex via tilt(x, y))."""
    n = len(poly)
    top = [(x, y, (tilt(x, y) if tilt else z) + thick / 2) for x, y in poly]
    bot = [(x, y, (tilt(x, y) if tilt else z) - thick / 2) for x, y in poly]
    v = np.array(top + bot)
    f = [[0, i, i + 1] for i in range(1, n - 1)] + [[n, n + i + 1, n + i] for i in range(1, n - 1)]
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i]]
    f = np.array(f)
    out.append((v, np.vstack([f, f[:, ::-1]])))  # both windings: no culled faces at any angle


def _fin(poly_xz, y, thick, out: list):
    """A vertical plate: polygon [(x, z)] in the plane y."""
    _slab([(x, z) for x, z in poly_xz], 0, thick, out)
    v, f = out.pop()
    out.append((v[:, [0, 2, 1]] + [0, y, 0], f))  # swap y and z: the plate stands up


def _tube(stations, k, out: list):
    """A lofted tube along x: stations [(x, radius_y, radius_z, z_centre)]."""
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    rings = [np.stack([np.full(k, x), ry * np.cos(a), zc + rz * np.sin(a)], 1) for x, ry, rz, zc in stations]
    v = np.vstack(rings)
    f = []
    for s in range(len(stations) - 1):
        for i in range(k):
            j = (i + 1) % k
            p, q, r_, t = s * k + i, s * k + j, (s + 1) * k + j, (s + 1) * k + i
            f += [[p, r_, q], [p, t, r_]]
    f = np.array(f)
    out.append((v, np.vstack([f, f[:, ::-1]])))


def airliner(kind: str) -> dict[str, list]:
    """An airliner in local metres, nose along +x, wheels on z = 0: parts per material."""
    L, S, r = AIRLINERS[kind]
    parts: dict[str, list] = {m: [] for m in ("body", "wing", "engine", "dark", "tail")}
    zc = r + 0.18 * r + 1.0  # fuselage centre: belly ~1.4 m (narrow) above the apron
    nose, tail = L / 2, -L / 2
    prof = [(0.0, 0.0, 0.0), (0.012, 0.32, 0.0), (0.035, 0.62, -0.03), (0.07, 0.86, -0.02),
            (0.12, 0.98, 0.0), (0.18, 1.0, 0.0), (0.68, 1.0, 0.0), (0.8, 0.86, 0.12),
            (0.9, 0.58, 0.32), (0.97, 0.3, 0.48), (1.0, 0.14, 0.52)]  # (from nose, radius, rise)
    _tube([(nose - t * L, rr * r, rr * r * 1.04, zc + up * r) for t, rr, up in prof], 18, parts["body"])
    # cabin window band and cockpit windows
    for side in (-1, 1):
        y = side * (r * 1.002)
        _fin([(nose - 0.17 * L, zc + 0.28 * r), (tail + 0.27 * L, zc + 0.28 * r),
              (tail + 0.27 * L, zc + 0.36 * r), (nose - 0.17 * L, zc + 0.36 * r)], y, 0.02, parts["dark"])
        _fin([(nose - 0.035 * L, zc + 0.3 * r), (nose - 0.075 * L, zc + 0.3 * r),
              (nose - 0.075 * L, zc + 0.52 * r), (nose - 0.045 * L, zc + 0.48 * r)],
             side * r * 0.86, 0.02, parts["dark"])
    # wings: swept, low, with dihedral
    xr = nose - 0.40 * L  # root leading edge
    c_root, c_tip, sweep = 0.19 * L, 0.055 * L, 0.24 * S
    zw = zc - 0.62 * r
    for side in (-1, 1):
        tip = side * S / 2
        poly = [(xr, side * r * 0.8), (xr - sweep, tip), (xr - sweep - c_tip, tip), (xr - c_root, side * r * 0.8)]
        _slab(poly if side > 0 else poly[::-1], zw, 0.45 * r / 2, parts["wing"],
              tilt=lambda x, y: zw + 0.09 * abs(y))
        # winglet
        _fin([(xr - sweep - c_tip * 0.1, zw + 0.09 * S / 2), (xr - sweep - c_tip, zw + 0.09 * S / 2),
              (xr - sweep - c_tip * 1.05, zw + 0.09 * S / 2 + 0.07 * S), (xr - sweep - c_tip * 0.7,
              zw + 0.09 * S / 2 + 0.07 * S)], tip, 0.15, parts["tail"])
        # engine under the wing, ahead of it
        ey = side * 0.32 * S / 2
        ex = xr - sweep * 0.32 * 1.0 + 0.06 * L
        er, el = 0.56 * r, 0.12 * L
        ez = zw + 0.09 * abs(ey) - er * 1.05
        _tube([(ex, er * 0.92, er * 0.92, ez), (ex - 0.02 * L, er, er, ez), (ex - 0.8 * el, er * 0.9, er * 0.9, ez),
               (ex - el, er * 0.55, er * 0.55, ez)], 14, parts["engine"])
        _tube([(ex + 0.01, er * 0.8, er * 0.8, ez), (ex + 0.02, 0.01, 0.01, ez)], 14, parts["dark"])  # intake
    # tailplane and fin (fin in the livery colour)
    xt = tail + 0.16 * L
    for side in (-1, 1):
        poly = [(xt, side * r * 0.4), (xt - 0.1 * L, side * 0.17 * S), (xt - 0.13 * L, side * 0.17 * S),
                (xt - 0.12 * L, side * r * 0.4)]
        _slab(poly if side > 0 else poly[::-1], zc + 0.35 * r, 0.25, parts["wing"])
    fz = zc + 0.55 * r
    _fin([(tail + 0.2 * L, fz), (tail + 0.01 * L, fz), (tail - 0.02 * L, fz + 0.17 * L),
          (tail + 0.06 * L, fz + 0.17 * L)], 0.0, 0.5, parts["tail"])
    # landing gear: nose strut and two main legs with wheels
    for gx, gy in ((nose - 0.1 * L, 0.0), (xr - 0.1 * L, 0.18 * S / 2), (xr - 0.1 * L, -0.18 * S / 2)):
        _slab([(gx - 0.2, gy - 0.2), (gx + 0.2, gy - 0.2), (gx + 0.2, gy + 0.2), (gx - 0.2, gy + 0.2)],
              (zc - r) / 2 + 0.5, zc - r - 0.5, parts["dark"])
        _slab([(gx - 0.6, gy - 0.5), (gx + 0.6, gy - 0.5), (gx + 0.6, gy + 0.5), (gx - 0.6, gy + 0.5)],
              0.6, 1.1, parts["dark"])
    return parts


def stand_aircraft(raw: dict, city: City, assets: Path, tile=1000.0) -> list[tuple[str, str]]:
    """Park airliners nose-in on most OSM stands (the end of the lead-in line nearer a building
    is the nose), merged per tile and material so cameras cull them. -> [(material, file)]."""
    st = raw.get("stands") or []
    if not st:
        return []
    srng = np.random.default_rng(11)
    tree = STRtree([b.footprint for b in city.buildings])
    mids = np.array([np.mean(s["line"], axis=0) for s in st])
    models = {k: airliner(k) for k in AIRLINERS}
    objs: dict[tuple, Obj] = {}
    x0, y0 = city.bounds[:2]
    parked: list[Polygon] = []  # footprints so far: overlapping stands (MARS) hold one aircraft
    for s, mid in zip(st, mids, strict=True):
        if srng.random() > 0.78:
            continue  # an empty stand
        a, b = np.array(s["line"][0], float), np.array(s["line"][1], float)
        da = tree.query_nearest(Point(*a), return_distance=True)[1][0]
        db = tree.query_nearest(Point(*b), return_distance=True)[1][0]
        nose_pt, back_pt = (a, b) if da < db else (b, a)
        d = nose_pt - back_pt
        if np.linalg.norm(d) < 1:
            continue
        d /= np.linalg.norm(d)
        gap = np.sort(np.linalg.norm(mids - mid, axis=1))[1] if len(mids) > 1 else 999
        kind = "wide" if gap > 62 and srng.random() < 0.55 else "narrow"
        L = AIRLINERS[kind][0]
        c = nose_pt - d * (L / 2 + 4.0)  # nose a few metres short of the stop bar
        yaw = math.atan2(d[1], d[0])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        S = AIRLINERS[kind][1]
        foot = Polygon([c + R @ [sx * L / 2, sy * S / 2] for sx, sy in ((1, 0), (0, 1), (-1, 0), (0, -1))])
        if any(foot.intersects(q) for q in parked):
            continue
        parked.append(foot)
        z0 = float(city.ground(*c))
        livery = int(srng.integers(len(TAIL_RGB)))
        key_t = (int((c[0] - x0) // tile), int((c[1] - y0) // tile))
        for mat, plist in models[kind].items():
            name = f"tail{livery}" if mat == "tail" else mat
            o = objs.setdefault((*key_t, name), Obj())
            for v, f in plist:
                w = v.copy()
                w[:, :2] = v[:, :2] @ R.T + c
                w[:, 2] += z0
                o.add(w, f)
    out = []
    for (i, j, name), o in sorted(objs.items()):
        fname = f"aircraft_{name}_{i}_{j}.obj"
        o.write(assets / fname)
        out.append((name, fname))
    return out


def aircraft_look(name: str) -> str:
    if name.startswith("tail"):
        return pbr(TAIL_RGB[int(name[4:])], 0.5)
    rgb, rough = AIRCRAFT_LOOK[name]
    return pbr(rgb, rough)


def mesh_shape(url: str, rgb, rough=0.9) -> str:
    return f'Shape {{ appearance {pbr(rgb, rough)} geometry Mesh {{ url "{url}" }} castShadows FALSE }}'


def photo_roofs(obj: Path, tiles: list[dict], city_dir: Path, out: Path, m_per_px=0.5) -> None:
    """Drape the aerial photo over a roof mesh: crop the photo under the mesh's footprint
    (it may straddle ground tiles) into out.jpg and rewrite out.obj with planar UVs."""
    verts, faces = [], []
    for line in obj.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append(line.split()[1:])
    xy = np.asarray(verts)[:, :2]
    bx0, by0 = xy.min(axis=0) - 2.0
    bx1, by1 = xy.max(axis=0) + 2.0
    w, h = min(2048, _pow2((bx1 - bx0) / m_per_px)), min(2048, _pow2((by1 - by0) / m_per_px))
    sx, sy = w / (bx1 - bx0), h / (by1 - by0)
    crop = Image.new("RGB", (w, h), (128, 128, 128))
    for t in tiles:
        tx0, ty0, tx1, ty1 = t["bounds"]
        if tx1 <= bx0 or tx0 >= bx1 or ty1 <= by0 or ty0 >= by1:
            continue
        im = Image.open(city_dir / t["image"])
        crop.paste(im.resize((round((tx1 - tx0) * sx), round((ty1 - ty0) * sy)), Image.LANCZOS),
                   (round((tx0 - bx0) * sx), round((by1 - ty1) * sy)))
    crop.save(out.with_suffix(".jpg"), quality=85)
    uv = (xy - [bx0, by0]) / [bx1 - bx0, by1 - by0]
    lines_ = [f"v {x} {y} {z}" for x, y, z in verts] + [f"vt {u:.5f} {v:.5f}" for u, v in uv]
    lines_ += ["f " + " ".join(f"{i}/{i}" for i in f) for f in faces]
    out.with_suffix(".obj").write_text("\n".join(lines_) + "\n")


FACADE_W, FACADE_H = 6.0, 4.0  # m of wall per facade texture repeat: one storey, four panes


def facade_texture(out: Path) -> None:
    """A storey of curtain wall: a concrete spandrel under a band of tinted glass with mullions."""
    n = 512
    img = np.empty((n, n, 3), float)
    img[:] = (0.30, 0.36, 0.42)  # glass
    img[:, :] *= np.linspace(1.15, 0.85, n)[:, None, None]  # sky reflection, brighter at the top
    sill = int(n * 0.28)
    img[n - sill:] = (0.72, 0.71, 0.68)  # spandrel (image rows run top-down)
    img[n - sill - 6:n - sill] = (0.55, 0.55, 0.55)
    for k in range(4):  # mullions every 1.5 m
        c = int(k * n / 4)
        img[:n - sill, max(0, c - 3):c + 3] = (0.58, 0.59, 0.6)
    img += rng.normal(0, 0.015, img.shape)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out)


def facade_walls(obj: Path, out: Path) -> None:
    """Give every wall triangle UVs in metres along the wall and up it, for facade_texture."""
    verts = [[float(v) for v in ln.split()[1:4]] for ln in obj.read_text().splitlines() if ln.startswith("v ")]
    p = np.asarray(verts).reshape(-1, 3, 3)
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    t = np.stack([-nrm[:, 1], nrm[:, 0]], axis=1)  # horizontal, along the wall
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    u = np.einsum("tvk,tk->tv", p[:, :, :2], t) / FACADE_W
    v = p[:, :, 2] / FACADE_H
    lines_ = [f"v {x} {y} {z}" for x, y, z in verts]
    lines_ += [f"vt {a:.4f} {b:.4f}" for a, b in zip(u.ravel(), v.ravel(), strict=True)]
    lines_ += [f"f {k}/{k} {k + 1}/{k + 1} {k + 2}/{k + 2}" for k in range(1, len(verts) + 1, 3)]
    out.write_text("\n".join(lines_) + "\n")


def _pow2(px: float) -> int:
    return 1 << max(0, round(math.log2(max(px, 1.0))))


# -- world parts ------------------------------------------------------------------------------

def elevation(origin, spacing, heights2d, url) -> str:
    n = heights2d.shape[0]
    hs = " ".join(f"{h:.1f}" for h in heights2d.ravel())
    return (f"Pose {{ translation {origin[0]} {origin[1]} 0 children [ Shape {{ appearance "
            f"{pbr((1, 1, 1), 1.0, texture=url)} geometry ElevationGrid {{ xDimension {n} "
            f"xSpacing {spacing} yDimension {n} ySpacing {spacing} height [ {hs} ] }} castShadows FALSE }} ] }}")


def outer_ring(raw: dict, city: City) -> np.ndarray | None:
    t = raw.get("terrain_outer")
    if not t:
        return None
    n = int(t["n"])
    H = np.asarray(t["heights"], float).reshape(n, n)
    xs = t["origin"][0] + t["spacing"] * np.arange(n)
    X, Y = np.meshgrid(xs, xs)
    x0, y0, x1, y1 = city.bounds
    inside = (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
    # outside: blend from the inner terrain's edge height into the coarse hills (no seam cliff)
    cx, cy = np.clip(X, x0, x1), np.clip(Y, y0, y1)
    edge = np.asarray(city.ground(cx, cy))
    w = np.clip(np.hypot(X - cx, Y - cy) / 800.0, 0, 1) ** 2
    H = edge * (1 - w) + H * w
    return np.where(inside, edge - 6.0, H)  # hidden below the detailed terrain


def building(b: dict, k: int) -> str:
    c = Polygon(b["footprint"]).centroid
    pts = [(x - c.x, y - c.y) for x, y in b["footprint"]]
    if Polygon(pts).exterior.is_ccw:
        pts = pts[::-1]  # SimpleBuilding wants clockwise corners
    shape = b.get("roof", "flat roof")
    flat = shape == "flat roof"
    material = "gravel" if flat else ("old tiles" if k % 3 == 0 else "tiled")
    floors = int(b.get("floors", max(1, round(b["height"] / 3))))
    return (f'SimpleBuilding {{ translation {c.x:.2f} {c.y:.2f} {b.get("base_z", 0.0):.2f} name "bld{k}" '
            f'floorHeight {b.get("floor_height", 3.0)} floorNumber {floors} '
            f'corners [ {", ".join(f"{x:.2f} {y:.2f}" for x, y in pts)} ] '
            f'wallType "{b.get("wall", "concrete building")}" roofType "{material}" '
            f'roofShape "{shape}" roofHeight {1.0 if flat else 4.0} bottom TRUE enableBoundingObject FALSE }}')


def road(pts, width, city: City, k: int, town: bool) -> str:
    dense = densify(pts, 20.0)
    wp = ", ".join(f"{x:.1f} {y:.1f} {float(city.ground(x, y)) + 0.4 + 0.01 * (k % 5):.2f}" for x, y in dense)
    borders = "TRUE" if town else "FALSE"
    return (f'Road {{ name "road{k}" width {width} numberOfLanes 2 wayPoints [ {wp} ] '
            f"splineSubdivision 1 rightBorder {borders} leftBorder {borders} roadBorderWidth [ 1.5 ] }}")


def marker(rgb) -> str:
    return (f"Pose {{ children [ Shape {{ appearance {pbr(rgb, 1, rgb, 1.5, 0.5)} geometry Cylinder {{ "
            f"radius 1.3 height 0.04 subdivision 24 }} castShadows FALSE }} Pose {{ translation 0 0 14 children [ "
            f"Shape {{ appearance {pbr(rgb, 1, rgb, 1.5, 0.55)} geometry Cylinder {{ radius 0.1 height 28 "
            f"subdivision 8 }} castShadows FALSE }} ] }} ] }}")


QUAD_LOOKS = {  # body colour, scale (1 = 0.9 m motor to motor across the diagonal)
    "SWARM": ((1.0, 0.32, 0.02), 1.0),  # signal orange: red stays the intruder's colour
    "INTRUDER": ((0.13, 0.13, 0.14), 1.3),  # a larger, dark drone
}


def _metal(rgb, rough=0.35) -> str:
    return f"PBRAppearance {{ baseColor {fmt(rgb, 3)} roughness {rough} metalness 0.9 }}"


def quad_model(kind: str) -> str:
    """A quadcopter from primitives, x forward: body shell, carbon arms, motors, blurred
    props, nav lights (red left, green right, white tail), a gimbal camera under the nose.
    Nothing hangs below the body forward of its centre: the Verifier camera sits there."""
    rgb, k = QUAD_LOOKS[kind]
    shell = pbr(rgb, 0.35, rgb if kind == "SWARM" else None, 0.25)
    carbon, white = pbr((0.05, 0.05, 0.06), 0.45), pbr((0.93, 0.93, 0.93), 0.3)

    def at(x, y, z, node, rot="0 0 1 0"):
        return f"Pose {{ translation {x * k:.4f} {y * k:.4f} {z * k:.4f} rotation {rot} children [ {node} ] }}"

    def shape(look, geom):
        return f"Shape {{ appearance {look} geometry {geom} castShadows FALSE }}"

    def ellipsoid(look, rx, ry, rz):
        return (f"Transform {{ scale {rx * k:.4f} {ry * k:.4f} {rz * k:.4f} children [ "
                f"{shape(look, 'Sphere { radius 1 subdivision 3 }')} ] }}")

    parts = [
        ellipsoid(shell, 0.17, 0.095, 0.06),  # streamlined shell
        at(-0.01, 0, 0.035, ellipsoid(shell, 0.12, 0.075, 0.04)),  # canopy hump
        at(-0.05, 0, 0.07, shape(carbon, f"Box {{ size {0.12 * k:.4f} {0.08 * k:.4f} {0.014 * k:.4f} }}")),
        at(0.06, 0, 0.068, shape(white, f"Box {{ size {0.035 * k:.4f} {0.09 * k:.4f} {0.004 * k:.4f} }}")),
        at(0.15, 0, -0.035, shape(carbon, f"Sphere {{ radius {0.035 * k:.4f} subdivision 2 }}")),
        at(0.18, 0, -0.035, shape(_metal((0.1, 0.15, 0.25), 0.05),
                                  f"Sphere {{ radius {0.015 * k:.4f} subdivision 2 }}")),
        at(-0.08, 0.07, -0.1, shape(carbon, f"Box {{ size {0.2 * k:.4f} {0.012 * k:.4f} {0.012 * k:.4f} }}")),
        at(-0.08, -0.07, -0.1, shape(carbon, f"Box {{ size {0.2 * k:.4f} {0.012 * k:.4f} {0.012 * k:.4f} }}")),
        at(-0.08, 0.07, -0.05, shape(carbon, f"Box {{ size {0.012 * k:.4f} {0.012 * k:.4f} {0.1 * k:.4f} }}")),
        at(-0.08, -0.07, -0.05, shape(carbon, f"Box {{ size {0.012 * k:.4f} {0.012 * k:.4f} {0.1 * k:.4f} }}")),
    ]
    r = 0.32  # arm length from the centre
    for a_deg, light in ((45, (1, 0.05, 0.05)), (-45, (0.1, 1, 0.2)), (135, None), (-135, None)):
        a = math.radians(a_deg)
        c, s_ = math.cos(a), math.sin(a)
        parts.append(at(c * r / 2, s_ * r / 2, 0.01, shape(carbon, f"Cylinder {{ radius {0.02 * k:.4f} "
                        f"height {r * k:.4f} subdivision 10 }}"), f"{-s_:.4f} {c:.4f} 0 1.5708"))
        parts.append(at(c * r, s_ * r, 0.03, shape(_metal((0.2, 0.2, 0.22)), f"Cylinder {{ radius "
                        f"{0.032 * k:.4f} height {0.05 * k:.4f} subdivision 14 }}")))
        parts.append(at(c * r, s_ * r, 0.052, shape(shell, f"Cylinder {{ radius {0.034 * k:.4f} "
                        f"height {0.006 * k:.4f} subdivision 14 }}")))  # motor cap in body colour
        parts.append(at(c * r, s_ * r, 0.058, shape(pbr((0.85, 0.85, 0.88), 0.3, transparency=0.78),
                        f"Cylinder {{ radius {0.17 * k:.4f} height {0.003 * k:.4f} subdivision 24 }}")))
        parts.append(at(c * r, s_ * r, 0.06, shape(carbon, f"Box {{ size {0.3 * k:.4f} {0.025 * k:.4f} "
                        f"{0.004 * k:.4f} }}"), f"0 0 1 {a + 0.7:.4f}"))
        if light:
            parts.append(at(c * (r + 0.04), s_ * (r + 0.04), 0.02, shape(
                pbr(light, 1, light, 4), f"Sphere {{ radius {0.018 * k:.4f} subdivision 1 }}")))
    parts.append(at(-0.13, 0, 0.03, shape(pbr((1, 1, 1), 1, (1, 1, 1), 4),
                                         f"Sphere {{ radius {0.016 * k:.4f} subdivision 1 }}")))
    return f"DEF QUAD_{kind} Pose {{ children [ {' '.join(parts)} ] }}"


_defined_quads: set[str] = set()


def quad(kind: str) -> str:
    """The model on first use in a world, USE after that (all drones share one geometry)."""
    if kind in _defined_quads:
        return f"USE QUAD_{kind}"
    _defined_quads.add(kind)
    return quad_model(kind)


def swarm_drone(def_name, name, x, y, z) -> str:
    return f"DEF {def_name} Pose {{ translation {x:.1f} {y:.1f} {z:.1f} children [ {quad('SWARM')} ] }}"


def intruder_drone(def_name, name, markers) -> str:
    """Pose (moved by the backend) > recognisable Solid with the body, plus the red marker
    outside that Solid so the beam never counts as part of what the Verifier recognises."""
    slot = marker(INTRUDER_RGB) if markers else ""
    return (f'DEF {def_name} Pose {{ translation 0 0 -100 children [ Solid {{ name "{name}" '
            f"recognitionColors [ {fmt(INTRUDER_RGB, 2)} ] children [ {quad('INTRUDER')} ] }} {slot} ] }}")


def wagon(def_name, name, k) -> str:
    if k == 0:  # locomotive
        body = (f"Pose {{ translation 0 0 2.4 children [ Shape {{ appearance {pbr((0.62, 0.12, 0.1), 0.5)} "
                f"geometry Box {{ size 18 3 3.4 }} }} ] }} Pose {{ translation 0 0 4.3 children [ Shape {{ "
                f"appearance {pbr((0.2, 0.2, 0.22), 0.6)} geometry Box {{ size 16 2.6 0.4 }} }} ] }}")
    else:
        colors = [(0.12, 0.3, 0.55), (0.6, 0.45, 0.15), (0.2, 0.45, 0.25), (0.55, 0.55, 0.55), (0.5, 0.2, 0.15)]
        body = (f"Pose {{ translation 0 0 1.2 children [ Shape {{ appearance {pbr((0.25, 0.25, 0.27), 0.7)} "
                f"geometry Box {{ size 19 2.9 0.6 }} }} ] }} Pose {{ translation 0 0 2.8 children [ Shape {{ "
                f"appearance {pbr(colors[k % len(colors)], 0.6)} geometry Box {{ size 12.2 2.45 2.6 }} }} ] }}")
    return (f'DEF {def_name} Solid {{ translation 0 0 -100 name "{name}" '
            f"recognitionColors [ 0.6 0.4 0.2 ] children [ {body} ] }}")


def source_nodes(sid: str, cls: str, markers: bool) -> list[str]:
    if cls == "DRONE":
        return [intruder_drone(f"SRC_{sid}", f"src_{sid}", markers)]
    if cls == "VEHICLE":
        return [(f'DEF SRC_{sid} TruckSimple {{ translation 0 0 -100 name "src_{sid}" '
                f"color 0.55 0.3 0.12 recognitionColors [ 1 0.6 0.1 ] }}")]
    if cls == "TRAIN":
        return [wagon(f"SRC_{sid}_w{k}", f"src_{sid}_w{k}", k) for k in range(TRAIN_WAGONS)]
    return [(f'DEF SRC_{sid} Solid {{ translation 0 0 -100 name "src_{sid}" recognitionColors [ 0.7 0.7 0.7 ] '
            f"children [ Shape {{ appearance {pbr((0.62, 0.63, 0.64), 0.5)} geometry Box {{ size 2.4 1.6 1.4 }} }} "
            f"Pose {{ translation 0 0 0.75 children [ Shape {{ appearance {pbr((0.2, 0.2, 0.2), 0.5)} "
            f"geometry Cylinder {{ radius 0.6 height 0.1 }} }} ] }} ] }}")]


# -- main -------------------------------------------------------------------------------------

def build(scenario_path: str, out_path: str, markers: bool = True, force: bool = False) -> Path:
    sc = Scenario.load(scenario_path)
    _defined_quads.clear()  # DEF the drone models afresh in every world
    city_file = Path(sc.city) if Path(sc.city).is_absolute() else ROOT / sc.city
    raw = json.loads(city_file.read_text())
    city = City(raw)
    out = Path(out_path)
    assets = out.with_name(out.stem + "_assets")
    assets.mkdir(parents=True, exist_ok=True)
    side = [t["image"] for t in raw.get("ground_tiles", [])]  # files the City JSON points to
    side += [f for m in raw.get("building_meshes", []) for f in m.values()]
    key = hashlib.sha256(city_file.read_bytes() + Path(scenario_path).read_bytes()
                         + b"".join((city_file.parent / f).read_bytes() for f in side)
                         + Path(__file__).read_bytes()  # and the backend's node layout:
                         + f"{markers}{BUILDER_VERSION}{LIVE_SLOTS}{N_VIEWS}{TRAIN_WAGONS}".encode()).hexdigest()
    stamp = assets / "build.hash"
    if not force and out.exists() and stamp.exists() and stamp.read_text() == key:
        print(f"{out}: up to date")
        return out

    x0, y0, x1, y1 = city.bounds
    lu = landuse(raw, city_file.parent, city)
    photo = raw.get("ground_tiles")  # a real aerial photo instead of the painted land use
    if photo:
        for t in photo:
            (assets / Path(t["image"]).name).write_bytes((city_file.parent / t["image"]).read_bytes())
    else:
        paint_ground(lu, x1 - x0).save(assets / "ground.png")
    meshes = raw.get("building_meshes")  # real wall and roof geometry (LoD2) per tile
    for m in meshes or []:
        for kind, f in m.items():
            if kind == "roofs" and photo:
                photo_roofs(city_file.parent / f, photo, city_file.parent, assets / Path(f).name)
            elif kind == "walls":
                facade_walls(city_file.parent / f, assets / Path(f).name)
            else:
                (assets / Path(f).name).write_bytes((city_file.parent / f).read_bytes())
    if meshes:
        facade_texture(assets / "facade.png")
    tree_tiles = trees(lu, city, assets, forest_only=bool(photo))
    rail_meshes(city, assets)
    aircraft = stand_aircraft(raw, city, assets)
    rel = assets.name

    parts = ["#VRML_SIM R2025a utf8",
             f"# generated by tools/build_webots_world.py from {scenario_path} -- do not edit by hand"]
    parts += [f'EXTERNPROTO "{PROTO}/{p}"' for p in EXTERNPROTOS]
    span = max(x1 - x0, y1 - y0)
    cx = (x0 + x1) / 2
    parts += [
        'WorldInfo { info [ "Earshot" ] basicTimeStep 20 gravity 0 }',
        (f"DEF VIEW Viewpoint {{ orientation {axis_angle_view(0, 14)} position {cx:.0f} "
        f"{y0 - 0.45 * span:.0f} {float(city.ground(cx, y0)) + 0.22 * span:.0f} far 25000"
        f"{' near 1' if photo else ''} }}"),
        'TexturedBackground { texture "noon_cloudy_countryside" }',
        'TexturedBackgroundLight { texture "noon_cloudy_countryside" castShadows FALSE }',
        f"Fog {{ color 0.78 0.82 0.87 visibilityRange {1.5 * span:.0f} }}",
    ]
    if "terrain" in raw:
        t = raw["terrain"]
        n = int(t["n"])
        parts.append(elevation(t["origin"], t["spacing"], np.asarray(t["heights"]).reshape(n, n),
                               f"{rel}/ground.png"))
        outer = outer_ring(raw, city)
        if outer is not None:
            paint_outer().save(assets / "outer.png")
            to = raw["terrain_outer"]
            parts.append(elevation(to["origin"], to["spacing"], outer, f"{rel}/outer.png"))
    elif photo:  # flat city on its aerial photo, one textured plate per tile
        for t in photo:
            tx0, ty0, tx1, ty1 = t["bounds"]
            img = f"{rel}/{Path(t['image']).name}"
            parts.append(f"Pose {{ translation {(tx0 + tx1) / 2} {(ty0 + ty1) / 2} -0.05 children [ Shape {{ "
                         f"appearance {photo_look(img)} geometry Box {{ size {tx1 - tx0} "
                         f"{ty1 - ty0} 0.1 }} castShadows FALSE }} ] }}")
        # grass around the map as a frame, not under it: coplanar plates z-fight at a distance
        m = 2.5 * span
        for fx0, fy0, fx1, fy1 in ((x0 - m, y1, x1 + m, y1 + m), (x0 - m, y0 - m, x1 + m, y0),
                                   (x0 - m, y0, x0, y1), (x1, y0, x1 + m, y1)):
            parts.append(f"Pose {{ translation {(fx0 + fx1) / 2} {(fy0 + fy1) / 2} -0.05 children [ Shape {{ "
                         f"appearance {pbr((0.42, 0.55, 0.3), 1)} geometry Box {{ size {fx1 - fx0} "
                         f"{fy1 - fy0} 0.1 }} castShadows FALSE }} ] }}")
    else:  # flat city: a textured ground plane, wider grass around it
        parts.append(f"Pose {{ translation {cx} {(y0 + y1) / 2} -0.05 children [ Shape {{ appearance "
                     f"{pbr((1, 1, 1), 1, texture=f'{rel}/ground.png')} geometry Box {{ size {x1 - x0} {y1 - y0} 0.1 }} }} ] }}")
        parts.append(f"Pose {{ translation {cx} {(y0 + y1) / 2} -0.2 children [ Shape {{ appearance "
                     f"{pbr((0.42, 0.55, 0.3), 1)} geometry Box {{ size {6 * span} {6 * span} 0.1 }} }} ] }}")
    for w in raw.get("water", []):
        parts.append(f"Pose {{ translation {w['x']} {w['y']} {w['z']} children [ Shape {{ appearance "
                     f"{pbr((0.12, 0.25, 0.33), 0.05, transparency=0.1)} geometry Cylinder {{ radius {w['r'] + 30} "
                     f"height 0.1 subdivision 64 }} castShadows FALSE }} ] }}")
    tree_rgb = {"conifer_dark": (0.10, 0.22, 0.12), "conifer": (0.14, 0.29, 0.14),
                "broadleaf": (0.20, 0.36, 0.13), "broadleaf_light": (0.30, 0.42, 0.16),
                "trunk": (0.30, 0.22, 0.15)}
    defined = set()  # one appearance per species, shared by its tiles
    for name, file in tree_tiles:
        look = f"USE TREE_{name}" if name in defined else f"DEF TREE_{name} {pbr(tree_rgb[name])}"
        defined.add(name)
        parts.append(f'Shape {{ appearance {look} geometry Mesh {{ url "{rel}/{file}" }} castShadows FALSE }}')
    if city.rails:
        parts += [mesh_shape(f"{rel}/rail_ballast.obj", (0.42, 0.38, 0.34), 1.0),
                  mesh_shape(f"{rel}/rail_sleepers.obj", (0.28, 0.2, 0.14), 1.0),
                  mesh_shape(f"{rel}/rail_steel.obj", (0.55, 0.55, 0.58), 0.3)]
    widths = raw.get("road_widths", [8.0] * len(city.roads))
    town_discs = [Point(t["x"], t["y"]).buffer(t["r"]) for t in raw.get("towns", [])]
    for k, (r, wdt) in enumerate(zip(city.roads, widths, strict=True)):
        if photo:
            break  # the photo shows the roads
        mid = r.interpolate(0.5, normalized=True)
        parts.append(road(list(r.coords), wdt, city, k, any(d.contains(mid) for d in town_discs)))
    if meshes:
        defined_look = set()  # one appearance per kind, DEF on first use
        for m in meshes:
            for kind, rgb in (("walls", (1, 1, 1)), ("roofs", (0.36, 0.37, 0.39))):
                if kind not in m:
                    continue
                if kind == "roofs" and photo:
                    look = photo_look(f"{rel}/{Path(m[kind]).stem}.jpg")
                else:
                    tex = f"{rel}/facade.png" if kind == "walls" else None
                    look = (f"USE {kind.upper()}" if kind in defined_look
                            else f"DEF {kind.upper()} {pbr(rgb, 0.6, texture=tex)}")
                    defined_look.add(kind)
                parts.append(f'Shape {{ appearance {look} geometry Mesh {{ url "{rel}/{Path(m[kind]).name}" }} '
                             f"castShadows FALSE }}")
    else:
        for k, b in enumerate(raw["buildings"]):
            parts.append(building(b, k))
    looks = set()  # one appearance per aircraft material, DEF on first use
    for name, file in aircraft:
        look = f"USE AC_{name.upper()}" if name in looks else f"DEF AC_{name.upper()} {aircraft_look(name)}"
        looks.add(name)
        parts.append(f'Shape {{ appearance {look} geometry Mesh {{ url "{rel}/{file}" }} castShadows FALSE }}')

    for d in sc.drones:
        z = float(city.ground(d.x, d.y)) + 40.0
        parts.append(swarm_drone(f"DRONE_{d.id}", f"drone_{d.id}", d.x, d.y, z))
    slots = [(s.id, s.cls.name) for s in sc.sources] + [
        (live_slot(c, k), c) for c in ("DRONE", "VEHICLE", "TRAIN", "NOTHING") for k in range(LIVE_SLOTS)]
    for sid, cls in slots:
        parts += source_nodes(sid, cls, markers)
    # the RIG stays at the origin; its child Solids carry the cameras and are moved by the
    # supervisor: CAMV (the Verifier's camera, with recognition) and CAM_0..3 (game screen views)
    views = " ".join(
        f'DEF CAM_{k} Solid {{ translation 0 0 -100 name "cam{k}" children [ Camera {{ name "view{k}" '
        f'width {960 if k == 0 else 384} height {540 if k == 0 else 216} fieldOfView 1.15 near 0.05 far 9000 '
        f'antiAliasing TRUE ambientOcclusionRadius 2 bloomThreshold 6 }} ] }}'
        for k in range(N_VIEWS))
    parts.append(
        'DEF RIG Robot { translation 0 0 0 name "supervisor" supervisor TRUE controller "<extern>" '
        'children [ DEF CAMV Solid { translation 0 0 -100 name "camv" children [ Camera { name "camera" '
        "width 400 height 240 fieldOfView 1.0 far 3000 recognition Recognition { maxRange 100 "
        f"segmentation TRUE }} }} ] }} {views} ] }}"
    )
    out.write_text("\n".join(parts) + "\n")
    stamp.write_text(key)
    print(f"{out}: {len(raw['buildings'])} buildings, {len(city.roads)} roads, {len(tree_tiles)} tree meshes, "
          f"{len(sc.drones)} drones, {len(sc.sources)} sources (assets in {assets.name}/)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="scenarios/valley.yaml")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--no-markers", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    name = Scenario.load(a.scenario).name
    build(a.scenario, a.out or f"webots/worlds/{name}.wbt", markers=not a.no_markers, force=a.force)

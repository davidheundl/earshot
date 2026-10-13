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
import shapely
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


def trees(lu: np.ndarray, city: City, assets: Path, forest_only: bool = False,
          extra=None, far=None) -> list[tuple[str, str]]:
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
    if extra is not None and len(extra):  # single trees and tree rows mapped in OSM
        ex = np.asarray(extra, float)
        xs, ys = np.concatenate([xs, ex[:, 0]]), np.concatenate([ys, ex[:, 1]])
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
    for x, y in far or []:  # distant forest: one five-sided cone each, only the canopy line shows
        z = float(city.ground(x, y))
        tile = (int((x - x0) // (4 * TREE_TILE)) * 4, int((y - y0) // (4 * TREE_TILE)) * 4)
        h = rng.uniform(13, 22)
        name = "conifer_dark" if rng.random() < 0.5 else "broadleaf"
        meshes.setdefault((name, *tile), Obj()).add(*cone(x, y, z, z + h, h * 0.28, 5))
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


def stand_aircraft(raw: dict, city: City, assets: Path, tile=1000.0) -> tuple[list[tuple[str, str]], list[dict]]:
    """Park airliners nose-in on most OSM stands (the end of the lead-in line nearer a building
    is the nose), merged per tile and material so cameras cull them. Returns the mesh files
    [(material, file)] and every stand [{foot, nose, dir, kind, parked}] (the photo is cleaned
    under each, jet bridges reach the parked ones)."""
    st = raw.get("stands") or []
    if not st:
        return [], []
    srng = np.random.default_rng(11)
    tree = STRtree([b.footprint for b in city.buildings])
    mids = np.array([np.mean(s["line"], axis=0) for s in st])
    models = {k: airliner(k) for k in AIRLINERS}
    objs: dict[tuple, Obj] = {}
    x0, y0 = city.bounds[:2]
    parked: list[Polygon] = []  # footprints so far: overlapping stands (MARS) hold one aircraft
    stands = []
    for s, mid in zip(st, mids, strict=True):
        occupied = srng.random() <= 0.78
        a, b = np.array(s["line"][0], float), np.array(s["line"][1], float)
        da = tree.query_nearest(Point(*a), return_distance=True)[1][0]
        db = tree.query_nearest(Point(*b), return_distance=True)[1][0]
        nose_pt, back_pt = (a, b) if da < db else (b, a)
        d = nose_pt - back_pt
        if np.linalg.norm(d) < 1:
            continue
        d /= np.linalg.norm(d)
        gap = np.sort(np.linalg.norm(mids - mid, axis=1))[1] if len(mids) > 1 else 999
        if occupied:
            kind = "wide" if gap > 62 and srng.random() < 0.55 else "narrow"
        else:
            kind = "wide" if gap > 62 else "narrow"
        L, S = AIRLINERS[kind][:2]
        c = nose_pt - d * (L / 2 + 4.0)  # nose a few metres short of the stop bar
        yaw = math.atan2(d[1], d[0])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        foot = Polygon([c + R @ [sx * L / 2, sy * S / 2] for sx, sy in ((1, 0), (0, 1), (-1, 0), (0, -1))])
        info = {"foot": Polygon([c + R @ [sx * (L / 2 + 6), sy * (S / 2 + 6)]
                                 for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]),
                "nose": c + d * L / 2, "dir": d, "kind": kind, "parked": False}
        stands.append(info)
        if not occupied or any(foot.intersects(q) for q in parked):
            continue
        parked.append(foot)
        info["parked"] = True
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
    return out, stands


# -- small objects from OSM: what the photo shows flat, built as 3D ----------------------------

PROP_LOOK = {  # material -> (rgb, roughness, extra PBRAppearance fields)
    "car0": ((0.85, 0.85, 0.86), 0.35, ""), "car1": ((0.08, 0.08, 0.09), 0.3, ""),
    "car2": ((0.45, 0.47, 0.5), 0.3, ""), "car3": ((0.55, 0.06, 0.06), 0.3, ""),
    "car4": ((0.1, 0.2, 0.42), 0.3, ""), "car5": ((0.62, 0.64, 0.66), 0.3, ""),
    "glass": ((0.06, 0.08, 0.1), 0.1, ""), "tyre": ((0.03, 0.03, 0.03), 0.9, ""),
    "hedge": ((0.15, 0.26, 0.1), 0.95, ""), "wall": ((0.63, 0.61, 0.57), 0.9, ""),
    "post": ((0.42, 0.44, 0.46), 0.5, " metalness 0.6"),
    "fence": ((0.6, 0.62, 0.64), 0.5, " metalness 0.5 transparency 0.6"),
    "mast": ((0.6, 0.61, 0.63), 0.45, " metalness 0.7"),
    "lamp": ((1.0, 0.96, 0.85), 0.5, " emissiveColor 1 0.96 0.85 emissiveIntensity 2"),
    "steel": ((0.5, 0.52, 0.53), 0.5, " metalness 0.7"), "cable": ((0.1, 0.1, 0.1), 0.6, ""),
    "bridge": ((0.8, 0.81, 0.82), 0.45, " metalness 0.3"), "bridge_glass": ((0.22, 0.3, 0.36), 0.15, ""),
    "gse_tug": ((0.95, 0.75, 0.1), 0.5, ""), "gse_loader": ((0.9, 0.9, 0.88), 0.5, ""),
    "gse_cart": ((0.55, 0.56, 0.58), 0.6, ""), "gse_fuel": ((0.85, 0.85, 0.85), 0.4, " metalness 0.4"),
}


def prop_look(name: str) -> str:
    rgb, rough, extra = PROP_LOOK[name]
    return f"PBRAppearance {{ baseColor {fmt(rgb, 3)} roughness {rough} metalness 0{extra} }}".replace(
        " metalness 0 metalness", " metalness")


class MeshSet:
    """Meshes merged per material and per `tile` square, so cameras cull whole tiles."""

    def __init__(self, prefix: str, origin, tile=1000.0):
        self.prefix, self.origin, self.tile = prefix, origin, tile
        self.objs: dict[tuple, Obj] = {}

    def add(self, mat: str, x: float, y: float, verts, faces):
        key = (mat, int((x - self.origin[0]) // self.tile), int((y - self.origin[1]) // self.tile))
        self.objs.setdefault(key, Obj()).add(verts, faces)

    def write(self, assets: Path) -> list[tuple[str, str]]:
        for old in assets.glob(f"{self.prefix}_*.obj"):
            old.unlink()
        out = []
        for (mat, i, j), o in sorted(self.objs.items()):
            name = f"{self.prefix}_{mat}_{i}_{j}.obj"
            o.write(assets / name)
            out.append((mat, name))
        return out


_BOX_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                   [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])


def obox(cx, cy, z0, z1, length, width, yaw):
    """A box standing on z0, `length` along compass-free direction yaw (radians from +x)."""
    c, s_ = math.cos(yaw), math.sin(yaw)
    pts = [(-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2)]
    ring = [(cx + c * u - s_ * v, cy + s_ * u + c * v) for u, v in pts]
    return np.array([(x, y, z0) for x, y in ring] + [(x, y, z1) for x, y in ring]), _BOX_F


def seg_box(a, b, width, z0, z1, za=0.0, zb=0.0):
    """A box along the segment a-b (its ends may sit at different ground heights za, zb)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1e-3
    nx, ny = -dy / L * width / 2, dx / L * width / 2
    ring = [(a[0] - nx, a[1] - ny, za), (b[0] - nx, b[1] - ny, zb), (b[0] + nx, b[1] + ny, zb),
            (a[0] + nx, a[1] + ny, za)]
    v = [(x, y, z + z0) for x, y, z in ring] + [(x, y, z + z1) for x, y, z in ring]
    return np.array(v), _BOX_F


def strut(p, q, w):
    """A thin square beam between two 3D points (lattice members, cables)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.linalg.norm(d) or 1e-3
    d /= L
    u = np.cross(d, [0, 0, 1.0]) if abs(d[2]) < 0.95 else np.cross(d, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    h = w / 2
    ring = [-u * h - v * h, u * h - v * h, u * h + v * h, -u * h + v * h]
    return np.array([p + r for r in ring] + [q + r for r in ring]), _BOX_F[4:]  # open ends


def osm_trees(raw: dict, city_dir: Path) -> list:
    """Single trees and tree rows (one every 8 m) mapped in OSM."""
    ref = raw.get("features")
    if not ref:
        return []
    feats = json.loads((city_dir / ref).read_text())
    pts = [tuple(p) for p in feats.get("trees", [])]
    for line in feats.get("tree_rows", []):
        pts += densify(line, 8.0)
    return pts


def street_props(raw: dict, city: City, city_dir: Path, assets: Path, stands: list[dict]) -> tuple[list, list]:
    """Hedges, walls, fences, light and radio masts, power lines, cars on the surface car
    parks and jet bridges to the parked aircraft. Returns the mesh files and the car-park
    polygons (the photo under them is cleaned of its flat cars)."""
    ref = raw.get("features")
    feats = json.loads((city_dir / ref).read_text()) if ref else {}
    ms = MeshSet("props", city.bounds[:2])
    prng = np.random.default_rng(23)
    g = city.ground
    bx0, by0, bx1, by1 = city.bounds
    reach = 1500.0  # small props further out than this would be sub-pixel: leave them out

    def near(line) -> bool:
        return any(bx0 - reach < x < bx1 + reach and by0 - reach < y < by1 + reach for x, y in line)

    for k in ("hedges", "walls", "fences", "parking"):
        feats[k] = [ln for ln in feats.get(k, []) if near(ln)]

    for line in feats.get("hedges", []):
        for a, b in itertools.pairwise(line):
            ms.add("hedge", *a, *seg_box(a, b, 1.1, 0, 1.7, float(g(*a)), float(g(*b))))
    for line in feats.get("walls", []):
        for a, b in itertools.pairwise(line):
            ms.add("wall", *a, *seg_box(a, b, 0.3, 0, 2.6, float(g(*a)), float(g(*b))))
    for line in feats.get("fences", []):  # mesh panels and a top rail per segment, posts every 4 m
        for a, b in itertools.pairwise(line):
            za, zb = float(g(*a)), float(g(*b))
            ms.add("fence", *a, *seg_box(a, b, 0.03, 0.05, 2.4, za, zb))
            ms.add("post", *a, *seg_box(a, b, 0.06, 2.45, 2.55, za, zb))
        for x, y in densify(line, 4.0):
            z = float(g(x, y))
            ms.add("post", x, y, *trunk(x, y, z, z + 2.6, 0.06, 3))
    for x, y, kind, h in feats.get("masts", []):
        z = float(g(x, y))
        if kind == "light":  # apron floodlight: a tapered pole with a lamp head
            ms.add("mast", x, y, *trunk(x, y, z, z + h, 0.35, 8))
            ms.add("mast", x, y, *obox(x, y, z + h, z + h + 0.4, 4.5, 1.2, prng.uniform(0, math.pi)))
            ms.add("lamp", x, y, *obox(x, y, z + h - 0.25, z + h, 4.2, 1.0, prng.uniform(0, math.pi)))
        else:  # radio mast with panel antennas
            ms.add("mast", x, y, *trunk(x, y, z, z + h, 0.5, 8))
            for k in range(3):
                a = k * 2 * math.pi / 3
                ms.add("mast", x, y, *obox(x + 0.7 * math.cos(a), y + 0.7 * math.sin(a), z + h - 3, z + h - 0.5,
                                           0.15, 0.4, a))
    tower_h = 38.0
    towers = [(round(x, 1), round(y, 1)) for x, y in feats.get("power_towers", [])]
    tower_set = set(towers)
    perp: dict[tuple, np.ndarray] = {}  # cross-arm direction: across the line at each pylon
    spans = []
    for line in feats.get("power_lines", []):
        pts = [(round(p[0], 1), round(p[1], 1)) for p in line]
        pts = [p for p in pts if p in tower_set]
        for a, b in itertools.pairwise(pts):
            d = np.subtract(b, a)
            if np.linalg.norm(d) < 1:
                continue
            n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            for p in (a, b):
                perp.setdefault(p, n)
            spans.append((a, b))
    arms = ((tower_h * 0.72, 12.0, 11.0), (tower_h * 0.9, 8.0, 7.0))  # height, half length, cable offset
    for x, y in towers:  # lattice pylon: four legs, bracing, two cross arms
        z = float(g(x, y))
        n = perp.get((x, y), np.array([1.0, 0.0]))
        t = np.array([-n[1], n[0]])
        base, top = 4.0, 1.2
        legs = []
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            off = n * sx + t * sy
            legs.append(((x + off[0] * base, y + off[1] * base, z), (x + off[0] * top, y + off[1] * top, z + tower_h)))
        for p, q in legs:
            ms.add("steel", x, y, *strut(p, q, 0.35))
        for k in range(4):
            (p0, q0), (p1, q1) = legs[k], legs[(k + 1) % 4]
            for t0, t1 in ((0.0, 0.35), (0.35, 0.65), (0.65, 0.95)):
                pa = np.add(p0, np.multiply(t0, np.subtract(q0, p0)))
                pb = np.add(p1, np.multiply(t1, np.subtract(q1, p1)))
                ms.add("steel", x, y, *strut(pa, pb, 0.18))
        for zh, half, _ in arms:
            ms.add("steel", x, y, *strut((x - n[0] * half, y - n[1] * half, z + zh),
                                         (x + n[0] * half, y + n[1] * half, z + zh), 0.6))
    for a, b in spans:  # four cables per span, hanging from the arm tips
        za, zb = float(g(*a)), float(g(*b))
        na, nb = perp[a], perp[b]
        for zh, _, off in arms:
            for sgn in (-1, 1):
                ms.add("cable", *a, *strut((a[0] + sgn * off * na[0], a[1] + sgn * off * na[1], za + zh - 1),
                                           (b[0] + sgn * off * nb[0], b[1] + sgn * off * nb[1], zb + zh - 1), 0.08))

    # cars in rows on the surface car parks: 2.6 m bays, 5 m deep, a 6.5 m aisle every two rows
    lots = [Polygon(p) for p in feats.get("parking", [])]
    lots = [p for p in lots if p.is_valid and p.area > 40]
    for lot in lots:
        rect = lot.minimum_rotated_rectangle
        xs_, ys_ = rect.exterior.coords.xy
        e1 = np.array([xs_[1] - xs_[0], ys_[1] - ys_[0]])
        e2 = np.array([xs_[2] - xs_[1], ys_[2] - ys_[1]])
        u, wdt = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
        lu_, lw = np.linalg.norm(u), np.linalg.norm(wdt)
        u, wdt = u / lu_, wdt / lw
        o = np.array([xs_[0], ys_[0]]) if np.linalg.norm(e1) >= np.linalg.norm(e2) else np.array([xs_[1], ys_[1]])
        inner = lot.buffer(-1.0)
        yaw = math.atan2(wdt[1], wdt[0])  # cars point across the rows
        if lw < 5.5:  # a strip along a street: one row, cars nose to tail
            rows, bay, yaw = [lw / 2], 5.8, math.atan2(u[1], u[0])
        else:  # two rows back to back, then an aisle
            rows = [v0 + dv for v0 in np.arange(1.0, lw, 16.5) for dv in (2.5, 7.7) if v0 + dv < lw - 2.0]
            bay = 2.6
        for row_v in rows:
            for bay_u in np.arange(bay / 2 + 0.2, lu_ - bay / 2, bay):
                c = o + u * bay_u + wdt * row_v
                if prng.random() > 0.72 or not inner.contains(Point(*c)):
                    continue
                z = float(g(*c))
                paint = f"car{int(prng.integers(6))}"
                ms.add(paint, *c, *obox(c[0], c[1], z + 0.3, z + 1.05, 4.4, 1.8, yaw))
                cc = c - np.array([math.cos(yaw), math.sin(yaw)]) * 0.25
                ms.add("glass", *c, *obox(cc[0], cc[1], z + 1.05, z + 1.48, 2.3, 1.6, yaw))
                ms.add("tyre", *c, *obox(c[0], c[1], z, z + 0.35, 3.7, 1.7, yaw))

    # jet bridges: from the nearest terminal wall to the forward left door of a parked aircraft
    walls_tree = STRtree([b.footprint for b in city.buildings]) if city.buildings else None
    for s in stands:
        if not s["parked"] or walls_tree is None:
            continue
        L, _, r = AIRLINERS[s["kind"]]
        d = np.asarray(s["dir"])
        left = np.array([-d[1], d[0]])
        door = np.asarray(s["nose"]) - d * 0.13 * L + left * (r + 0.4)
        i = walls_tree.query_nearest(Point(*door))[0]
        wall_pt = city.buildings[i].footprint.exterior.interpolate(
            city.buildings[i].footprint.exterior.project(Point(*door)))
        start = np.array([wall_pt.x, wall_pt.y])
        if np.linalg.norm(door - start) > 45:
            continue  # no terminal close by: a remote stand, served by buses
        zc = r + 0.18 * r + 1.0 + float(g(*door))
        zr = float(g(*start)) + 5.0
        ms.add("bridge", *door, *seg_box(start, door, 3.2, 0, 3.0, zr - 1.5, zc - 1.5))
        ms.add("bridge_glass", *door, *seg_box(start, door, 3.25, 1.6, 2.4, zr - 1.5, zc - 1.5))
        mid = (start + door) / 2
        zm = (zr + zc) / 2 - 1.5
        ms.add("post", *mid, *trunk(mid[0], mid[1], float(g(*mid)), zm, 0.35, 6))  # drive column
        ms.add("tyre", *mid, *obox(mid[0], mid[1], float(g(*mid)), float(g(*mid)) + 0.9, 1.2, 3.0,
                                    math.atan2(d[1], d[0])))
    # ground equipment at each parked aircraft: a tug at the nose, a belt loader and a train of
    # baggage carts at the forward hold, sometimes a fuel truck under the wing
    gse = {"tug": (3.2, 1.8, 1.4), "loader": (7.5, 2.0, 1.2), "cart": (3.0, 1.5, 1.6), "fuel": (9.0, 2.5, 3.0)}
    for s in stands:
        if not s["parked"]:
            continue
        L, S, r = AIRLINERS[s["kind"]]
        d = np.asarray(s["dir"])
        right = np.array([d[1], -d[0]])
        nose = np.asarray(s["nose"])
        yaw = math.atan2(d[1], d[0])
        z = float(g(*nose))
        hold = nose - d * 0.3 * L + right * (r + 2.0)
        place = [("tug", nose + d * 3.5, yaw), ("loader", hold + right * 1.0, yaw + math.pi / 2)]
        place += [("cart", hold + right * (6.0 + 3.6 * k), yaw + math.pi / 2) for k in range(int(prng.integers(2, 5)))]
        if prng.random() < 0.4:
            place.append(("fuel", nose - d * 0.45 * L + right * (0.3 * S), yaw))
        for name, c, a in place:
            ln, wd, h = gse[name]
            ms.add(f"gse_{name}", *c, *obox(c[0], c[1], z + 0.35, z + h, ln, wd, a))
            ms.add("tyre", *c, *obox(c[0], c[1], z, z + 0.4, ln * 0.85, wd * 0.95, a))
    return ms.write(assets), lots


def aircraft_look(name: str) -> str:
    if name.startswith("tail"):
        return pbr(TAIL_RGB[int(name[4:])], 0.5)
    rgb, rough = AIRCRAFT_LOOK[name]
    return pbr(rgb, rough)


def mesh_shape(url: str, rgb, rough=0.9) -> str:
    return f'Shape {{ appearance {pbr(rgb, rough)} geometry Mesh {{ url "{url}" }} castShadows FALSE }}'


PAVED = ("road", "runway", "taxiway", "apron", "industrial")  # land use where vehicles stand


def _open(mask: np.ndarray, k: int) -> np.ndarray:
    """Morphological opening: drop anything thinner than k px."""
    m = Image.fromarray(mask.astype(np.uint8) * 255)
    if k > 1:
        m = m.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))
    return np.asarray(m) > 0


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate by about `px` pixels; large radii on a coarser grid (PIL's filters are O(k^2))."""
    if px <= 0:
        return mask
    f = max(1, px // 4)
    H, W = mask.shape
    h, w = -(-H // f), -(-W // f)
    pad = np.zeros((h * f, w * f), bool)
    pad[:H, :W] = mask
    small = pad.reshape(h, f, w, f).any(axis=(1, 3))
    k = 2 * -(-px // f) + 1
    small = np.asarray(Image.fromarray(small.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(k))) > 0
    return np.repeat(np.repeat(small, f, axis=0), f, axis=1)[:H, :W]


def _local_mean(a: np.ndarray, valid: np.ndarray, f: int = 24) -> np.ndarray:
    """Smooth local colour from the valid pixels only (normalised convolution on a coarse grid)."""
    H, W = valid.shape
    w = valid.astype(np.float32)[..., None]

    def blur(x):
        out = []
        for c in range(x.shape[2]):
            g = np.asarray(Image.fromarray(x[..., c]).resize((max(1, W // f), max(1, H // f)), Image.BOX))
            p = np.pad(g, 1, mode="edge")  # a 3x3 mean on the coarse grid widens the reach
            g = sum(p[i:i + g.shape[0], j:j + g.shape[1]] for i in range(3) for j in range(3)) / 9
            out.append(np.asarray(Image.fromarray(g.astype(np.float32)).resize((W, H), Image.BILINEAR)))
        return np.stack(out, axis=2)

    num, den = blur(a * w), blur(w)
    out = num / np.maximum(den, 1e-3)
    out[den[..., 0] < 0.02] = np.nan  # too little valid ground nearby to say
    return out


def _fill_colour(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The local colour of the valid ground: near first, further out where there is none near."""
    out = _local_mean(a, valid, 24)
    far = np.isnan(out[..., 0])
    if far.any():
        out[far] = _local_mean(a, valid, 128)[far]
    return out


def clean_surface(a: np.ndarray, bounds, lu: np.ndarray, city_bounds, near_bld: np.ndarray) -> None:
    """Every vehicle the photo shows on paved ground (taxiing aircraft, buses, tugs, cars on
    the roads) and the shadows beside them: blobs much brighter or darker than the pavement
    around them, wider than paint lines, become that pavement. Building shadows stay. In place."""
    H, W = a.shape[:2]
    tx0, ty0, tx1, ty1 = bounds
    cx0, cy0, cx1, cy1 = city_bounds
    nrow, ncol = lu.shape[:2]
    xs = tx0 + (np.arange(W) + 0.5) / W * (tx1 - tx0)
    ys = ty1 - (np.arange(H) + 0.5) / H * (ty1 - ty0)
    cols = np.clip(((xs - cx0) / (cx1 - cx0) * ncol).astype(int), 0, ncol - 1)
    rows = np.clip(((1 - (ys - cy0) / (cy1 - cy0)) * nrow).astype(int), 0, nrow - 1)
    cls = lu[np.ix_(rows, cols)]
    road = cls == LANDUSE.index("road")
    field = np.isin(cls, [LANDUSE.index(c) for c in PAVED if c != "road"])
    # pavement the land use misses (an apron OSM has no outline for): grey, not green
    grey = (a.max(axis=2) - a.min(axis=2) < 18) & (a.mean(axis=2) > 150)  # bright concrete only
    ppm_ = W / (tx1 - tx0)
    beside = _grow(np.isin(cls, [LANDUSE.index(c) for c in ("apron", "taxiway")]), int(60 * ppm_))
    field |= grey & beside & np.isin(cls, [LANDUSE.index("meadow"), LANDUSE.index("urban")])
    if not (road.any() or field.any()):
        return
    lum = a.mean(axis=2)
    paved = road | field
    bg = _fill_colour(a, paved)
    d = np.nan_to_num(lum - bg.mean(axis=2))
    ppm = W / (tx1 - tx0)  # px per metre
    big = max(3, int(2.5 * ppm) | 1)  # aircraft, buses: wider than any paint line
    small = max(3, int(1.4 * ppm) | 1)  # cars on the roads
    on_field = field & (d > 20)
    bright = _touching(on_field, _open(on_field, big))  # bodies and what is joined to them
    bright |= _open(road & ((d > 30) | (d < -40)), small)
    bright |= _open(field & ((d > 13) | ((d < -20) & ~near_bld)), small)  # carts, tugs, vans, pale or dark
    bright = _grow(bright, int(1.5 * ppm))
    dark = _open(field & (d < -20) & ~near_bld, big) & _grow(bright, int(40 * ppm))  # their shadows
    m = _grow(bright | dark, max(1, int(1.0 * ppm))) & paved
    if m.any():
        fill = _fill_colour(a, paved & ~m & (d > -30))  # the sunlit pavement around, without the blobs
        m &= ~np.isnan(fill[..., 0])  # no clean pavement within reach: leave the photo as it is
        a[m] = fill[m] + np.random.default_rng(5).normal(0, 2.0, (m.sum(), 1))


def _touching(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """The parts of `mask` connected (4-neighbour) to any pixel of `seed`."""
    reach = mask & seed
    if not reach.any():
        return reach
    while True:  # grow inside the mask until nothing changes
        grown = reach.copy()
        grown[1:] |= reach[:-1]
        grown[:-1] |= reach[1:]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= mask
        if (grown == reach).all():
            return reach
        reach = grown


def near_buildings(img_size, bounds, buildings, metres=25.0) -> np.ndarray:
    """Pixels within `metres` of a building footprint: their dark patches are its shadow."""
    W, H = img_size
    tx0, ty0, tx1, ty1 = bounds
    s = W / (tx1 - tx0)
    tile = Polygon([(tx0, ty0), (tx1, ty0), (tx1, ty1), (tx0, ty1)]).buffer(metres)
    m = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(m)
    for b in buildings:
        fp = b.footprint
        if not fp.intersects(tile):
            continue
        g = fp.buffer(metres, quad_segs=2)
        dr.polygon([((x - tx0) * s, (ty1 - y) * s) for x, y in g.exterior.coords], fill=255)
    return np.asarray(m) > 0


def clean_photo(img: Image.Image, bounds, stands: list[dict], lots: list, lu=None,
                city_bounds=None, buildings=()) -> Image.Image:
    """Take the flat objects out of an aerial photo tile where 3D ones stand now: on each
    stand, blobs much brighter (fuselages) or darker (their shadows) than the concrete become
    concrete (thin paint lines survive the opening); car parks become plain asphalt."""
    tx0, ty0, tx1, ty1 = bounds
    s = img.width / (tx1 - tx0)
    tile = Polygon([(tx0, ty0), (tx1, ty0), (tx1, ty1), (tx0, ty1)])
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    nrng = np.random.default_rng(7)
    near_bld = near_buildings(img.size, bounds, buildings)
    on_wall = near_buildings(img.size, bounds, buildings, metres=2.5)
    if lu is not None:
        clean_surface(a, bounds, lu, city_bounds, near_bld)

    def patch(poly):
        """(rows, cols) slice of the polygon's bounding box and its mask there, or None."""
        bx0, by0, bx1, by1 = poly.bounds
        c0, c1 = max(0, int((bx0 - tx0) * s)), min(img.width, int((bx1 - tx0) * s) + 1)
        r0, r1 = max(0, int((ty1 - by1) * s)), min(img.height, int((ty1 - by0) * s) + 1)
        if c1 - c0 < 3 or r1 - r0 < 3:
            return None
        m = Image.new("L", (c1 - c0, r1 - r0), 0)
        ImageDraw.Draw(m).polygon([((x - tx0) * s - c0, (ty1 - y) * s - r0) for x, y in poly.exterior.coords],
                                  fill=255)
        return (slice(r0, r1), slice(c0, c1)), np.asarray(m) > 0

    for lot in lots:
        if not lot.intersects(tile) or (p := patch(lot)) is None:
            continue
        sl, m = p
        sub = a[sl]
        if m.sum() < 20:
            continue
        sub[m] = np.median(sub[m], axis=0) + nrng.normal(0, 3.0, (m.sum(), 1))
    for st in stands:
        if not st["foot"].intersects(tile) or (p := patch(st["foot"])) is None:
            continue
        sl, m = p
        sub = a[sl]
        if m.sum() < 50:
            continue
        lum = sub.mean(axis=2)
        base = np.median(lum[m])
        dark = (lum < base - 18) & m
        # a dark patch that reaches a wall is that building's shadow and stays; one standing
        # free on the apron is an aircraft's
        dark &= ~_touching(dark, on_wall[sl])
        raw_ = m & ((lum > base + 14) | dark)
        k = max(3, int(2.5 * s) | 1)  # opening: drop anything thinner than ~2.5 m (paint lines)
        body = _open(raw_, k)
        bm = _grow(_touching(raw_, body), 4) & m  # and everything joined to a body: wings, tails
        if not bm.any():
            continue
        rest = m & ~bm & (lum >= base - 18)  # fill with sunlit concrete, not shadow
        fill = np.median(sub[rest if rest.any() else m], axis=0)
        sub[bm] = fill + nrng.normal(0, 2.5, (bm.sum(), 1))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


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
    facade_obj(np.asarray(verts).reshape(-1, 3, 3), out, FACADE_W, FACADE_H)


def facade_obj(p: np.ndarray, out: Path, w: float, h: float) -> None:
    """Triangles (n, 3, 3) -> an OBJ with UVs `w` m along each wall and `h` m up it."""
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    t = np.stack([-nrm[:, 1], nrm[:, 0]], axis=1)  # horizontal, along the wall
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    u = np.einsum("tvk,tk->tv", p[:, :, :2], t) / w
    v = p[:, :, 2] / h
    lines_ = [f"v {x:.2f} {y:.2f} {z:.2f}" for x, y, z in p.reshape(-1, 3)]
    lines_ += [f"vt {a:.4f} {b:.4f}" for a, b in zip(u.ravel(), v.ravel(), strict=True)]
    lines_ += [f"f {k}/{k} {k + 1}/{k + 1} {k + 2}/{k + 2}" for k in range(1, len(p) * 3 + 1, 3)]
    out.write_text("\n".join(lines_) + "\n")


HOUSE_W, HOUSE_H = 7.0, 2.9  # m of wall per house facade texture repeat: one storey, two windows


def house_texture(out: Path) -> None:
    """A storey of a plastered house: off-white render, two windows with frames and shutters."""
    n = 256
    img = np.empty((n, n, 3), float)
    img[:] = (0.86, 0.84, 0.78)
    img += rng.normal(0, 0.012, img.shape)
    for cx in (0.27, 0.73):  # windows, image rows run top-down
        x0, x1 = int((cx - 0.09) * n), int((cx + 0.09) * n)
        y0, y1 = int(0.22 * n), int(0.62 * n)
        img[y0 - 4:y1 + 4, x0 - 4:x1 + 4] = (0.95, 0.95, 0.93)  # frame
        img[y0:y1, x0:x1] = (0.16, 0.2, 0.24)  # glass
        img[y0:y1, (x0 + x1) // 2 - 1:(x0 + x1) // 2 + 1] = (0.95, 0.95, 0.93)
        img[y0:y1, x0 - 18:x0 - 6] = (0.35, 0.22, 0.13)  # shutters
        img[y0:y1, x1 + 6:x1 + 18] = (0.35, 0.22, 0.13)
    img[int(0.95 * n):] = (0.7, 0.68, 0.62)  # plinth line
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out)


def surround_parts(raw: dict, city_dir: Path, assets: Path, rel: str) -> tuple[list[str], list, list, tuple]:
    """The land around the City: its aerial photo as plates, the LoD2 villages (plastered
    walls, tiled or gravel roofs). Returns VRML parts, tree points for the forests there
    (denser near the City) and the bounds it covers."""
    sur = raw.get("surround")
    if not sur:
        return [], [], [], None
    parts = []
    for t in sur["tiles"]:
        tx0, ty0, tx1, ty1 = t["bounds"]
        name = Path(t["image"]).name
        (assets / name).write_bytes((city_dir / t["image"]).read_bytes())
        parts.append(f"Pose {{ translation {(tx0 + tx1) / 2} {(ty0 + ty1) / 2} -0.05 children [ Shape {{ "
                     f"appearance {photo_look(f'{rel}/{name}')} geometry Box {{ size {tx1 - tx0} {ty1 - ty0} 0.1 }} "
                     f"castShadows FALSE }} ] }}")
    house_texture(assets / "house.png")
    looks = {"walls": pbr((1, 1, 1), 0.85, texture=f"{rel}/house.png"),
             "roofs_pitched": pbr((0.56, 0.24, 0.16), 0.75), "roofs_flat": pbr((0.5, 0.5, 0.49), 0.9)}
    data = np.load(city_dir / sur["buildings"])
    for old in assets.glob("sur_*.obj"):
        old.unlink()
    defined = set()
    for mesh in sur["meshes"]:
        kind = mesh.rsplit("_", 2)[0]
        v = data[f"{mesh}_v"].astype(float) / 10
        f = data[f"{mesh}_f"]
        out = assets / f"sur_{mesh}.obj"
        if kind == "walls":
            facade_obj(v[f], out, HOUSE_W, HOUSE_H)
        else:
            lines_ = [f"v {x:.1f} {y:.1f} {z:.1f}" for x, y, z in v] + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
            out.write_text("\n".join(lines_) + "\n")
        look = f"USE SUR_{kind.upper()}" if kind in defined else f"DEF SUR_{kind.upper()} {looks[kind]}"
        defined.add(kind)
        parts.append(f'Shape {{ appearance {look} geometry Mesh {{ url "{rel}/{out.name}" }} castShadows FALSE }}')
    trees_, far_trees = [], []
    ref = raw.get("features")
    if ref:
        x0, y0, x1, y1 = raw["bounds"]
        frng = np.random.default_rng(31)
        for ring in json.loads((city_dir / ref).read_text()).get("woods", []):
            poly = Polygon(ring)
            if not poly.is_valid or poly.area < 200:
                continue
            bx0, by0, bx1, by1 = poly.bounds
            far = max(0.0, max(x0 - bx1, bx0 - x1, y0 - by1, by0 - y1))
            if bx0 >= x0 and bx1 <= x1 and by0 >= y0 and by1 <= y1:
                continue  # inside the City the land-use raster already plants its forests
            spacing = 18.0 if far < 1500 else 30.0  # thinner where only the canopy line shows
            n = int(poly.area / spacing**2)
            if n == 0:
                continue
            cand = np.column_stack([frng.uniform(bx0, bx1, 3 * n), frng.uniform(by0, by1, 3 * n)])
            inside = shapely.contains_xy(poly, cand[:, 0], cand[:, 1])
            pts = cand[inside][:n]
            out_city = ~((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1))
            (trees_ if far < 1500 else far_trees).extend(tuple(p) for p in pts[out_city])
    return parts, trees_, far_trees, tuple(sur["bounds"])


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
    side += [raw["features"]] if raw.get("features") else []
    if raw.get("surround"):
        side += [t["image"] for t in raw["surround"]["tiles"]] + [raw["surround"]["buildings"]]
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
    aircraft, stands = stand_aircraft(raw, city, assets)
    props, lots = street_props(raw, city, city_file.parent, assets, stands)
    if photo:  # with the flat aircraft and cars taken out where 3D ones stand
        for t in photo:
            clean_photo(Image.open(city_file.parent / t["image"]), t["bounds"], stands, lots, lu, city.bounds, city.buildings).save(
                assets / Path(t["image"]).name, quality=88)
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
    rel = assets.name
    ring_parts, ring_trees, far_trees, ring_bounds = surround_parts(raw, city_file.parent, assets, rel)
    osm_near, osm_far = [], []  # mapped trees: full detail near the City, cones further out
    for x, y in osm_trees(raw, city_file.parent):
        (osm_near if x0 - 1500 < x < x1 + 1500 and y0 - 1500 < y < y1 + 1500 else osm_far).append((x, y))
    tree_tiles = trees(lu, city, assets, forest_only=bool(photo), extra=osm_near + ring_trees,
                       far=far_trees + osm_far)
    rail_meshes(city, assets)

    parts = ["#VRML_SIM R2025a utf8",
             f"# generated by tools/build_webots_world.py from {scenario_path} -- do not edit by hand"]
    parts += [f'EXTERNPROTO "{PROTO}/{p}"' for p in EXTERNPROTOS]
    span = max(x1 - x0, y1 - y0)
    cx = (x0 + x1) / 2
    parts += [
        'WorldInfo { info [ "Earshot" ] basicTimeStep 20 gravity 0 }',
        (f"DEF VIEW Viewpoint {{ orientation {axis_angle_view(0, 14)} position {cx:.0f} "
        f"{y0 - 0.45 * span:.0f} {float(city.ground(cx, y0)) + 0.22 * span:.0f} "
        f"far {14000 if raw.get('surround') else 25000}"  # with surroundings: only as far as the haze
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
        parts += ring_parts  # the photographed land around the City, its villages
        # grass around all of it as a frame, not under it: coplanar plates z-fight at a distance
        m = 2.5 * span
        fx0_, fy0_, fx1_, fy1_ = ring_bounds or (x0, y0, x1, y1)
        field = next((Path(t["image"]).name for t in raw["surround"]["tiles"]), None) if ring_bounds else None
        for fx0, fy0, fx1, fy1 in ((fx0_ - m, fy1_, fx1_ + m, fy1_ + m), (fx0_ - m, fy0_ - m, fx1_ + m, fy0_),
                                   (fx0_ - m, fy0_, fx0_, fy1_), (fx1_, fy0_, fx1_ + m, fy1_)):
            if field:  # one photographed square of farmland, repeated per km out into the haze
                look = (f'PBRAppearance {{ baseColor 0.5 0.5 0.5 roughness 1 metalness 0 baseColorMap '
                        f'ImageTexture {{ url "{rel}/{field}" }} textureTransform TextureTransform {{ '
                        f"scale {(fx1 - fx0) / 1000:.1f} {(fy1 - fy0) / 1000:.1f} }} }}")
            else:
                look = pbr((0.42, 0.55, 0.3), 1)
            parts.append(f"Pose {{ translation {(fx0 + fx1) / 2} {(fy0 + fy1) / 2} -0.05 children [ Shape {{ "
                         f"appearance {look} geometry Box {{ size {fx1 - fx0} "
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
    looks = set()  # one appearance per material, DEF on first use
    for prefix, files, look_of in (("AC", aircraft, aircraft_look), ("PROP", props, prop_look)):
        for name, file in files:
            key = f"{prefix}_{name.upper()}"
            look = f"USE {key}" if key in looks else f"DEF {key} {look_of(name)}"
            looks.add(key)
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
        f'width {854 if k == 0 else 384} height {480 if k == 0 else 216} fieldOfView 1.15 near 0.05 far 9000 '
        f'antiAliasing TRUE ambientOcclusionRadius {2 if k == 0 else 0} bloomThreshold 6 }} ] }}'
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

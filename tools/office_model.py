"""The office, modelled clean (tools/make_office.py): what the scan found, drawn as the
furniture the photos show, inside the walls a layout file gives.

The scan says where things stand and how high they are; the layout (data/cities/<name>_layout.yaml)
adds what a phone scan misses: straight walls, windows, the glass meeting rooms, doors, TVs and
the kitchen. Only the look: the City's buildings (the scan's height map) stay what sound and
agents know. `model(...)` writes one .obj per material into out_dir and returns the City's
`interior` list: [{"mesh", "rgb", "rough", "texture"?, "emissive"?, "transparency"?, "shadows"?}].

The look, from the photos: warm wide oak planks, white walls with a dark skirting, long
live-edge walnut tables on black steel frames with sage-green aluminium chairs, white
sideboards and tall cabinets, a white kitchen with open walnut shelves, black-framed glass,
concrete columns, snake plants in black planters.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from shapely import LineString, Point, Polygon, box, union_all

GRID = 0.25  # m: floor plan cells when there is no layout; their union is an orthogonal outline
WALL_T = 0.12  # m
SKIRT_H = 0.09  # m
TABLE_H = 0.75  # m
TABLE_W = 0.95  # m: a communal table; the scan's piece also holds the chairs either side
SEAT_PITCH = 0.62  # m between chairs along a table
SILL, HEAD = 0.45, 2.45  # m: windows
DOOR_W, DOOR_H = 1.7, 2.2  # m: a double door

# material: rgb, roughness, emissive, transparency
LOOKS = {
    "oak_floor": ((1.0, 1.0, 1.0), 0.6, None, 0.0),
    "carpet": ((0.40, 0.41, 0.43), 1.0, None, 0.0),
    "wall": ((0.93, 0.93, 0.92), 0.9, None, 0.0),
    "kitchen_wall": ((0.55, 0.60, 0.63), 0.9, None, 0.0),
    "skirting": ((0.22, 0.22, 0.23), 0.7, None, 0.0),
    "concrete": ((0.62, 0.62, 0.60), 0.95, None, 0.0),
    "walnut": ((0.28, 0.15, 0.07), 0.8, None, 0.0),
    "steel": ((0.05, 0.05, 0.06), 0.4, None, 0.0),
    "sage": ((0.55, 0.65, 0.61), 0.35, None, 0.0),
    "white": ((0.95, 0.95, 0.94), 0.35, None, 0.0),
    "gap": ((0.55, 0.55, 0.55), 0.6, None, 0.0),
    "worktop": ((0.97, 0.97, 0.96), 0.2, None, 0.0),
    "stainless": ((0.70, 0.71, 0.72), 0.25, None, 0.0),
    "glass": ((0.80, 0.88, 0.90), 0.05, None, 0.82),
    "tv": ((0.03, 0.03, 0.035), 0.2, None, 0.0),
    "display": ((0.02, 0.03, 0.05), 0.6, (0.10, 0.16, 0.24), 0.0),
    "exit": ((0.1, 0.6, 0.2), 0.5, (0.1, 0.9, 0.3), 0.0),
    "upholstery": ((0.44, 0.45, 0.49), 1.0, None, 0.0),
    "mesh_chair": ((0.08, 0.08, 0.09), 0.8, None, 0.0),
    "oak": ((0.78, 0.62, 0.44), 0.55, None, 0.0),
    "planter": ((0.07, 0.07, 0.08), 0.6, None, 0.0),
    "leaves": ((0.17, 0.36, 0.16), 0.8, None, 0.0),
    "leaves_light": ((0.36, 0.52, 0.22), 0.8, None, 0.0),
    "ceramic": ((0.96, 0.96, 0.95), 0.3, None, 0.0),
    "slats": ((0.80, 0.62, 0.42), 0.6, None, 0.0),
    "ceiling_walnut": ((0.52, 0.31, 0.16), 0.5, None, 0.0),
    "ceiling_black": ((0.04, 0.04, 0.04), 1.0, None, 0.0),
    "ceiling_white": ((0.93, 0.93, 0.92), 0.9, None, 0.0),
    "downlight": ((1.0, 1.0, 1.0), 1.0, (1.0, 0.95, 0.85), 0.0),
}
NO_SHADOW = {"glass", "tv", "display", "exit", "slats", "ceiling_walnut", "ceiling_black", "ceiling_white", "downlight"}


class Mesh:
    """Triangles per material, flat-shaded (each face its own vertices and normal); floors
    carry their own texture coordinates."""

    def __init__(self):
        self.tris: dict[str, list] = {k: [] for k in LOOKS}
        self.uvs: dict[str, list] = {k: [] for k in LOOKS}

    def tri(self, mat, a, b, c, uv=None) -> None:
        self.tris[mat].append((a, b, c))
        self.uvs[mat].append(uv or ((0, 0), (0, 0), (0, 0)))

    def quad(self, mat, a, b, c, d, uv=None) -> None:
        self.tri(mat, a, b, c, uv and (uv[0], uv[1], uv[2]))
        self.tri(mat, a, c, d, uv and (uv[0], uv[2], uv[3]))

    def box(self, mat, cx, cy, z0, z1, lx, ly, yaw=0.0, top=None) -> None:
        """Box of lx x ly (along yaw, across), from z0 to z1; `top` another material on top."""
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(cx + c * u - s * v, cy + s * u + c * v)
                   for u, v in ((-lx / 2, -ly / 2), (lx / 2, -ly / 2), (lx / 2, ly / 2), (-lx / 2, ly / 2))]
        lo = [(x, y, z0) for x, y in corners]
        hi = [(x, y, z1) for x, y in corners]
        self.quad(top or mat, *hi)
        self.quad(mat, lo[3], lo[2], lo[1], lo[0])
        for i in range(4):
            j = (i + 1) % 4
            self.quad(mat, lo[i], lo[j], hi[j], hi[i])

    def beam(self, mat, p, q, w, h=None) -> None:
        """A square-ish bar from point p to point q (3D), w wide, h thick."""
        p, q = np.asarray(p, float), np.asarray(q, float)
        h = w if h is None else h
        a = q - p
        a /= np.linalg.norm(a)
        up = np.array((0, 0, 1.0)) if abs(a[2]) < 0.95 else np.array((1.0, 0, 0))
        u = np.cross(a, up)
        u /= np.linalg.norm(u)
        v = np.cross(u, a)
        ring = [u * w / 2 + v * h / 2, -u * w / 2 + v * h / 2, -u * w / 2 - v * h / 2, u * w / 2 - v * h / 2]
        lo = [tuple(p + r) for r in ring]
        hi = [tuple(q + r) for r in ring]
        for i in range(4):
            j = (i + 1) % 4
            self.quad(mat, lo[i], lo[j], hi[j], hi[i])
        self.quad(mat, *hi[::-1])
        self.quad(mat, *lo)

    def cylinder(self, mat, cx, cy, z0, z1, r, k=16, r_top=None, cap=True) -> None:
        r_top = r if r_top is None else r_top
        ring = [(math.cos(2 * math.pi * i / k), math.sin(2 * math.pi * i / k)) for i in range(k)]
        for i in range(k):
            (ax, ay), (bx, by) = ring[i], ring[(i + 1) % k]
            self.quad(mat, (cx + r * ax, cy + r * ay, z0), (cx + r * bx, cy + r * by, z0),
                      (cx + r_top * bx, cy + r_top * by, z1), (cx + r_top * ax, cy + r_top * ay, z1))
            if cap:
                self.tri(mat, (cx, cy, z1), (cx + r_top * ax, cy + r_top * ay, z1),
                         (cx + r_top * bx, cy + r_top * by, z1))

    def blob(self, mat, cx, cy, cz, r, rz, n_lat=5, n_lon=9) -> None:
        """A low-poly sphere (a plant's leaves)."""
        pts = [[(cx + r * math.sin(math.pi * i / n_lat) * math.cos(2 * math.pi * j / n_lon + 0.35 * i),
                 cy + r * math.sin(math.pi * i / n_lat) * math.sin(2 * math.pi * j / n_lon + 0.35 * i),
                 cz + rz * math.cos(math.pi * i / n_lat)) for j in range(n_lon)] for i in range(n_lat + 1)]
        for i in range(n_lat):
            for j in range(n_lon):
                jj = (j + 1) % n_lon
                a, b, c, d = pts[i][j], pts[i + 1][j], pts[i + 1][jj], pts[i][jj]
                if i == 0:
                    self.tri(mat, a, b, c)
                elif i == n_lat - 1:
                    self.tri(mat, a, b, d)
                else:
                    self.quad(mat, a, b, c, d)

    def floor(self, mat, x0, y0, x1, y1, along_x: bool, z=0.0, tile=1.6) -> None:
        """A floor rectangle; its planks run along x or y (tile m per texture repeat)."""
        def uv(x, y):
            return (x / tile, y / tile) if not along_x else (y / tile, x / tile)
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        self.quad(mat, *[(x, y, z) for x, y in pts], uv=[uv(x, y) for x, y in pts])

    def write(self, out_dir: Path, scale: float) -> list[dict]:
        out = []
        for mat, tris in self.tris.items():
            if not tris:
                continue
            t = np.asarray(tris, float)
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            lines = [f"v {x * scale:.2f} {y * scale:.2f} {z * scale:.2f}" for x, y, z in t.reshape(-1, 3)]
            lines += [f"vn {a:.3f} {b:.3f} {c:.3f}" for a, b, c in n]
            textured = mat == "oak_floor"
            if textured:
                lines += [f"vt {u:.4f} {v:.4f}" for tri in self.uvs[mat] for u, v in tri]
                lines += [f"f {3 * k + 1}/{3 * k + 1}/{k + 1} {3 * k + 2}/{3 * k + 2}/{k + 1} "
                          f"{3 * k + 3}/{3 * k + 3}/{k + 1}" for k in range(len(t))]
            else:
                lines += [f"f {3 * k + 1}//{k + 1} {3 * k + 2}//{k + 1} {3 * k + 3}//{k + 1}" for k in range(len(t))]
            (out_dir / f"model_{mat}.obj").write_text("\n".join(lines) + "\n")
            rgb, rough, emissive, transparency = LOOKS[mat]
            item = {"mesh": f"{out_dir.name}/model_{mat}.obj", "rgb": list(rgb), "rough": rough}
            if textured:
                oak_planks(out_dir / "oak_planks.jpg")
                item["texture"] = f"{out_dir.name}/oak_planks.jpg"
            if emissive:
                item["emissive"] = list(emissive)
            if transparency:
                item["transparency"] = transparency
            if mat in NO_SHADOW:
                item["shadows"] = False
            out.append(item)
        return out


def oak_planks(out: Path, px: int = 1024, planks: int = 8) -> None:
    """Wide warm-oak planks, staggered, along the texture's v; one repeat = 1.6 m (power of two
    for Webots): 0.2 m wide planks as in the photos."""
    rng = np.random.default_rng(3)
    img = Image.new("RGB", (px, px))
    d = ImageDraw.Draw(img)
    w = px // planks
    ends = []
    for i in range(planks):
        y = -int(rng.uniform(0, px))
        while y < px:
            L = int(rng.uniform(0.55, 1.0) * px)
            tone = rng.uniform(-16, 16)
            base = np.array((176, 124, 74)) + tone * np.array((1.0, 0.85, 0.7))
            d.rectangle((i * w, y, (i + 1) * w - 1, y + L), fill=tuple(int(c) for c in base))
            ends.append((i, y + L))
            y += L
    a = np.asarray(img, float)
    grain = ndimage.gaussian_filter(rng.normal(0, 1, (px, px)), (40, 1.2))  # long along the plank
    figure = ndimage.gaussian_filter(rng.normal(0, 1, (px, px)), (90, 6))
    a += (9 * grain / grain.std() + 7 * figure / figure.std())[..., None] * np.array((1.0, 0.8, 0.6))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))
    d = ImageDraw.Draw(img)
    for i in range(planks):
        d.line((i * w, 0, i * w, px), fill=(112, 76, 44), width=2)
    for i, y in ends:
        d.line((i * w, y % px, (i + 1) * w, y % px), fill=(118, 80, 48), width=2)
    img.save(out, quality=90)


def floor_plan(seen: np.ndarray, cell: float, bounds) -> Polygon:
    """Without a layout: the floor where the scan saw it, on GRID cells, closed and filled,
    the largest piece."""
    x0, y0, _, _ = bounds
    f = round(GRID / cell)
    ny, nx = seen.shape[0] // f, seen.shape[1] // f
    coarse = seen[: ny * f, : nx * f].reshape(ny, f, nx, f).mean(axis=(1, 3)) > 0.3
    square = np.ones((5, 5), bool)  # 1.25 m: notches and bumps smaller than that are scan fringe
    coarse = ndimage.binary_closing(np.pad(coarse, 5), square)[5:-5, 5:-5]
    coarse = ndimage.binary_opening(ndimage.binary_fill_holes(coarse), square)
    labels, n = ndimage.label(coarse)
    if n == 0:
        raise SystemExit("no floor found in the scan")
    sizes = ndimage.sum(coarse, labels, range(1, n + 1))
    iy, ix = np.nonzero(labels == 1 + int(np.argmax(sizes)))
    shape = union_all(box(x0 + ix * GRID, y0 + iy * GRID, x0 + (ix + 1) * GRID, y0 + (iy + 1) * GRID))
    return shape if shape.geom_type == "Polygon" else max(shape.geoms, key=lambda g: g.area)


# -- walls ----------------------------------------------------------------------------------

def on_segment(seg, a, b, tol=0.05) -> tuple[float, float] | None:
    """seg's extent [s0, s1] along the edge a -> b (metres from a), if it lies on that edge."""
    line = LineString([a, b])
    p, q = Point(seg[:2]), Point(seg[2:])
    if line.distance(p) > tol or line.distance(q) > tol:
        return None
    s0, s1 = sorted((line.project(p), line.project(q)))
    return (s0, s1) if s1 - s0 > 0.05 else None


def wall_run(m: Mesh, a, b, H, outward: int, windows=(), doors=(), mat="wall", skirting=True) -> None:
    """A wall along a -> b, WALL_T thick on the `outward` side (+1: left of a -> b), with
    windows [(s0, s1)] cut in, extended by WALL_T at both ends to close the corners."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    t = (b - a) / L
    n = np.array((-t[1], t[0])) * outward  # into the wall, away from the floor
    yaw = math.atan2(t[1], t[0])

    def piece(s0, s1, z0, z1, material=mat):
        c = a + t * (s0 + s1) / 2 + n * WALL_T / 2
        m.box(material, c[0], c[1], z0, z1, s1 - s0, WALL_T, yaw)

    cuts = sorted(windows)
    s = -WALL_T
    for s0, s1 in cuts + [(L + WALL_T, L + WALL_T)]:
        if s0 > s:
            piece(s, s0, 0, H)
            if skirting:
                c = a + t * (max(s, 0) + min(s0, L)) / 2 - n * 0.006
                m.box("skirting", c[0], c[1], 0, SKIRT_H, min(s0, L) - max(s, 0), 0.012, yaw)
        if s1 > s0:  # a window: wall below and above, glass and black frame in between
            piece(s0, s1, 0, SILL)
            piece(s0, s1, HEAD, H)
            window(m, a + t * s0, a + t * s1, n, yaw)
        s = s1


def window(m: Mesh, p, q, n, yaw, z0=SILL, z1=HEAD, mullion=1.25, frame=0.05) -> None:
    """Glass in black frames between p and q, at the wall's middle."""
    p, q = np.asarray(p, float) + n * WALL_T / 2, np.asarray(q, float) + n * WALL_T / 2
    L = float(np.linalg.norm(q - p))
    c = (p + q) / 2
    m.box("glass", c[0], c[1], z0, z1, L, 0.012, yaw)
    for z in (z0, z1 - frame):
        m.box("steel", c[0], c[1], z, z + frame, L, frame, yaw)
    k = max(1, round(L / mullion))
    for i in range(k + 1):
        x = p + (q - p) * i / k
        m.box("steel", x[0], x[1], z0, z1, frame, frame, yaw)


def glass_wall(m: Mesh, a, b, H) -> None:
    """A black-framed glass partition (the meeting rooms), floor to ceiling."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = (b - a) / np.linalg.norm(b - a)
    window(m, a, b, np.zeros(2), math.atan2(t[1], t[0]), z0=0.0, z1=H, mullion=1.5)
    c = (a + b) / 2  # a frosted band at eye height
    m.box("white", c[0], c[1], 1.0, 1.08, float(np.linalg.norm(b - a)), 0.016, math.atan2(t[1], t[0]))


def walls(m: Mesh, room: Polygon, H: float, layout: dict) -> None:
    """Outer walls along the floor's outline (windows cut in), then the inner walls."""
    ring = room.exterior
    pts = list(ring.coords)
    outward = -1 if ring.is_ccw else 1  # the wall stands outside the floor
    for a, b in itertools.pairwise(pts):
        cuts = [c for w in layout.get("windows", []) if (c := on_segment(w, a, b))]
        wall_run(m, a, b, H, outward, cuts)
    for seg in layout.get("glass", []):
        glass_wall(m, seg[:2], seg[2:], H)
    for seg in layout.get("partitions", []):
        a, b = np.asarray(seg[:2], float), np.asarray(seg[2:], float)
        t = (b - a) / np.linalg.norm(b - a)
        c = (a + b) / 2
        m.box("wall", c[0], c[1], 0, H, float(np.linalg.norm(b - a)), WALL_T, math.atan2(t[1], t[0]))


def facing_frame(at, facing_deg):
    """Point on the wall and the unit vector into the room."""
    f = math.radians(facing_deg)
    return np.asarray(at, float), np.array((math.cos(f), math.sin(f)))


def double_door(m: Mesh, at, facing, exit_sign=False) -> None:
    p, n = facing_frame(at, facing)
    yaw = math.atan2(n[0], -n[1])  # along the wall
    t = np.array((math.cos(yaw), math.sin(yaw)))
    c = p + n * 0.012
    m.box("white", c[0], c[1], 0.0, DOOR_H + 0.06, DOOR_W + 0.12, 0.02, yaw)  # frame
    for side in (-1, 1):
        leaf = c + n * 0.012 + t * side * DOOR_W / 4
        m.box("white", leaf[0], leaf[1], 0.005, DOOR_H, DOOR_W / 2 - 0.01, 0.03, yaw)
        h = c + n * 0.04 + t * side * 0.08
        m.box("stainless", h[0], h[1], 1.0, 1.04, 0.12, 0.02, yaw)
    g = c + n * 0.03
    m.box("gap", g[0], g[1], 0.005, DOOR_H, 0.008, 0.008, yaw)
    if exit_sign:
        e = p + n * 0.06
        m.box("exit", e[0], e[1], DOOR_H + 0.15, DOOR_H + 0.32, 0.4, 0.06, yaw)


def tv(m: Mesh, at, facing, w=1.65, h=0.95, z=1.55) -> None:
    p, n = facing_frame(at, facing)
    yaw = math.atan2(n[0], -n[1])
    c = p + n * 0.04
    m.box("tv", c[0], c[1], z - h / 2, z + h / 2, w, 0.05, yaw)
    d = p + n * 0.072
    m.box("display", d[0], d[1], z - h / 2 + 0.02, z + h / 2 - 0.02, w - 0.04, 0.003, yaw)
    s = p + n * 0.06
    m.box("tv", s[0], s[1], z + h / 2 + 0.1, z + h / 2 + 0.18, 0.9, 0.1, yaw)  # camera bar


# -- furniture ------------------------------------------------------------------------------

def rect(footprint) -> tuple[float, float, float, float, float]:
    """(cx, cy, length, width, yaw) of the piece's minimum rotated rectangle, length along yaw,
    yaw snapped to the walls when it is within 12 degrees of them."""
    r = Polygon(footprint).minimum_rotated_rectangle
    p = np.asarray(r.exterior.coords)[:4]
    e1, e2 = p[1] - p[0], p[2] - p[1]
    if np.linalg.norm(e1) < np.linalg.norm(e2):
        e1, e2 = e2, e1
    yaw = math.atan2(e1[1], e1[0])
    snapped = round(yaw / (math.pi / 2)) * (math.pi / 2)
    if abs(yaw - snapped) < math.radians(12):
        yaw = snapped
    c = r.centroid
    return c.x, c.y, float(np.linalg.norm(e1)), float(np.linalg.norm(e2)), yaw


def front(room: Polygon, cx, cy, yaw, depth) -> float:
    """Direction a piece lying along yaw faces: across it, away from the nearer wall."""
    n = np.array((-math.sin(yaw), math.cos(yaw)))
    ahead = room.exterior.distance(Point(*(np.array((cx, cy)) + n * (depth / 2 + 0.5))))
    behind = room.exterior.distance(Point(*(np.array((cx, cy)) - n * (depth / 2 + 0.5))))
    return yaw + math.pi / 2 if ahead >= behind else yaw - math.pi / 2


def navy_chair(m: Mesh, x, y, facing, seat=0.45, bar=False) -> None:
    """The sage-green aluminium chair (or bar stool): four legs, seat, slatted back; it faces
    `facing` (its back on the far side)."""
    f = np.array((math.cos(facing), math.sin(facing)))
    s = np.array((-f[1], f[0]))
    c = np.array((x, y))
    for u, v in ((0.18, 0.18), (0.18, -0.18), (-0.18, 0.18), (-0.18, -0.18)):
        top = c + f * u * 0.9 + s * v * 0.9
        foot = c + f * u * 1.08 + s * v * 1.08
        m.beam("sage", (*foot, 0.0), (*top, seat), 0.022)
    if bar:  # foot ring
        for u, v, w in ((0.19, 0, 0.42), (-0.19, 0, 0.42)):
            p = c + f * u
            m.box("sage", p[0], p[1], 0.28, 0.3, w, 0.02, facing + math.pi / 2)
    m.box("sage", x, y, seat, seat + 0.035, 0.42, 0.42, facing)
    back = c - f * 0.2
    for v in (-0.19, 0.19):
        p = back + s * v
        m.beam("sage", (*p, seat), (*(p - f * 0.04), seat + 0.42), 0.025)
    for v in (-0.09, 0.0, 0.09):
        p = back + s * v - f * 0.02
        m.box("sage", p[0], p[1], seat + 0.12, seat + 0.36, 0.015, 0.05, facing)
    p = back - f * 0.04
    m.box("sage", p[0], p[1], seat + 0.34, seat + 0.42, 0.04, 0.42, facing)


def walnut_table(m: Mesh, cx, cy, length, yaw, h=TABLE_H, w=TABLE_W, bar=False) -> None:
    """A live-edge walnut top on two black trapezoid steel frames, chairs along both sides."""
    t = np.array((math.cos(yaw), math.sin(yaw)))
    n = np.array((-t[1], t[0]))
    c = np.array((cx, cy))
    m.box("walnut", cx, cy, h - 0.045, h, length, w, yaw)
    for end in (-1, 1):
        p = c + t * end * (length / 2 - 0.25)
        for side in (-1, 1):
            top = p + n * side * (w / 2 - 0.12)
            foot = p + n * side * (w / 2 - 0.02)
            m.beam("steel", (*foot, 0.0), (*top, h - 0.045), 0.05, 0.03)
        m.box("steel", p[0], p[1], h - 0.09, h - 0.045, 0.05, w - 0.2, yaw)
        m.box("steel", p[0], p[1], 0.0, 0.04, 0.05, w - 0.02, yaw)
    k = max(1, int((length - 0.2) // SEAT_PITCH))
    for i in range(k):
        u = -length / 2 + (length - (k - 1) * SEAT_PITCH) / 2 + i * SEAT_PITCH
        for side in (-1, 1):
            p = c + t * u + n * side * (w / 2 + 0.22)
            navy_chair(m, p[0], p[1], yaw - side * math.pi / 2, seat=0.76 if bar else 0.45, bar=bar)


def sideboard(m: Mesh, cx, cy, length, depth, yaw, h=0.85, doors=None) -> None:
    """White credenza: body, door gaps, small handles, a dark plinth. yaw: its front."""
    t = np.array((-math.sin(yaw), math.cos(yaw)))  # along it
    f = np.array((math.cos(yaw), math.sin(yaw)))
    along = yaw + math.pi / 2
    c = np.array((cx, cy))
    m.box("skirting", cx, cy, 0, 0.08, length - 0.04, depth - 0.06, along)
    m.box("white", cx, cy, 0.08, h, length, depth, along)
    k = doors or max(2, round(length / 0.5))
    for i in range(1, k):
        g = c + t * (-length / 2 + length * i / k) + f * (depth / 2 + 0.001)
        m.box("gap", g[0], g[1], 0.1, h - 0.02, 0.006, 0.004, along)
    for i in range(k):
        g = c + t * (-length / 2 + length * (i + 0.5) / k + (0.18 if i % 2 == 0 else -0.18) * length / k) \
            + f * (depth / 2 + 0.012)
        m.box("stainless", g[0], g[1], h - 0.16, h - 0.06, 0.015, 0.02, along)


def tall_cabinet(m: Mesh, cx, cy, length, depth, yaw, h=2.0) -> None:
    sideboard(m, cx, cy, length, depth, yaw, h=h, doors=max(2, round(length / 0.5)))


def kitchen(m: Mesh, spec: dict, H: float) -> None:
    """Base cabinets with a white worktop, the blue-grey wall above, two open walnut shelves with
    plants and crockery, fridges at the `from` end, a sink, a coffee machine, a microwave."""
    a, b = np.asarray(spec["from"], float), np.asarray(spec["to"], float)
    f = np.array((math.cos(math.radians(spec["facing"])), math.sin(math.radians(spec["facing"]))))
    t = (b - a) / np.linalg.norm(b - a)
    L = float(np.linalg.norm(b - a))
    yaw = math.atan2(f[1], f[0])
    along = yaw + math.pi / 2
    fridge_w, depth = 1.8, 0.64
    c = a + t * fridge_w / 2 + f * 0.38
    m.box("stainless", c[0], c[1], 0, 1.85, fridge_w, 0.74, along)  # two french-door fridges
    for k in (-1, 1):
        g = c + t * k * fridge_w / 4 + f * 0.375
        m.box("gap", g[0], g[1], 0.05, 1.8, 0.006, 0.004, along)
        for side in (-1, 1):
            h_ = g + t * side * 0.05 + f * 0.02
            m.box("steel", h_[0], h_[1], 0.9, 1.5, 0.02, 0.03, along)
    run0, run1 = fridge_w + 0.05, L
    mid = a + t * (run0 + run1) / 2
    run = run1 - run0
    cab = mid + f * depth / 2
    sideboard(m, cab[0], cab[1], run, depth, yaw, h=0.87, doors=max(2, round(run / 0.6)))
    top = mid + f * (depth / 2 + 0.01)
    m.box("worktop", top[0], top[1], 0.87, 0.91, run, depth + 0.03, along)
    back = mid + f * 0.006
    m.box("kitchen_wall", back[0], back[1], 0.91, H - 0.15, run, 0.012, along)
    rng = np.random.default_rng(5)
    for z in (1.45, 1.85):  # open walnut shelves
        s = mid + f * 0.15
        m.box("walnut", s[0], s[1], z, z + 0.035, run - 0.2, 0.28, along)
        for u in np.arange(-run / 2 + 0.3, run / 2 - 0.2, 0.42):
            p = mid + t * u + f * 0.15
            r = rng.random()
            if r < 0.4:  # a trailing plant
                m.cylinder("ceramic", p[0], p[1], z + 0.035, z + 0.15, 0.07, k=10)
                m.blob("leaves_light" if rng.random() < 0.5 else "leaves", p[0], p[1], z + 0.2, 0.16, 0.09)
            elif r < 0.75:  # a stack of bowls
                m.cylinder("ceramic", p[0], p[1], z + 0.035, z + 0.1, 0.09, k=12, r_top=0.11)
            else:  # jars
                for d in (-0.06, 0.06):
                    q = p + t * d
                    m.cylinder("glass", q[0], q[1], z + 0.035, z + 0.2, 0.04, k=8)
    sink = a + t * (run0 + 0.55 * run) + f * 0.35
    m.box("stainless", sink[0], sink[1], 0.905, 0.915, 0.7, 0.42, along)
    tap = a + t * (run0 + 0.55 * run) + f * 0.1
    m.beam("stainless", (*tap, 0.91), (*tap, 1.2), 0.025)
    m.beam("stainless", (*tap, 1.2), (*(tap + f * 0.2), 1.2), 0.025)
    for u, (w, d, h) in ((0.25, (0.32, 0.38, 0.4)), (0.8, (0.5, 0.38, 0.3)), (0.9, (0.25, 0.3, 0.35))):
        p = a + t * (run0 + u * run) + f * 0.25
        m.box("tv" if u != 0.8 else "stainless", p[0], p[1], 0.91, 0.91 + h, w, d, along)


def lounge_chair(m: Mesh, x, y, facing) -> None:
    """A grey upholstered shell chair on a black swivel base."""
    f = np.array((math.cos(facing), math.sin(facing)))
    for k in range(4):
        a = facing + math.pi / 4 + k * math.pi / 2
        m.beam("steel", (x, y, 0.06), (x + 0.3 * math.cos(a), y + 0.3 * math.sin(a), 0.02), 0.03)
    m.cylinder("steel", x, y, 0.06, 0.38, 0.03, k=8)
    m.box("upholstery", x, y, 0.38, 0.47, 0.6, 0.58, facing)
    b = np.array((x, y)) - f * 0.26
    m.beam("upholstery", (*b, 0.45), (*(b - f * 0.1), 0.92), 0.6, 0.08)
    s = np.array((-f[1], f[0]))
    for side in (-1, 1):
        p = np.array((x, y)) + s * side * 0.29 - f * 0.05
        m.box("upholstery", p[0], p[1], 0.45, 0.66, 0.06, 0.45, facing + math.pi / 2)


def sofa(m: Mesh, cx, cy, length, depth, yaw) -> None:
    f = np.array((math.cos(yaw), math.sin(yaw)))
    along = yaw + math.pi / 2
    c = np.array((cx, cy))
    m.box("steel", cx, cy, 0, 0.12, length - 0.15, depth - 0.15, along)
    m.box("upholstery", cx, cy, 0.12, 0.44, length, depth, along)
    b = c - f * (depth / 2 - 0.1)
    m.box("upholstery", b[0], b[1], 0.44, 0.8, length, 0.2, along)
    t = np.array((-f[1], f[0]))
    for side in (-1, 1):
        p = c + t * side * (length / 2 - 0.1)
        m.box("upholstery", p[0], p[1], 0.44, 0.62, 0.2, depth, along)


def stool(m: Mesh, x, y) -> None:
    """The round oak side stool."""
    m.cylinder("oak", x, y, 0.43, 0.46, 0.17, k=20)
    for k in range(3):
        a = 2 * math.pi * k / 3
        m.beam("oak", (x + 0.17 * math.cos(a), y + 0.17 * math.sin(a), 0.0),
               (x + 0.1 * math.cos(a), y + 0.1 * math.sin(a), 0.43), 0.025)


def snake_plant(m: Mesh, x, y, h, rng) -> None:
    """Sansevieria in a black cylindrical planter, as in the corridor."""
    pot = min(0.62, 0.45 * h)
    m.cylinder("planter", x, y, 0, pot, 0.2, k=20, r_top=0.23)
    m.cylinder("skirting", x, y, pot - 0.03, pot - 0.02, 0.22, k=20)
    for _ in range(24):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0, 0.12)
        base = (x + r * math.cos(a), y + r * math.sin(a), pot - 0.02)
        lean = rng.uniform(0.05, 0.28)
        tip = (base[0] + lean * math.cos(a), base[1] + lean * math.sin(a), pot + rng.uniform(0.45, 0.95) * (h - pot))
        m.beam("leaves" if rng.random() < 0.7 else "leaves_light", base, tip, 0.04, 0.008)


def column(m: Mesh, x, y, H) -> None:
    m.box("concrete", x, y, 0, H, 0.55, 0.55)


def meeting_room(m: Mesh, r) -> None:
    x0, y0, x1, y1 = r
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    yaw = 0.0 if x1 - x0 >= y1 - y0 else math.pi / 2
    length = 0.55 * max(x1 - x0, y1 - y0)
    m.box("steel", cx, cy, 0, 0.71, 0.1, 0.1)
    m.box("white", cx, cy, 0.71, 0.74, length, 1.0, yaw)
    t = np.array((math.cos(yaw), math.sin(yaw)))
    n = np.array((-t[1], t[0]))
    for u in (-length / 3, 0.0, length / 3):
        for side in (-1, 1):
            p = np.array((cx, cy)) + t * u + n * side * 0.75
            office_chair(m, p[0], p[1], yaw - side * math.pi / 2)


def office_chair(m: Mesh, x, y, facing) -> None:
    f = np.array((math.cos(facing), math.sin(facing)))
    for a in range(5):
        ang = facing + 2 * math.pi * a / 5
        m.beam("steel", (x, y, 0.07), (x + 0.3 * math.cos(ang), y + 0.3 * math.sin(ang), 0.04), 0.03)
    m.cylinder("steel", x, y, 0.07, 0.42, 0.025, k=8)
    m.box("mesh_chair", x, y, 0.42, 0.5, 0.5, 0.48, facing)
    b = np.array((x, y)) - f * 0.24
    m.beam("mesh_chair", (*b, 0.52), (*(b - f * 0.06), 1.05), 0.46, 0.04)


def ceiling(m: Mesh, rects, H: float, walnut=(), plain=()) -> None:
    """The wooden slat ceiling of the photos (walnut planks over the kitchen, white tiles in
    the meeting rooms), drawn as faces that look down only: from inside it is a ceiling, from
    above (back faces, culled) the overview camera looks straight through it. Downlights every
    couple of metres."""
    walnut, plain = [box(*r) for r in walnut], [box(*r) for r in plain]
    for x0, y0, x1, y1 in rects:
        along_x = (x1 - x0) >= (y1 - y0)
        mid = box(x0, y0, x1, y1).centroid
        planks = any(w.contains(mid) for w in walnut)
        pitch, width, mat = (0.32, 0.3, "ceiling_walnut") if planks else (0.1, 0.055, "slats")
        z = H - 0.12
        backing = "ceiling_white" if any(p.contains(mid) for p in plain) else "ceiling_black"
        m.quad(backing, (x0, y0, H - 0.02), (x0, y1, H - 0.02), (x1, y1, H - 0.02), (x1, y0, H - 0.02))
        if backing == "ceiling_white":
            span = ()  # tiles: no slats
        else:
            span = (y0, y1) if along_x else (x0, x1)
        for c in (np.arange(span[0] + pitch / 2, span[1], pitch) if span else ()):  # undersides only: no sides to block the view
            a, b = c - width / 2, c + width / 2
            if along_x:
                m.quad(mat, (x0, a, z), (x0, b, z), (x1, b, z), (x1, a, z))
            else:
                m.quad(mat, (a, y0, z), (a, y1, z), (b, y1, z), (b, y0, z))
        for lx in np.arange(x0 + 1.0, x1 - 0.5, 2.2):
            for ly in np.arange(y0 + 1.0, y1 - 0.5, 2.2):
                k = 12
                ring = [(lx + 0.07 * math.cos(2 * math.pi * i / k), ly + 0.07 * math.sin(2 * math.pi * i / k))
                        for i in range(k)]
                for i in range(k):  # a disc facing down
                    m.tri("downlight", (lx, ly, z - 0.005), ring[(i + 1) % k] + (z - 0.005,), ring[i] + (z - 0.005,))


def place(m: Mesh, room: Polygon, pieces: list[dict], H: float, used: list[Polygon], rng) -> None:
    """Each piece of the scan as the furniture it most likely is (by height, size and where it
    stands), unless the layout already put something there."""
    for p in pieces:
        poly = Polygon(p["footprint"])
        ctr = poly.centroid
        if not room.buffer(0.3).contains(ctr) or any(u.contains(ctr) for u in used):
            continue
        cx, cy, length, width, yaw = rect(p["footprint"])
        h, area = p["height"], poly.area
        if 0.6 <= h <= 0.95 and length >= 1.8 and width >= 1.2:  # communal tables, chairs included
            n_tables = 2 if width > 2.2 else 1
            normal = np.array((-math.sin(yaw), math.cos(yaw)))
            for k in range(n_tables):
                off = 0.0 if n_tables == 1 else (k - 0.5) * (width / 2 + 0.2)
                q = np.array((cx, cy)) + normal * off
                walnut_table(m, q[0], q[1], length - 0.1, yaw)
        elif 0.95 <= h <= 1.3 and length >= 1.8:  # the high table, bar stools along it
            walnut_table(m, cx, cy, max(length, 2.4), yaw, h=1.05, w=0.9, bar=True)
        elif h >= 1.95 and area < 0.5:
            column(m, cx, cy, H)
        elif area < 0.35:
            if h >= 0.9:
                snake_plant(m, ctr.x, ctr.y, min(h, 1.5), rng)
            else:
                stool(m, ctr.x, ctr.y)
        elif h < 0.6:
            facing = front(room, cx, cy, yaw, width)
            if area < 1.4:
                lounge_chair(m, cx, cy, facing)
            else:
                sofa(m, cx, cy, length, min(width, 0.95), facing)
        elif h >= 1.6:
            tall_cabinet(m, cx, cy, length, min(max(width, 0.45), 0.6), front(room, cx, cy, yaw, width))
        else:
            sideboard(m, cx, cy, length, min(max(width, 0.4), 0.55), front(room, cx, cy, yaw, width))


def model(seen: np.ndarray, cell: float, bounds, pieces: list[dict], out_dir: Path, scale: float,
          layout: dict | None = None) -> list[dict]:
    """The clean office. pieces: make_office.furniture's (room metres); layout: the
    <name>_layout.yaml (optional: without it, the scan's floor and plain walls)."""
    layout = layout or {}
    H = float(layout.get("wall_height", 2.7))
    rng = np.random.default_rng(11)
    m = Mesh()
    if layout.get("plan"):
        room = union_all([box(*r) for r in layout["plan"]])
        room = room if room.geom_type == "Polygon" else max(room.geoms, key=lambda g: g.area)
        room = room.simplify(0.001)  # no corners where two rectangles meet in a straight line
    else:
        room = floor_plan(seen, cell, bounds)
    carpet = [box(*r) for r in layout.get("carpet", [])]
    for r in (layout.get("plan") or [room.bounds]):
        x0, y0, x1, y1 = r
        mat = "carpet" if any(c.contains(box(*r).centroid) for c in carpet) else "oak_floor"
        m.floor(mat, x0, y0, x1, y1, along_x=(x1 - x0) >= (y1 - y0))
    walls(m, room, H, layout)
    used = []
    for d in layout.get("doors", []):
        double_door(m, d["at"], d["facing"], d.get("exit", False))
    for t in layout.get("tvs", []):
        tv(m, t["at"], t["facing"])
    if layout.get("kitchen"):
        kitchen(m, layout["kitchen"], H)
        k = layout["kitchen"]
        used.append(LineString([k["from"], k["to"]]).buffer(0.9))
    for r in layout.get("meeting_rooms", []):
        meeting_room(m, r)
        used.append(box(*r))
    place(m, room, pieces, H, used, rng)
    if layout.get("ceiling", True) and layout.get("plan"):
        ceiling(m, layout["plan"], H, layout.get("walnut_ceiling", []), layout.get("carpet", []))
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("model_*.obj"):
        old.unlink()
    return m.write(out_dir, scale)

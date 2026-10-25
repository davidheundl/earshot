"""A Polycam (or any phone LiDAR) scan of the office -> the office City, 1 m room = 100 m city.

What the Webots camera sees is the office modelled clean from the scan (tools/office_model.py;
--look scan: the raw scan); the City's buildings (desks, cabinets, shelves from the scan's
height map) are what sound, RF and the agents know. Same 1:100 fiction as scenarios/room.yaml, so
acoustics, radio and drone kinematics keep their city-scale constants.

    uv run python tools/make_office.py data/cities/office_scan/<scan>.glb --rotate 0

1. load the scan (.glb, .gltf, .obj with its .mtl and texture; trimesh), Y-up unless --up z
2. floor to z = 0, walls squared to the axes (dominant wall direction), then --rotate DEG
   counter-clockwise so that +y points north (check office_top.jpg and run again)
3. cut everything above --ceiling (the overview camera looks in from above), centre, x100
4. a height map (5 cm) of what stands on the floor -> one box building per piece of furniture,
   a Perch Site on every flat top (the arrays stand on one: put drones 0 and 1 there)
Writes data/cities/office.json, data/cities/office_twin/ (one .obj per material, the City's
`interior` key, tools/build_webots_world.py) and office_top.jpg (the scan from above: the
dashboard `basemap`, and the check that north is up).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import office_model
import shapely
import trimesh
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely import Polygon

ROOT = Path(__file__).resolve().parents[1]
SCALE = 100.0  # 1 m in the room = 100 m in the city (scenarios/room.yaml)
CELL = 0.05  # height map cell, room metres
MIN_OBSTACLE = 0.25  # lower than this is floor clutter (bags, cables), room metres
MIN_AREA = 0.06  # m^2: smaller pieces are chair legs and scan noise
WALL_MARGIN = 0.2  # m from the room's bounding box: outer wall, not furniture
FLAT_TOP = 0.03  # m: within this of the commonest height = the top (>= 0.1 m^2: a Perch Site)
# height bands, room metres: desks and tables, cabinets and sideboards, shelves and walls
BANDS = ((MIN_OBSTACLE, 1.0), (1.0, 1.6), (1.6, 10.0))


def load_parts(path: Path, up: str) -> list[dict]:
    """Every textured piece of the scan, transforms applied, Z-up, room metres:
    [{"v": (n, 3), "f": (m, 3), "uv": (n, 2) | None, "image": PIL.Image | None}]."""
    scene = trimesh.load(path, force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        tf, gname = scene.graph[node]
        m = scene.geometry[gname]
        if not isinstance(m, trimesh.Trimesh) or not len(m.faces):
            continue
        v = trimesh.transform_points(np.asarray(m.vertices, float), tf)
        if up == "y":  # glTF and Polycam's OBJ: y up, -z forward
            v = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])
        uv, image = None, None
        vis = m.visual
        if isinstance(vis, trimesh.visual.TextureVisuals) and vis.uv is not None:
            uv = np.asarray(vis.uv, float)
            mat = vis.material
            image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        parts.append({"v": v, "f": np.asarray(m.faces, int), "uv": uv, "image": image})
    if not parts:
        raise SystemExit(f"{path}: no mesh in it")
    return parts


def wall_angle(parts: list[dict]) -> float:
    """Direction of the walls (degrees, 0-90): the area-weighted mode of the horizontal face
    normals, folded into one quadrant."""
    angles, weights = [], []
    for p in parts:
        tri = p["v"][p["f"]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(n, axis=1)
        ok = area > 0
        n, area = n[ok] / area[ok, None], area[ok]
        side = np.abs(n[:, 2]) < 0.2  # vertical faces: walls, cabinet sides
        angles.append(np.degrees(np.arctan2(n[side, 1], n[side, 0])) % 90)
        weights.append(area[side])
    hist, edges = np.histogram(np.concatenate(angles), bins=90, range=(0, 90),
                               weights=np.concatenate(weights))
    hist = ndimage.uniform_filter1d(hist, 3, mode="wrap")
    return float(edges[np.argmax(hist)] + 0.5)


def floor_z(parts: list[dict]) -> float:
    """Height of the floor: the lowest strong level of upward-facing faces."""
    zs, w = [], []
    for p in parts:
        tri = p["v"][p["f"]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(n, axis=1)
        up = (area > 0) & (n[:, 2] > 0.9 * np.maximum(area, 1e-12))
        zs.append(tri[up, :, 2].mean(axis=1))
        w.append(area[up])
    zs, w = np.concatenate(zs), np.concatenate(w)
    lo, hi = np.percentile(zs, [0.5, 99.5])
    hist, edges = np.histogram(zs, bins=max(10, int((hi - lo) / 0.02)), range=(lo, hi), weights=w)
    strong = np.flatnonzero(hist > 0.25 * hist.max())
    return float(edges[strong[0]] + (edges[1] - edges[0]) / 2)


def height_map(parts: list[dict], bounds) -> tuple[np.ndarray, np.ndarray]:
    """Highest surface per CELL square (room metres above the floor), row 0 = south, and
    where the scan saw floor."""
    x0, y0, x1, y1 = bounds
    nx, ny = math.ceil((x1 - x0) / CELL), math.ceil((y1 - y0) / CELL)
    hmap = np.zeros((ny, nx))
    seen = np.zeros((ny, nx), bool)
    for p in parts:
        mesh = trimesh.Trimesh(p["v"], p["f"], process=False)
        n = int(min(4_000_000, max(10_000, mesh.area / (CELL / 3) ** 2)))
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
        ix = np.clip(((pts[:, 0] - x0) / CELL).astype(int), 0, nx - 1)
        iy = np.clip(((pts[:, 1] - y0) / CELL).astype(int), 0, ny - 1)
        np.maximum.at(hmap, (iy, ix), pts[:, 2])
        low = pts[:, 2] < 0.08
        seen[iy[low], ix[low]] = True
    return hmap, seen


def outline(cells: np.ndarray, bounds) -> Polygon:
    """The cells' own outline (room metres), not a hull: an L-shaped desk stays an L."""
    x0, y0, _, _ = bounds
    iy, ix = np.nonzero(cells)
    squares = shapely.box(x0 + ix * CELL, y0 + iy * CELL, x0 + (ix + 1) * CELL, y0 + (iy + 1) * CELL)
    shape = shapely.union_all(squares).simplify(CELL, preserve_topology=True)
    if shape.geom_type != "Polygon":
        shape = max(shape.geoms, key=lambda g: g.area)
    return Polygon(shape.exterior)


def furniture(hmap: np.ndarray, bounds) -> list[dict]:
    """Buildings (room metres) from the height map: one per connected piece of each height
    band, so a desk against a wall or a cabinet is a piece of its own."""
    x0, y0, x1, y1 = bounds
    ny, nx = hmap.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = x0 + (xx + 0.5) * CELL, y0 + (yy + 0.5) * CELL
    inside = ((cx > x0 + WALL_MARGIN) & (cx < x1 - WALL_MARGIN)
              & (cy > y0 + WALL_MARGIN) & (cy < y1 - WALL_MARGIN))
    out = []
    for band_k, (lo, hi) in enumerate(BANDS):
        band = (hmap >= lo) & (hmap < hi) & inside
        labels, n = ndimage.label(ndimage.binary_opening(band, iterations=1))
        for k in range(1, n + 1):
            cells = labels == k
            if cells.sum() * CELL**2 < MIN_AREA:
                continue
            h = hmap[cells]
            hist, edges = np.histogram(h, bins=max(1, int((h.max() - h.min()) / 0.02)) or 1)
            top = float(edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2)  # the surface, not its clutter
            level = np.abs(h - top) < FLAT_TOP
            flat = level.sum() * CELL**2 >= 0.1
            out.append({"footprint": list(outline(cells, bounds).exterior.coords)[:-1], "height": top, "band": band_k,
                        "flat": flat, "top": (float(cx[cells][level].mean()), float(cy[cells][level].mean()))})
    return out


def top_view(parts: list[dict], bounds, px_per_m: float = 200.0) -> Image.Image:
    """Orthographic photo from above: every triangle in its texture colour, lowest first."""
    x0, y0, x1, y1 = bounds
    w, h = int((x1 - x0) * px_per_m), int((y1 - y0) * px_per_m)
    img = Image.new("RGB", (w, h), (40, 40, 44))
    draw = ImageDraw.Draw(img)
    tris, cols, zs = [], [], []
    for p in parts:
        t = p["v"][p["f"]]
        if p["uv"] is not None and p["image"] is not None:
            tex = np.asarray(p["image"].convert("RGB"))
            uv = p["uv"][p["f"]].mean(axis=1) % 1.0
            c = tex[((1 - uv[:, 1]) * (tex.shape[0] - 1)).astype(int), (uv[:, 0] * (tex.shape[1] - 1)).astype(int)]
        else:
            c = np.full((len(t), 3), 160)
        tris.append(t)
        cols.append(c)
        zs.append(t[:, :, 2].max(axis=1))
    tris, cols, zs = np.concatenate(tris), np.concatenate(cols), np.concatenate(zs)
    px = np.stack([(tris[:, :, 0] - x0) * px_per_m, (y1 - tris[:, :, 1]) * px_per_m], axis=-1)
    for k in np.argsort(zs):
        draw.polygon([tuple(q) for q in px[k]], fill=tuple(int(c) for c in cols[k]))
    return img


def write_obj(p: dict, out: Path, tex_name: str | None) -> None:
    lines = [f"v {x:.2f} {y:.2f} {z:.2f}" for x, y, z in p["v"]]
    if tex_name:
        lines += [f"vt {u:.5f} {v:.5f}" for u, v in p["uv"]]
        lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in p["f"]]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in p["f"]]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", type=Path, help=".glb / .gltf / .obj from Polycam")
    ap.add_argument("--up", choices=("y", "z"), default="y", help="the scan's up axis")
    ap.add_argument("--rotate", type=float, default=0.0,
                    help="degrees counter-clockwise after squaring, so that +y is north")
    ap.add_argument("--no-square", action="store_true", help="keep the scan's heading as it is")
    ap.add_argument("--ceiling", type=float, default=2.1, help="cut above this, room metres")
    ap.add_argument("--look", choices=("model", "scan"), default="model",
                    help="what Webots draws: the office modelled clean (tools/office_model.py) or the raw scan")
    ap.add_argument("--layout", type=Path, default=None,
                    help="walls, windows, rooms the scan misses (default: <out>_layout.yaml next to --out)")
    ap.add_argument("--out", type=Path, default=ROOT / "data/cities/office.json")
    a = ap.parse_args()

    parts = load_parts(a.scan, a.up)
    yaw = (0.0 if a.no_square else -wall_angle(parts)) + a.rotate
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for p in parts:
        p["v"] = p["v"] @ rot.T
    z0 = floor_z(parts)
    for p in parts:
        p["v"][:, 2] -= z0
        keep = (p["v"][p["f"], 2].min(axis=1) < a.ceiling) & (p["v"][p["f"], 2].max(axis=1) > -0.1)
        p["f"] = p["f"][keep]
    parts = [p for p in parts if len(p["f"])]
    allv = np.concatenate([p["v"][np.unique(p["f"])] for p in parts])
    lo, hi = np.percentile(allv[:, :2], [0.5, 99.5], axis=0)  # stray points through a window
    mid = (lo + hi) / 2
    for p in parts:
        p["v"][:, :2] -= mid
    room = (float(lo[0] - mid[0]), float(lo[1] - mid[1]), float(hi[0] - mid[0]), float(hi[1] - mid[1]))

    hmap, seen = height_map(parts, room)
    pieces = furniture(hmap, room)
    name = a.out.stem
    city_dir = a.out.parent
    twin = city_dir / f"{name}_twin"
    twin.mkdir(parents=True, exist_ok=True)
    top = top_view(parts, room)
    top.save(city_dir / f"{name}_top.jpg", quality=88)

    for old in twin.glob("scan_*"):
        old.unlink()
    interior = []
    if a.look == "model":
        layout_file = a.layout or a.out.with_name(f"{name}_layout.yaml")
        layout = yaml.safe_load(layout_file.read_text()) if layout_file.exists() else None
        if layout:
            print(f"layout: {layout_file}")
        interior = office_model.model(seen, CELL, room, pieces, twin, SCALE, layout)
    for k, p in enumerate(parts if a.look == "scan" else []):
        tex = None
        if p["uv"] is not None and p["image"] is not None:
            tex = f"scan_{k}.jpg"
            p["image"].convert("RGB").save(twin / tex, quality=90)
        p["v"] = p["v"] * SCALE
        write_obj(p, twin / f"scan_{k}.obj", tex)
        interior.append({"mesh": f"{twin.name}/scan_{k}.obj", "rgb": [0.6, 0.6, 0.6], "rough": 1.0}
                        | ({"photo": f"{twin.name}/{tex}"} if tex else {}))

    buildings, perch = [], []
    for k, b in enumerate(sorted(pieces, key=lambda b: (-b["top"][1], b["top"][0]))):
        bid = f"f{k}"
        buildings.append({"id": bid, "footprint": [[round(x * SCALE, 1), round(y * SCALE, 1)] for x, y in b["footprint"]],
                          "height": round(b["height"] * SCALE, 1)})
        if b["flat"]:
            perch.append({"id": f"p{k}", "building": bid, "x": round(b["top"][0] * SCALE, 1),
                          "y": round(b["top"][1] * SCALE, 1), "z": round(b["height"] * SCALE, 1)})
    city = {
        "name": name,
        "origin": {"lat": 0.0, "lon": 0.0, "note": f"{a.scan.name}, 1 m room = {SCALE:g} m city"},
        "bounds": [round(v * SCALE, 1) for v in room],
        "buildings": buildings,
        "perch_sites": perch,
        "rails": [],
        "roads": [],
        "basemap": f"{name}_top.jpg",
        "interior": interior,
    }
    a.out.write_text(json.dumps(city, indent=1))
    w, d = room[2] - room[0], room[3] - room[1]
    print(f"{a.out}: room {w:.2f} x {d:.2f} m (city {w * SCALE:.0f} x {d * SCALE:.0f} m), turned {yaw:.1f} deg, "
          f"{len(buildings)} pieces of furniture, {len(interior)} {a.look} meshes in {twin.name}/")
    for b, p in ((b, next((p for p in perch if p["building"] == b["id"]), None)) for b in buildings):
        where = f"perch {p['id']} at x {p['x']:.0f}, y {p['y']:.0f}" if p else "not flat"
        print(f"  {b['id']}: {b['height'] / SCALE:.2f} m high, {where}")
    print(f"check {name}_top.jpg: north (+y) is up; else run again with --rotate 90 / 180 / 270")


if __name__ == "__main__":
    main()

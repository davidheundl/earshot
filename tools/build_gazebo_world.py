"""City JSON + scenario -> Gazebo Sim world (SDF 1.9) plus its assets, with the model names
GazeboBackend expects (DRONE_<id>, SRC_<id>[_w<k>], CAMV, CAM_<k>).

    uv run python tools/build_gazebo_world.py scenarios/demo.yaml gazebo/worlds/demo.sdf

The same city as the Webots world (tools/build_webots_world.py makes the meshes and textures:
terrain texture or aerial photo, LoD2 walls and photo roofs, trees, rails, aircraft, street
props), drawn with Gazebo's own primitives: terrain as a textured mesh, extruded footprints
merged into one wall and one roof mesh when the City has no LoD2 meshes, roads as flat
ribbons, quadcopters from boxes, cylinders and ellipsoids. Local files only, no Fuel models
(nothing to download at the venue). Every model is static: GazeboBackend sets the poses.

Sources carry a Label (gazebo.LABELS) on their body visuals only, so the Verifier's
bounding-box camera reports the class and never the intruder's red marker beam.
Assets go to <world>_assets/ next to the world; a hash skips rebuilding when nothing changed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely import Polygon, constrained_delaunay_triangles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import build_webots_world as wb  # the shared mesh and texture makers

from earshot.backend.gazebo import (
    BASIC_MS,
    LABELS,
    LIVE_SLOTS,
    N_VIEWS,
    RECOGNITION_RANGE,
    TRAIN_WAGONS,
    VERIFY_FOV,
    VERIFY_SIZE,
    VIEW_FOV,
    VIEW_SIZE,
    live_slot,
)
from earshot.city import City
from earshot.scenario import Scenario

BUILDER_VERSION = "3"
INTRUDER_RGB = wb.INTRUDER_RGB
GRASS = (0.42, 0.55, 0.3)
TREE_RGB = {"conifer_dark": (0.10, 0.22, 0.12), "conifer": (0.14, 0.29, 0.14),
            "broadleaf": (0.20, 0.36, 0.13), "broadleaf_light": (0.30, 0.42, 0.16),
            "trunk": (0.30, 0.22, 0.15)}


# -- SDF snippets -------------------------------------------------------------------------------

def f3(v) -> str:
    return " ".join(f"{x:.4g}" for x in v)


def material(rgb, rough=0.9, texture=None, emissive=None, metal=0.0) -> str:
    s = (f"<material><ambient>{f3(rgb)} 1</ambient><diffuse>{f3(rgb)} 1</diffuse>"
         f"<specular>0.05 0.05 0.05 1</specular>")
    if emissive:
        s += f"<emissive>{f3(emissive)} 1</emissive>"
    s += f"<pbr><metal><roughness>{rough}</roughness><metalness>{metal}</metalness>"
    if texture:
        s += f"<albedo_map>{texture}</albedo_map>"
    return s + "</metal></pbr></material>"


def visual(name, geometry, look, pose=(0, 0, 0, 0, 0, 0), transparency=0.0, label=None) -> str:
    plug = (f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">'
            f"<label>{label}</label></plugin>") if label is not None else ""
    tr = f"<transparency>{transparency}</transparency>" if transparency else ""
    return (f'<visual name="{name}"><pose>{f3(pose)}</pose><cast_shadows>false</cast_shadows>'
            f"{tr}<geometry>{geometry}</geometry>{look}{plug}</visual>")


def box(x, y, z) -> str:
    return f"<box><size>{x:.4g} {y:.4g} {z:.4g}</size></box>"


def cyl(r, h) -> str:
    return f"<cylinder><radius>{r:.4g}</radius><length>{h:.4g}</length></cylinder>"


def mesh(uri) -> str:
    return f"<mesh><uri>{uri}</uri></mesh>"


def model(name, visuals, pose=(0, 0, -100, 0, 0, 0), extra="") -> str:
    return (f'<model name="{name}"><static>true</static><pose>{f3(pose)}</pose>'
            f'<link name="body">{"".join(visuals)}{extra}</link></model>')


# -- drones and sources -----------------------------------------------------------------------
# Each body is one mesh with its colours in an MTL file: the Verifier's label render costs
# per visual in the scene (~0.2 ms each on a laptop), so a 25-part quadcopter must not be
# 25 visuals.

def _rot(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


class Parts:
    """Primitives merged into one OBJ (+ MTL), one group per material."""

    def __init__(self, materials: dict[str, tuple]):
        self.materials = materials  # name -> (rgb, emissive rgb | None, opacity)
        self.groups: dict[str, list] = {}

    def add(self, mtl, v, n, f, pose=(0, 0, 0, 0, 0, 0)):
        R = _rot(pose[3:])
        v = np.asarray(v, float) @ R.T + pose[:3]
        n = np.asarray(n, float) @ R.T
        self.groups.setdefault(mtl, []).append((v, n, np.asarray(f)))

    def box(self, mtl, size, pose=(0, 0, 0, 0, 0, 0)):
        h = np.asarray(size, float) / 2
        v, n, f = [], [], []
        for axis in range(3):
            for sgn in (1, -1):
                u, w = [a for a in range(3) if a != axis]
                base = len(v)
                for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    p = np.zeros(3)
                    p[axis], p[u], p[w] = sgn * h[axis], a * h[u], b * h[w]
                    v.append(p)
                    nn = np.zeros(3)
                    nn[axis] = sgn
                    n.append(nn)
                ccw = np.cross(np.eye(3)[u], np.eye(3)[w])[axis] * sgn > 0
                f += ([[base, base + 1, base + 2], [base, base + 2, base + 3]] if ccw
                      else [[base, base + 2, base + 1], [base, base + 3, base + 2]])
        self.add(mtl, v, n, f, pose)

    def ellipsoid(self, mtl, radii, pose=(0, 0, 0, 0, 0, 0), nu=16, nv=10):
        rx, ry, rz = radii
        v, n, f = [], [], []
        for i in range(nv + 1):
            th = math.pi * i / nv
            for j in range(nu):
                ph = 2 * math.pi * j / nu
                d = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
                v.append(d * radii)
                nn = d / np.array([rx, ry, rz])
                n.append(nn / (np.linalg.norm(nn) or 1))
        for i in range(nv):
            for j in range(nu):
                a, b = i * nu + j, i * nu + (j + 1) % nu
                f += [[a, a + nu, b], [b, a + nu, b + nu]]
        self.add(mtl, v, n, f, pose)

    def cylinder(self, mtl, r, h, pose=(0, 0, 0, 0, 0, 0), k=16):
        v, n, f = [], [], []
        for j in range(k):  # side
            a = 2 * math.pi * j / k
            c, s = math.cos(a), math.sin(a)
            v += [[r * c, r * s, -h / 2], [r * c, r * s, h / 2]]
            n += [[c, s, 0], [c, s, 0]]
        for j in range(k):
            a, b = 2 * j, 2 * ((j + 1) % k)
            f += [[a, b, a + 1], [b, b + 1, a + 1]]
        for z, sgn in ((h / 2, 1), (-h / 2, -1)):  # caps
            centre = len(v)
            v.append([0, 0, z])
            n.append([0, 0, sgn])
            for j in range(k):
                a = 2 * math.pi * j / k
                v.append([r * math.cos(a), r * math.sin(a), z])
                n.append([0, 0, sgn])
            for j in range(k):
                a, b = centre + 1 + j, centre + 1 + (j + 1) % k
                f.append([centre, a, b] if sgn > 0 else [centre, b, a])
        self.add(mtl, v, n, f, pose)

    def write(self, path: Path) -> Path:
        mtl = path.with_suffix(".mtl")
        lines = []
        for name, (rgb, glow, opacity) in self.materials.items():
            lines += [f"newmtl {name}", f"Kd {f3(rgb)}", f"Ka {f3(rgb)}", "Ks 0.05 0.05 0.05"]
            lines += [f"Ke {f3(glow)}"] if glow else []
            lines += [f"d {opacity}"] if opacity < 1 else []
        mtl.write_text("\n".join(lines) + "\n")
        out, vt, k = [f"mtllib {mtl.name}"], [], 1
        body = []
        for name, parts in self.groups.items():
            body.append(f"usemtl {name}")
            for v, n, f in parts:
                out += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in v]
                vt += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in n]
                body += [f"f {a + k}//{a + k} {b + k}//{b + k} {c + k}//{c + k}" for a, b, c in f]
                k += len(v)
        path.write_text("\n".join(out + vt + body) + "\n")
        return path


QUAD_LOOKS = {"SWARM": ((1.0, 0.32, 0.02), 1.0), "INTRUDER": ((0.13, 0.13, 0.14), 1.3)}


def quad_mesh(kind: str, path: Path) -> Path:
    """A quadcopter, x forward: shell, canopy, arms, motors, blurred props, nav lights,
    gimbal camera; the proportions of the Webots drone."""
    rgb, k = QUAD_LOOKS[kind]
    m = Parts({"shell": (rgb, [c * 0.25 for c in rgb] if kind == "SWARM" else None, 1.0),
               "carbon": ((0.05, 0.05, 0.06), None, 1.0), "metal": ((0.2, 0.2, 0.22), None, 1.0),
               "prop": ((0.85, 0.85, 0.88), None, 0.22), "red": ((1, 0.05, 0.05), (1, 0.05, 0.05), 1.0),
               "green": ((0.1, 1, 0.2), (0.1, 1, 0.2), 1.0), "white": ((1, 1, 1), (1, 1, 1), 1.0)})
    m.ellipsoid("shell", (0.17 * k, 0.095 * k, 0.06 * k))
    m.ellipsoid("shell", (0.12 * k, 0.075 * k, 0.04 * k), (-0.01 * k, 0, 0.035 * k, 0, 0, 0))
    m.box("carbon", (0.12 * k, 0.08 * k, 0.014 * k), (-0.05 * k, 0, 0.07 * k, 0, 0, 0))
    m.ellipsoid("carbon", (0.035 * k,) * 3, (0.15 * k, 0, -0.035 * k, 0, 0, 0), 10, 6)
    for side in (1, -1):
        m.box("carbon", (0.2 * k, 0.012 * k, 0.012 * k), (-0.08 * k, 0.07 * side * k, -0.1 * k, 0, 0, 0))
        m.box("carbon", (0.012 * k, 0.012 * k, 0.1 * k), (-0.08 * k, 0.07 * side * k, -0.05 * k, 0, 0, 0))
    r = 0.32
    for a_deg, light in ((45, "red"), (-45, "green"), (135, None), (-135, None)):
        a = math.radians(a_deg)
        c, s = math.cos(a), math.sin(a)
        m.box("carbon", (r * k, 0.03 * k, 0.02 * k), (c * r / 2 * k, s * r / 2 * k, 0.01 * k, 0, 0, a))
        m.cylinder("metal", 0.032 * k, 0.05 * k, (c * r * k, s * r * k, 0.03 * k, 0, 0, 0), 12)
        m.cylinder("shell", 0.034 * k, 0.006 * k, (c * r * k, s * r * k, 0.052 * k, 0, 0, 0), 12)
        m.cylinder("prop", 0.17 * k, 0.003 * k, (c * r * k, s * r * k, 0.058 * k, 0, 0, 0), 24)
        if light:
            m.ellipsoid(light, (0.018 * k,) * 3, (c * (r + 0.04) * k, s * (r + 0.04) * k, 0.02 * k, 0, 0, 0), 8, 4)
    m.ellipsoid("white", (0.016 * k,) * 3, (-0.13 * k, 0, 0.03 * k, 0, 0, 0), 8, 4)
    return m.write(path)


def marker_mesh(rgb, path: Path) -> Path:
    """Glowing ring and beam over an intruder: its own visual without a Label, so the
    Verifier camera never reports it."""
    m = Parts({"glow": (rgb, rgb, 0.5)})
    m.cylinder("glow", 1.3, 0.04, k=24)
    m.cylinder("glow", 0.1, 28, (0, 0, 14, 0, 0, 0), 8)
    return m.write(path)


def truck_mesh(path: Path) -> Path:
    m = Parts({"body": ((0.55, 0.3, 0.12), None, 1.0), "box": ((0.85, 0.85, 0.82), None, 1.0),
               "glass": ((0.06, 0.08, 0.1), None, 1.0), "tyre": ((0.05, 0.05, 0.05), None, 1.0)})
    m.box("body", (2.2, 2.4, 2.6), (3.6, 0, 1.6, 0, 0, 0))
    m.box("glass", (0.05, 2.0, 0.9), (4.72, 0, 2.2, 0, 0, 0))
    m.box("box", (6.0, 2.5, 3.0), (-0.6, 0, 1.9, 0, 0, 0))
    for x in (3.6, -1.6, -2.8):
        for side in (1, -1):
            m.cylinder("tyre", 0.5, 0.35, (x, 1.1 * side, 0, math.pi / 2, 0, 0), 14)
    return m.write(path)


WAGON_RGB = [(0.12, 0.3, 0.55), (0.6, 0.45, 0.15), (0.2, 0.45, 0.25), (0.55, 0.55, 0.55), (0.5, 0.2, 0.15)]


def wagon_mesh(k: int, path: Path) -> Path:
    """k = 0: the locomotive; else a container wagon in colour k mod 5."""
    if k == 0:
        m = Parts({"body": ((0.62, 0.12, 0.1), None, 1.0), "roof": ((0.2, 0.2, 0.22), None, 1.0)})
        m.box("body", (18, 3, 3.4), (0, 0, 2.4, 0, 0, 0))
        m.box("roof", (16, 2.6, 0.4), (0, 0, 4.3, 0, 0, 0))
    else:
        m = Parts({"flat": ((0.25, 0.25, 0.27), None, 1.0), "box": (WAGON_RGB[k % 5], None, 1.0)})
        m.box("flat", (19, 2.9, 0.6), (0, 0, 1.2, 0, 0, 0))
        m.box("box", (12.2, 2.45, 2.6), (0, 0, 2.8, 0, 0, 0))
    return m.write(path)


def fan_mesh(path: Path) -> Path:
    m = Parts({"housing": ((0.62, 0.63, 0.64), None, 1.0), "grille": ((0.2, 0.2, 0.2), None, 1.0)})
    m.box("housing", (2.4, 1.6, 1.4))
    m.cylinder("grille", 0.6, 0.1, (0, 0, 0.75, 0, 0, 0), 20)
    return m.write(path)


def body_meshes(assets: Path, markers: bool) -> dict[str, Path]:
    out = {"SWARM": quad_mesh("SWARM", assets / "quad_swarm.obj"),
           "DRONE": quad_mesh("INTRUDER", assets / "quad_intruder.obj"),
           "VEHICLE": truck_mesh(assets / "truck.obj"), "NOTHING": fan_mesh(assets / "fan.obj")}
    for k in range(min(TRAIN_WAGONS, 6)):
        out[f"wagon{k}"] = wagon_mesh(k, assets / f"wagon{k}.obj")
    if markers:
        out["marker"] = marker_mesh(INTRUDER_RGB, assets / "marker.obj")
    return out


def body(uri: Path, label=None, name="body") -> str:
    return visual(name, mesh(uri), "", label=label)


def source_models(sid: str, cls: str, bodies: dict[str, Path]) -> list[str]:
    lb = LABELS[cls]
    if cls == "TRAIN":
        return [model(f"SRC_{sid}_w{k}", [body(bodies[f"wagon{(k - 1) % 5 + 1 if k else 0}"], lb)])
                for k in range(TRAIN_WAGONS)]
    parts = [body(bodies[cls], lb)]
    if cls == "DRONE" and "marker" in bodies:
        parts.append(body(bodies["marker"], name="marker"))
    return [model(f"SRC_{sid}", parts)]


# -- cameras ----------------------------------------------------------------------------------

def camera(name, topic, size, fov, rate, far, kind="camera") -> str:
    box_type = "<box_type>2d</box_type>" if kind == "boundingbox_camera" else ""
    return (f'<sensor name="{name}" type="{kind}"><always_on>true</always_on>'
            f"<update_rate>{rate}</update_rate><topic>{topic}</topic><camera>{box_type}"
            f"<horizontal_fov>{fov}</horizontal_fov><image><width>{size[0]}</width>"
            f"<height>{size[1]}</height><format>R8G8B8</format></image>"
            f"<clip><near>0.05</near><far>{far}</far></clip></camera></sensor>")


def cameras() -> list[str]:
    """CAMV: the Verifier's box camera (what it recognises, out to RECOGNITION_RANGE) and a
    plain camera for the dashboard preview; CAM_0..3: the game screen's view cameras."""
    out = [model("CAMV", [], extra=(
        camera("verify", "/earshot/verify/boxes", VERIFY_SIZE, VERIFY_FOV, 10, RECOGNITION_RANGE,
               "boundingbox_camera")
        + camera("preview", "/earshot/verify/image", VERIFY_SIZE, VERIFY_FOV, 2, 3000)))]
    for k in range(N_VIEWS):
        out.append(model(f"CAM_{k}", [], extra=camera(
            f"view{k}", f"/earshot/view{k}/image", VIEW_SIZE[k], VIEW_FOV, 10 if k == 0 else 5, 9000)))
    return out


# -- ground and city meshes -------------------------------------------------------------------

def grid_obj(path: Path, origin, spacing, heights: np.ndarray) -> None:
    """A height grid (row = y) as a mesh with UVs spanning the grid, texture north up."""
    n = heights.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    x, y = origin[0] + spacing * i, origin[1] + spacing * j
    v = np.stack([x.ravel(), y.ravel(), heights.ravel()], 1)
    uv = np.stack([i.ravel() / (n - 1), j.ravel() / (n - 1)], 1)
    idx = (j * n + i)[:-1, :-1].ravel() + 1
    faces = np.concatenate([np.stack([idx, idx + 1, idx + n + 1], 1),
                            np.stack([idx, idx + n + 1, idx + n], 1)])
    with open(path, "w") as fh:
        np.savetxt(fh, v, fmt="v %.2f %.2f %.2f")
        np.savetxt(fh, uv, fmt="vt %.5f %.5f")
        np.savetxt(fh, np.repeat(faces, 2, axis=1), fmt="f %d/%d %d/%d %d/%d")


def merge_trees(tiles: list[tuple[str, str]], assets: Path) -> list[tuple[str, str]]:
    """The Webots builder's tree tiles (for camera culling) merged into one mesh per species:
    in Gazebo the number of visuals is what costs."""
    by_species: dict[str, list[str]] = {}
    for name, file in tiles:
        by_species.setdefault(name, []).append(file)
    out = []
    for name, files in sorted(by_species.items()):
        v, f, k = [], [], 0
        for file in files:
            lines = (assets / file).read_text().splitlines()
            vs = [ln for ln in lines if ln.startswith("v ")]
            v += vs
            f += ["f " + " ".join(str(int(i) + k) for i in ln.split()[1:]) for ln in lines if ln.startswith("f ")]
            k += len(vs)
            (assets / file).unlink()
        merged = f"trees_{name}.obj"
        (assets / merged).write_text("\n".join(v + f) + "\n")
        out.append((name, merged))
    return out


def add_normals(path: Path) -> None:
    """Area-weighted vertex normals for an OBJ without any: Gazebo draws such a mesh unlit white."""
    text = path.read_text()
    if "\nvn " in text or not text.strip():
        return
    v, vt, faces = [], [], []
    for ln in text.splitlines():
        if ln.startswith("v "):
            v.append(ln)
        elif ln.startswith("vt "):
            vt.append(ln)
        elif ln.startswith("f "):
            faces.append(ln.split()[1:])
    pos = np.array([[float(x) for x in ln.split()[1:4]] for ln in v])
    tri = np.array([[int(c.split("/")[0]) - 1 for c in f[:3]] for f in faces])
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    nrm = np.zeros_like(pos)
    for k in range(3):
        np.add.at(nrm, tri[:, k], fn)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    out = v + vt + [f"vn {x:.3f} {y:.3f} {z:.3f}" for x, y, z in nrm]
    for f in faces:
        corners = []
        for corner in f:
            parts = corner.split("/")
            corners.append(f"{parts[0]}/{parts[1] if len(parts) > 1 else ''}/{parts[0]}")
        out.append("f " + " ".join(corners))
    path.write_text("\n".join(out) + "\n")


def extruded_buildings(buildings: list[dict], assets: Path) -> None:
    """Footprints without LoD2 meshes: one wall mesh (facade UVs) and one flat roof mesh."""
    walls, roofs = [], wb.Obj()
    for b in buildings:
        poly = Polygon(b["footprint"]).buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        z0 = b.get("base_z", 0.0) - 0.5
        z1 = b.get("base_z", 0.0) + b["height"]
        ring = list(poly.exterior.coords)
        if not poly.exterior.is_ccw:
            ring = ring[::-1]
        for (ax, ay), (bx, by) in itertools.pairwise(ring):
            walls += [(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z0), (bx, by, z1), (ax, ay, z1)]
        for tri in constrained_delaunay_triangles(poly).geoms:
            pts = list(tri.exterior.coords)[:3]
            if not Polygon(pts).exterior.is_ccw:
                pts = pts[::-1]
            roofs.add([(x, y, z1) for x, y in pts], [[0, 1, 2]])
    raw = assets / "walls_raw.obj"
    raw.write_text("".join(f"v {x:.2f} {y:.2f} {z:.2f}\n" for x, y, z in walls))
    wb.facade_walls(raw, assets / "walls.obj")
    raw.unlink()
    roofs.write(assets / "roofs.obj")


def roads_obj(city: City, widths, path: Path) -> None:
    """Roads as flat ribbons 0.15 m over the ground."""
    o = wb.Obj()
    for line, w in zip(city.roads, widths, strict=True):
        pts = wb.densify(list(line.coords), 10.0)
        for (ax, ay), (bx, by) in itertools.pairwise(pts):
            L = math.hypot(bx - ax, by - ay) or 1.0
            nx, ny = -(by - ay) / L * w / 2, (bx - ax) / L * w / 2
            za, zb = float(city.ground(ax, ay)) + 0.15, float(city.ground(bx, by)) + 0.15
            o.add([(ax - nx, ay - ny, za), (bx - nx, by - ny, zb), (bx + nx, by + ny, zb), (ax + nx, ay + ny, za)],
                  [[0, 1, 2], [0, 2, 3]])
    o.write(path)


def look_of(prefix: str, name: str) -> str:
    if prefix == "AC":
        if name.startswith("tail"):
            return material(wb.TAIL_RGB[int(name[4:])], 0.5)
        rgb, rough = wb.AIRCRAFT_LOOK[name]
        return material(rgb, rough)
    rgb, rough, extra = wb.PROP_LOOK[name]
    return material(rgb, rough, metal=0.6 if "metalness" in extra else 0.0)


def gui(cx, cy, z, look_from_y) -> str:
    """Gazebo's window: the 3D scene looking north over the city, the usual tools."""
    pitch = math.atan2(z, cy - look_from_y)
    return f"""<gui fullscreen="0">
  <plugin filename="MinimalScene" name="3D View">
    <gz-gui><title>3D View</title><property type="bool" key="showTitleBar">false</property>
      <property type="string" key="state">docked</property></gz-gui>
    <engine>ogre2</engine><scene>scene</scene>
    <ambient_light>0.5 0.5 0.5</ambient_light><background_color>0.72 0.8 0.88</background_color>
    <camera_pose>{cx:.0f} {look_from_y:.0f} {z:.0f} 0 {pitch:.3f} 1.5708</camera_pose>
    <camera_clip><near>0.5</near><far>25000</far></camera_clip>
  </plugin>
  <plugin filename="GzSceneManager" name="Scene Manager">
    <gz-gui><property key="state" type="string">floating</property>
      <property key="width" type="double">5</property><property key="height" type="double">5</property>
      <property key="showTitleBar" type="bool">false</property></gz-gui>
  </plugin>
  <plugin filename="InteractiveViewControl" name="Interactive view control">
    <gz-gui><property key="state" type="string">floating</property>
      <property key="width" type="double">5</property><property key="height" type="double">5</property>
      <property key="showTitleBar" type="bool">false</property></gz-gui>
  </plugin>
  <plugin filename="CameraTracking" name="Camera Tracking">
    <gz-gui><property key="state" type="string">floating</property>
      <property key="width" type="double">5</property><property key="height" type="double">5</property>
      <property key="showTitleBar" type="bool">false</property></gz-gui>
  </plugin>
  <plugin filename="SelectEntities" name="Select Entities">
    <gz-gui><property key="state" type="string">floating</property>
      <property key="width" type="double">5</property><property key="height" type="double">5</property>
      <property key="showTitleBar" type="bool">false</property></gz-gui>
  </plugin>
  <plugin filename="WorldStats" name="World stats">
    <gz-gui><title>World stats</title><property type="bool" key="showTitleBar">false</property>
      <property type="bool" key="resizable">false</property>
      <property type="double" key="height">110</property><property type="double" key="width">290</property>
      <property type="double" key="z">1</property><property type="string" key="state">floating</property>
      <anchors target="3D View"><line own="right" target="right"/><line own="bottom" target="bottom"/></anchors>
    </gz-gui>
    <sim_time>true</sim_time><real_time>true</real_time><real_time_factor>true</real_time_factor>
    <iterations>true</iterations>
  </plugin>
  <plugin filename="EntityTree" name="Entity tree"/>
</gui>"""


# -- main -------------------------------------------------------------------------------------

def build(scenario_path: str, out_path: str, markers: bool = True, force: bool = False) -> Path:
    sc = Scenario.load(scenario_path)
    city_file = Path(sc.city) if Path(sc.city).is_absolute() else ROOT / sc.city
    raw = json.loads(city_file.read_text())
    city = City(raw)
    out = Path(out_path).resolve()
    assets = out.with_name(out.stem + "_assets")
    assets.mkdir(parents=True, exist_ok=True)
    side = [t["image"] for t in raw.get("ground_tiles", [])]
    side += [f for m in raw.get("building_meshes", []) for f in m.values()]
    side += [raw["features"]] if raw.get("features") else []
    key = hashlib.sha256(city_file.read_bytes() + Path(scenario_path).read_bytes()
                         + b"".join((city_file.parent / f).read_bytes() for f in side)
                         + Path(__file__).read_bytes() + Path(wb.__file__).read_bytes()
                         + f"{markers}{BUILDER_VERSION}{LIVE_SLOTS}{N_VIEWS}{TRAIN_WAGONS}{out}".encode()
                         ).hexdigest()
    stamp = assets / "build.hash"
    if not force and out.exists() and stamp.exists() and stamp.read_text() == key:
        print(f"{out}: up to date")
        return out

    a = assets  # mesh and texture URIs are absolute: the world is generated per machine
    x0, y0, x1, y1 = city.bounds
    span = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lu = wb.landuse(raw, city_file.parent, city)
    photo = raw.get("ground_tiles")
    # the Webots builder grows on `prototype` (street props, OSM trees, cleaned photos): use
    # what this checkout's version offers, so both branches keep building
    aircraft = wb.stand_aircraft(raw, city, assets)
    aircraft, stands = aircraft if isinstance(aircraft, tuple) else (aircraft, [])
    props, lots = (wb.street_props(raw, city, city_file.parent, assets, stands)
                   if hasattr(wb, "street_props") else ([], []))
    if photo:
        for t in photo:
            img = Image.open(city_file.parent / t["image"])
            if hasattr(wb, "clean_photo"):
                img = wb.clean_photo(img, t["bounds"], stands, lots)
            img.convert("RGB").save(assets / Path(t["image"]).name, quality=88)
    else:
        wb.paint_ground(lu, x1 - x0).save(assets / "ground.png")
    meshes = raw.get("building_meshes")
    for m in meshes or []:
        for kind, f in m.items():
            if kind == "roofs" and photo:
                wb.photo_roofs(city_file.parent / f, photo, city_file.parent, assets / Path(f).name)
            elif kind == "walls":
                wb.facade_walls(city_file.parent / f, assets / Path(f).name)
            else:
                (assets / Path(f).name).write_bytes((city_file.parent / f).read_bytes())
    wb.facade_texture(assets / "facade.png")
    if not meshes:
        extruded_buildings(raw["buildings"], assets)
    extra = {"extra": wb.osm_trees(raw, city_file.parent)} if hasattr(wb, "osm_trees") else {}
    tree_tiles = merge_trees(wb.trees(lu, city, assets, forest_only=bool(photo), **extra), assets)
    wb.rail_meshes(city, assets)

    scenery: list[str] = []  # visuals of the one static "city" model
    if "terrain" in raw:
        t = raw["terrain"]
        n = int(t["n"])
        grid_obj(assets / "terrain.obj", t["origin"], t["spacing"], np.asarray(t["heights"], float).reshape(n, n))
        scenery.append(visual("terrain", mesh(a / "terrain.obj"), material((1, 1, 1), 1, a / "ground.png")))
        outer = wb.outer_ring(raw, city)
        if outer is not None:
            wb.paint_outer().save(assets / "outer.png")
            to = raw["terrain_outer"]
            grid_obj(assets / "outer.obj", to["origin"], to["spacing"], outer)
            scenery.append(visual("outer", mesh(a / "outer.obj"), material((1, 1, 1), 1, a / "outer.png")))
    elif photo:
        for k, t in enumerate(photo):
            tx0, ty0, tx1, ty1 = t["bounds"]
            scenery.append(visual(f"photo{k}", box(tx1 - tx0, ty1 - ty0, 0.1),
                                  material((0.52, 0.52, 0.52), 1, a / Path(t["image"]).name),
                                  ((tx0 + tx1) / 2, (ty0 + ty1) / 2, -0.05, 0, 0, 0)))
        m_ = 2.5 * span
        for k, (fx0, fy0, fx1, fy1) in enumerate(((x0 - m_, y1, x1 + m_, y1 + m_), (x0 - m_, y0 - m_, x1 + m_, y0),
                                                  (x0 - m_, y0, x0, y1), (x1, y0, x1 + m_, y1))):
            scenery.append(visual(f"grass{k}", box(fx1 - fx0, fy1 - fy0, 0.1), material(GRASS, 1),
                                  ((fx0 + fx1) / 2, (fy0 + fy1) / 2, -0.05, 0, 0, 0)))
    else:
        scenery.append(visual("ground", box(x1 - x0, y1 - y0, 0.1), material((1, 1, 1), 1, a / "ground.png"),
                              (cx, cy, -0.05, 0, 0, 0)))
        scenery.append(visual("grass", box(6 * span, 6 * span, 0.1), material(GRASS, 1), (cx, cy, -0.2, 0, 0, 0)))
    for k, w in enumerate(raw.get("water", [])):
        scenery.append(visual(f"water{k}", cyl(w["r"] + 30, 0.1), material((0.12, 0.25, 0.33), 0.05),
                              (w["x"], w["y"], w["z"], 0, 0, 0), transparency=0.1))
    for name, file in tree_tiles:
        scenery.append(visual(Path(file).stem, mesh(a / file), material(TREE_RGB[name])))
    if city.rails:
        for part, rgb, rough in (("ballast", (0.42, 0.38, 0.34), 1.0), ("sleepers", (0.28, 0.2, 0.14), 1.0),
                                 ("steel", (0.55, 0.55, 0.58), 0.3)):
            scenery.append(visual(f"rail_{part}", mesh(a / f"rail_{part}.obj"), material(rgb, rough)))
    if city.roads and not photo:
        roads_obj(city, raw.get("road_widths", [8.0] * len(city.roads)), assets / "roads.obj")
        scenery.append(visual("roads", mesh(a / "roads.obj"), material((0.25, 0.25, 0.27), 0.95)))
    facade = material((1, 1, 1), 0.6, a / "facade.png")
    if meshes:
        for k, m in enumerate(meshes):
            if "walls" in m:
                scenery.append(visual(f"walls{k}", mesh(a / Path(m["walls"]).name), facade))
            if "roofs" in m:
                look = (material((0.52, 0.52, 0.52), 1, a / f"{Path(m['roofs']).stem}.jpg") if photo
                        else material((0.36, 0.37, 0.39), 0.6))
                scenery.append(visual(f"roofs{k}", mesh(a / Path(m["roofs"]).name), look))
    else:
        scenery.append(visual("walls", mesh(a / "walls.obj"), facade))
        scenery.append(visual("roofs", mesh(a / "roofs.obj"), material((0.36, 0.37, 0.39), 0.8)))
    for prefix, files in (("AC", aircraft), ("PROP", props)):
        for i, (name, file) in enumerate(files):
            scenery.append(visual(f"{prefix.lower()}{i}_{name}", mesh(a / file), look_of(prefix, name),
                                  transparency=0.6 if name == "fence" else 0.0))

    for obj in assets.glob("*.obj"):
        add_normals(obj)
    bodies = body_meshes(assets, markers)  # after add_normals: these carry their own
    models = [model("city", scenery, pose=(0, 0, 0, 0, 0, 0))]
    for d in sc.drones:
        z = float(city.ground(d.x, d.y)) + 40.0
        models.append(model(f"DRONE_{d.id}", [body(bodies["SWARM"])], pose=(d.x, d.y, z, 0, 0, 0)))
    slots = [(s.id, s.cls.name) for s in sc.sources] + [
        (live_slot(c, k), c) for c in ("DRONE", "VEHICLE", "TRAIN", "NOTHING") for k in range(LIVE_SLOTS)]
    for sid, cls in slots:
        models += source_models(sid, cls, bodies)
    models += cameras()

    look_from = y0 - 0.45 * span
    sdf = f"""<?xml version="1.0"?>
<!-- generated by tools/build_gazebo_world.py from {scenario_path} -- do not edit by hand -->
<sdf version="1.9">
<world name="{sc.name}">
  <physics name="puppet" type="ignored">
    <max_step_size>{BASIC_MS / 1000}</max_step_size><real_time_factor>0</real_time_factor>
  </physics>
  <gravity>0 0 0</gravity>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
    <render_engine>ogre2</render_engine>
  </plugin>
  <scene>
    <ambient>0.55 0.55 0.58 1</ambient><background>0.72 0.8 0.88 1</background>
    <shadows>false</shadows><grid>false</grid><sky></sky>
  </scene>
  <light type="directional" name="sun">
    <cast_shadows>false</cast_shadows><pose>0 0 1000 0 0 0</pose>
    <diffuse>1 0.97 0.92 1</diffuse><specular>0.3 0.3 0.3 1</specular>
    <direction>-0.4 0.3 -0.85</direction>
  </light>
  {gui(cx, cy, float(city.ground(cx, y0)) + 0.22 * span, look_from)}
  {chr(10).join("  " + m for m in models)}
</world>
</sdf>
"""
    out.write_text(sdf)
    stamp.write_text(key)
    print(f"{out}: {len(raw['buildings'])} buildings, {len(city.roads)} roads, {len(tree_tiles)} tree meshes, "
          f"{len(sc.drones)} drones, {len(sc.sources)} sources (assets in {assets.name}/)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="scenarios/demo.yaml")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--no-markers", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    name = Scenario.load(a.scenario).name
    build(a.scenario, a.out or str(ROOT / f"gazebo/worlds/{name}.sdf"), markers=not a.no_markers, force=a.force)

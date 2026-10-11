"""City map: the static prior knowledge every drone carries (buildings, Perch Sites, rail, roads).

Loaded from a City JSON file. That file is the hand-off between the data pipeline
(LoD2 + OSM -> JSON, Nikolas) and everything else. Schema, local metric frame, metres:

    {
      "name": "synthetic",
      "origin": {"lat": 48.35, "lon": 11.78, "epsg": 25832},   # informational
      "bounds": [xmin, ymin, xmax, ymax],
      "buildings": [{"id": "b12", "footprint": [[x, y], ...], "height": 14.0,
                     "base_z": 31.5}],                           # base_z optional (0)
      "perch_sites": [{"id": "p3", "building": "b12", "x": .., "y": .., "z": ..}],  # z absolute
      "rails": [[[x, y], ...], ...],
      "roads": [[[x, y], ...], ...],
      # v2, all optional (flat world without them):
      "terrain": {"origin": [x0, y0], "spacing": 10, "n": 401, "heights": [...]},  # row = y
      "terrain_outer": {...same, coarse, visual only...},
      "landuse": {"image": "valley_landuse.png", "classes": [...]},
      "water": [{"x":, "y":, "r":, "z":}], "towns": [...], "road_widths": [...],
      # building styles for Webots: "floors", "floor_height", "wall", "roof", "town"
    }

The same file feeds both backends (Webots world generator and the 2D twin) and the agents.
Heights are absolute (metres above the map datum); `ground(x, y)` gives the terrain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely import LineString, Point, Polygon, STRtree


@dataclass(frozen=True)
class Building:
    id: str
    footprint: Polygon
    height: float
    base_z: float = 0.0

    @property
    def top(self) -> float:
        return self.base_z + self.height


@dataclass(frozen=True)
class PerchSite:
    id: str
    building: str
    x: float
    y: float
    z: float


class City:
    def __init__(self, data: dict):
        self.name: str = data.get("name", "unnamed")
        self.origin: dict = data.get("origin", {})
        self.bounds: tuple[float, float, float, float] = tuple(data["bounds"])
        self.buildings = [
            Building(b["id"], Polygon(b["footprint"]), float(b["height"]),
                     float(b.get("base_z", 0.0)))
            for b in data["buildings"]
        ]
        t = data.get("terrain")
        self._terrain = None
        if t:
            n = int(t["n"])
            self._terrain = (float(t["origin"][0]), float(t["origin"][1]), float(t["spacing"]),
                             np.asarray(t["heights"], dtype=float).reshape(n, n))
        self.perch_sites = [PerchSite(**p) for p in data.get("perch_sites", [])]
        self.rails = [LineString(r) for r in data.get("rails", [])]
        self.roads = [LineString(r) for r in data.get("roads", [])]
        self._tree = STRtree([b.footprint for b in self.buildings])
        self._raw = data

    def ground(self, x, y):
        """Terrain height at (x, y), bilinear; 0 on a flat map. Works on arrays too."""
        if self._terrain is None:
            return np.zeros_like(np.asarray(x, dtype=float)) if np.ndim(x) else 0.0
        x0, y0, sp, H = self._terrain
        n = H.shape[0]
        i = np.clip((np.asarray(x, dtype=float) - x0) / sp, 0, n - 1.001)
        j = np.clip((np.asarray(y, dtype=float) - y0) / sp, 0, n - 1.001)
        i0, j0 = np.floor(i).astype(int), np.floor(j).astype(int)
        fi, fj = i - i0, j - j0
        z = ((1 - fi) * (1 - fj) * H[j0, i0] + fi * (1 - fj) * H[j0, i0 + 1]
             + (1 - fi) * fj * H[j0 + 1, i0] + fi * fj * H[j0 + 1, i0 + 1])
        return float(z) if np.ndim(z) == 0 else z

    def obstacles_between(self, a, b, za: float | None = None, zb: float | None = None) -> int:
        """Buildings and hills blocking the straight line from (a, za) to (b, zb).

        za / zb are absolute heights; None means 2 m above the ground there. A building
        counts only if its top is above the line where the line crosses it; the terrain
        counts once if it rises above the line anywhere (sampled every ~20 m)."""
        za = self.ground(*a) + 2.0 if za is None else za
        zb = self.ground(*b) + 2.0 if zb is None else zb
        seg = LineString([a, b])
        length = seg.length or 1.0
        n = 0
        for i in self._tree.query(seg, predicate="intersects"):
            bld = self.buildings[i]
            fp = bld.footprint
            if fp.covers(Point(a)) or fp.covers(Point(b)):
                continue
            mid = seg.intersection(fp).centroid
            u = seg.project(mid) / length
            if bld.top > za + u * (zb - za):
                n += 1
        if self._terrain is not None and length > 40:
            u = np.linspace(0.05, 0.95, max(3, int(length / 20)))
            xs = a[0] + u * (b[0] - a[0])
            ys = a[1] + u * (b[1] - a[1])
            if np.any(self.ground(xs, ys) > za + u * (zb - za) + 1.0):
                n += 1
        return n

    @classmethod
    def load(cls, path: str | Path) -> City:
        return cls(json.loads(Path(path).read_text()))

    def to_json(self) -> dict:
        return self._raw

    def buildings_between(self, a: tuple[float, float], b: tuple[float, float]) -> int:
        """How many building footprints the straight segment a->b crosses (2D).

        Endpoints sitting on a roof do not count that building as blocking.
        """
        seg = LineString([a, b])
        n = 0
        for i in self._tree.query(seg, predicate="intersects"):
            fp = self.buildings[i].footprint
            if fp.covers(Point(a)) or fp.covers(Point(b)):
                continue
            n += 1
        return n

    def distance_to_rail(self, x: float, y: float) -> float:
        p = Point(x, y)
        return min((r.distance(p) for r in self.rails), default=float("inf"))

    def distance_to_road(self, x: float, y: float) -> float:
        p = Point(x, y)
        return min((r.distance(p) for r in self.roads), default=float("inf"))

    def grid(self, cell: float) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates (xs, ys) for a probability grid over the bounds."""
        x0, y0, x1, y1 = self.bounds
        return np.arange(x0 + cell / 2, x1, cell), np.arange(y0 + cell / 2, y1, cell)

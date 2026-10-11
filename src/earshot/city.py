"""City map: the static prior knowledge every drone carries (buildings, Perch Sites, rail, roads).

Loaded from a City JSON file. That file is the hand-off between the data pipeline
(LoD2 + OSM -> JSON, Nikolas) and everything else. Schema, local metric frame, metres:

    {
      "name": "synthetic",
      "origin": {"lat": 48.35, "lon": 11.78, "epsg": 25832},   # informational
      "bounds": [xmin, ymin, xmax, ymax],
      "buildings": [{"id": "b12", "footprint": [[x, y], ...], "height": 14.0}],
      "perch_sites": [{"id": "p3", "building": "b12", "x": .., "y": .., "z": ..}],
      "rails": [[[x, y], ...], ...],
      "roads": [[[x, y], ...], ...]
    }

The same file feeds both backends (Webots world generator and the 2D twin) and the agents.
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
            Building(b["id"], Polygon(b["footprint"]), float(b["height"]))
            for b in data["buildings"]
        ]
        self.perch_sites = [PerchSite(**p) for p in data.get("perch_sites", [])]
        self.rails = [LineString(r) for r in data.get("rails", [])]
        self.roads = [LineString(r) for r in data.get("roads", [])]
        self._tree = STRtree([b.footprint for b in self.buildings])
        self._raw = data

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

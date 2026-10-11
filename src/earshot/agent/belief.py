"""Belief: one drone's Probability Map (per-cell log-odds per TargetClass) and its Tracks.

Interface:
    belief = Belief(city, cell=20.0)
    belief.add_evidence(obs)        Observation from anyone, own or received; each exactly once
    belief.set_track(track)         Track replaces the previous one with the same key
    belief.decay(dt)                old evidence fades for moving sources
    belief.tracks() -> list[Track]  current fused suspicions (what gets broadcast)
    belief.summary() -> dict        small JSON for the dashboard

ON THE DAY (judged): measurement models (bearing wedge, RF ring, camera blob), explaining
away with rail/road priors, the Faulty Sensor hypothesis, track extraction.
The stub keeps the grid shape and just counts evidence.
"""

from __future__ import annotations

import numpy as np

from earshot.city import City
from earshot.messages import Observation, TargetClass, Track


class Belief:
    def __init__(self, city: City, cell: float = 20.0):
        self.city = city
        self.cell = cell
        self.xs, self.ys = city.grid(cell)
        self.log_odds = np.zeros((len(TargetClass), len(self.ys), len(self.xs)), dtype=np.float32)
        self._tracks: dict[tuple[int, int], Track] = {}
        self.n_evidence = 0

    def add_evidence(self, obs: Observation) -> None:
        self.n_evidence += 1  # STUB

    def set_track(self, track: Track) -> None:
        self._tracks[(track.header.origin, track.track_id)] = track

    def decay(self, dt: float) -> None:
        pass  # STUB

    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def summary(self) -> dict:
        return {"n_evidence": self.n_evidence, "n_tracks": len(self._tracks)}

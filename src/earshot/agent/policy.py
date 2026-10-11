"""Policy: choose this drone's Mode and Command from its own Belief and the Claims it knows.

Interface:
    policy = Policy(me, city, home)
    command, claims = policy.decide(t, state, belief, known_claims)

ON THE DAY (judged): Perch Site choice by crossing angle, verification value vs. Gap cost,
lease-based Claim auction, Voronoi sector re-partitioning.
The stub below is a fixed duty cycle (patrol home -> perch at the nearest Perch Site ->
patrol ...) so the pipeline has motion and sensor traffic to carry.
"""

from __future__ import annotations

import math

from earshot.city import City
from earshot.messages import Claim, Mode
from earshot.state import Command, DroneState

PATROL_S = 30.0
LISTEN_S = 60.0


class Policy:
    def __init__(self, me: int, city: City, home: tuple[float, float]):
        self.me = me
        self.city = city
        self.home = home
        self._site = min(
            city.perch_sites,
            key=lambda p: math.hypot(p.x - home[0], p.y - home[1]),
            default=None,
        )

    def decide(
        self, t: float, state: DroneState, belief, known_claims: list[Claim]
    ) -> tuple[Command, list[Claim]]:
        phase = (t + 7.0 * self.me) % (PATROL_S + LISTEN_S)  # stagger drones
        if self._site is not None and phase >= PATROL_S:
            return Command(Mode.LISTEN, perch_site=self._site.id), []
        r = 80.0
        a = t / 20.0 + self.me
        goal = (self.home[0] + r * math.cos(a), self.home[1] + r * math.sin(a))
        return Command(Mode.PATROL, goal=goal), []

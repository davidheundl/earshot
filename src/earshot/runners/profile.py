"""Where a live world tick's wall time goes (`earshot live --profile`).

    prof = TickProfile()
    ...work...; prof.mark("sensors")        # time since the previous mark goes to "sensors"
    prof.add("webots.step", seconds)        # time measured elsewhere (inside the backend)
    prof.count("verifying")                 # a tick where something was true
    prof.end_tick()                         # once per tick; prints a line every `every` s

Timing only: nothing here changes what a run does.
"""

from __future__ import annotations

import time


class TickProfile:
    def __init__(self, every: float = 10.0, budget_ms: float = 100.0, out=print):
        self.every, self.budget_ms, self.out = every, budget_ms, out
        self._t = time.perf_counter()
        self._since = time.monotonic()
        self._phases: dict[str, float] = {}  # name -> seconds in this report window
        self._ticks: list[float] = []  # wall ms of each tick in this window
        self._tick0 = self._t
        self._counts: dict[str, int] = {}  # name -> ticks it was true in this window

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self._phases[name] = self._phases.get(name, 0.0) + now - self._t
        self._t = now

    def add(self, name: str, seconds: float) -> None:
        self._phases[name] = self._phases.get(name, 0.0) + seconds

    def count(self, name: str) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1

    def end_tick(self) -> None:
        now = time.perf_counter()
        self._ticks.append((now - self._tick0) * 1000)
        self._tick0 = now
        if time.monotonic() - self._since >= self.every:
            self.report()

    def report(self) -> None:
        n = len(self._ticks)
        if not n:
            return
        ticks = sorted(self._ticks)
        parts = " ".join(f"{k}={v * 1000 / n:.1f}" for k, v in self._phases.items())
        parts += "".join(f" {k}={v}/{n}" for k, v in self._counts.items())
        over = sum(t > 1.5 * self.budget_ms for t in ticks)
        self.out(f"[profile] {n} ticks, ms/tick mean {sum(ticks) / n:.1f} "
                 f"p95 {ticks[int(0.95 * (n - 1))]:.0f} max {ticks[-1]:.0f} "
                 f">{1.5 * self.budget_ms:.0f}ms {over} | {parts}")
        self._phases, self._ticks, self._counts = {}, [], {}
        self._since = time.monotonic()

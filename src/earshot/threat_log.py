"""Threat Log: the operator's record of every danger the swarm verified.

    log = ThreatLog(threat="VEHICLE")
    log.add(event)            an Agent's decision event (dispatch, confirmed, following, ...)
    log.look(detections)      the World's detections of one step (camera labels)
    log.lines()               new log lines since the last call (runs/*.threats.jsonl)
    log.entries()             one summary per danger: the dashboard and RunResult

Runner side only, like world_proc.VerifyCounts: it reads the Agents' decision events and
what the Verifier's camera recognised (`Detection.label`), and no Agent ever reads it back.

A Verifier sortie runs from "dispatch" to its outcome. "confirmed" opens an entry (or joins
one seen in the last MERGE_S, within MERGE_M plus how far it can have driven since, at
MOVE_SPEED: the same poacher seen again), labelled
with what the camera recognised as the threat; while the Verifier follows it ("following",
once a second) the entry records its path, until "lost" or "handed_off". "dismissed" and
"timeout" log what the camera saw instead (an elephant herd) or "nothing". Sorties that
end otherwise (outbid, obsolete, drifted) leave no entry: another drone carries them on.
"""

from __future__ import annotations

import math
from collections import Counter

MERGE_M = 300.0  # a confirmation this close to a known threat is that threat again
MERGE_S = 120.0  # ... if it was seen this recently
MOVE_SPEED = 15.0  # m/s it may have moved meanwhile (belief.CONFIRM_SPEED)
PATH_SHOWN = 60  # path points (a minute of following) an entry carries to the dashboard


class ThreatLog:
    def __init__(self, threat: str = "DRONE"):
        self.threat = threat.lower()
        self._open: dict[int, dict] = {}  # drone -> its current sortie
        self._entries: list[dict] = []
        self._following: dict[int, dict] = {}  # drone -> the entry it follows
        self._lines: list[dict] = []

    # -- input ------------------------------------------------------------------------------

    def add(self, ev: dict) -> None:
        kind, drone = ev.get("kind"), ev.get("drone")
        if ev.get("task") != "verify" or drone is None:
            return
        if kind == "dispatch":
            self._following.pop(drone, None)  # a new sortie: the last one is over
            self._open[drone] = {"tid": ev.get("tid"), "t": ev["t"], "seen": Counter(),
                                 "instead": Counter(), "empty": 0}
        elif kind == "confirmed":
            self._confirmed(drone, ev)
        elif kind == "following":
            self._follow(drone, ev)
        elif kind in ("lost", "handed_off"):
            e = self._following.pop(drone, None)
            if e is not None:
                e.update(status=kind, t=ev["t"], t_end=ev["t"])
                self._line(e, kind, ev)
            self._open.pop(drone, None)
        elif kind in ("dismissed", "timeout"):
            self._dismissed(drone, ev)
        elif kind in ("obsolete", "outbid", "drifted"):
            self._open.pop(drone, None)

    def look(self, detections: list[dict]) -> None:
        """Camera readings of drones on a sortie: hits name the threat, a "nothing" names
        what was there instead (if anything)."""
        for d in detections:
            if d.get("sensor") != "CAMERA":
                continue
            s = self._open.get(d["drone"])
            if s is None:
                continue
            label = d.get("label")
            hit = d.get("bearing") == d.get("bearing")  # NaN: "nothing here"
            if hit:
                s["seen"][label or self.threat] += 1
            elif label:
                s["instead"].update(x.strip() for x in label.split(","))
            else:
                s["empty"] += 1
            e = self._following.get(d["drone"])
            if e is not None and hit and label:
                e["label"] = label

    # -- output -----------------------------------------------------------------------------

    def lines(self) -> list[dict]:
        out, self._lines = self._lines, []
        return out

    def entries(self, path_points: int | None = None) -> list[dict]:
        if path_points is None:
            return [dict(e) for e in self._entries]
        return [{**e, "path": e["path"][-path_points:]} for e in self._entries]

    # -- internals ----------------------------------------------------------------------------

    def _confirmed(self, drone: int, ev: dict) -> None:
        s = self._open.get(drone) or {"t": ev["t"], "seen": Counter()}
        label = s["seen"].most_common(1)[0][0] if s["seen"] else self.threat
        x, y, t = float(ev.get("x", 0.0)), float(ev.get("y", 0.0)), ev["t"]
        e = next((e for e in reversed(self._entries)
                  if e["status"] in ("confirmed", "following", "lost", "handed_off")
                  and t - e["t"] <= MERGE_S
                  and math.hypot(e["x"] - x, e["y"] - y) <= MERGE_M + MOVE_SPEED * (t - e["t"])),
                 None)
        if e is None:
            e = {"id": f"T{len(self._entries) + 1}", "cls": self.threat, "label": label,
                 "status": "confirmed", "drone": drone, "t_dispatch": s["t"], "t_confirmed": t,
                 "t": t, "x": x, "y": y, "path": [[round(t, 1), round(x), round(y)]]}
            self._entries.append(e)
        else:  # seen again, maybe by another Verifier
            e.update(status="confirmed", drone=drone, t=t, x=x, y=y)
            if s["seen"]:
                e["label"] = label
        self._following[drone] = e
        self._line(e, "confirmed", ev)

    def _follow(self, drone: int, ev: dict) -> None:
        e = self._following.get(drone)
        if e is None:
            return
        x, y, t = float(ev["x"]), float(ev["y"]), ev["t"]
        e.update(status="following", t=t, x=x, y=y)
        e["path"].append([round(t, 1), round(x), round(y)])
        self._line(e, "following", ev)

    def _dismissed(self, drone: int, ev: dict) -> None:
        s = self._open.pop(drone, None)
        if s is None:
            return
        instead = ", ".join(k for k, _ in s["instead"].most_common(2)) or "nothing"
        t, x, y = ev["t"], float(ev.get("x", 0.0)), float(ev.get("y", 0.0))
        e = {"id": f"T{len(self._entries) + 1}", "cls": "nothing", "label": instead,
             "status": "dismissed", "drone": drone, "t_dispatch": s["t"], "t": t, "x": x,
             "y": y, "path": [], "why": ev["kind"]}
        self._entries.append(e)
        self._line(e, "dismissed", ev)

    def _line(self, e: dict, event: str, ev: dict) -> None:
        self._lines.append({"t": round(ev["t"], 2), "id": e["id"], "event": event,
                            "cls": e["cls"], "label": e["label"], "drone": ev["drone"],
                            "x": round(float(ev.get("x", e["x"])), 1),
                            "y": round(float(ev.get("y", e["y"])), 1)})

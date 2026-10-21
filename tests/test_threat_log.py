"""The Threat Log turns decision events and camera labels into one entry per verified danger."""

import math

from earshot.threat_log import ThreatLog


def ev(kind, t, drone=7, x=100.0, y=200.0, tid=42, **kw):
    return {"t": t, "kind": kind, "drone": drone, "task": "verify", "tid": tid, "x": x, "y": y,
            **kw}


def cam(t, drone=7, label=None, hit=True):
    return {"t": t, "drone": drone, "sensor": "CAMERA", "bearing": 45.0 if hit else math.nan,
            **({"label": label} if label else {})}


def test_a_confirmed_poacher_is_logged_with_its_label_and_followed_path():
    log = ThreatLog("VEHICLE")
    log.add(ev("claim", 10.0))
    log.add(ev("dispatch", 13.0))
    log.look([cam(40.0, hit=False), cam(41.0, label="poacher pickup"),
              cam(42.0, label="poacher pickup"), cam(42.0, drone=3, label="elephant herd")])
    log.add(ev("confirmed", 43.0))
    for k in range(5):
        log.add(ev("following", 44.0 + k, x=100.0 + 10 * k, cls="VEHICLE"))
    log.add(ev("handed_off", 50.0, x=140.0, cls="VEHICLE"))
    [e] = log.entries()
    assert (e["id"], e["cls"], e["label"], e["status"]) == ("T1", "vehicle", "poacher pickup", "handed_off")
    assert (e["t_dispatch"], e["t_confirmed"], e["t_end"], e["drone"]) == (13.0, 43.0, 50.0, 7)
    assert [p[1] for p in e["path"]] == [100, 100, 110, 120, 130, 140]
    lines = log.lines()
    assert [x["event"] for x in lines] == ["confirmed"] + ["following"] * 5 + ["handed_off"]
    assert log.lines() == []  # each line once
    assert len(log.entries(path_points=2)[0]["path"]) == 2


def test_a_dismissed_check_names_what_the_camera_saw_instead():
    log = ThreatLog("VEHICLE")
    log.add(ev("dispatch", 5.0, drone=2, tid=9))
    log.look([cam(20.0, drone=2, label="elephant herd", hit=False),
              cam(21.0, drone=2, label="elephant herd, white rhino", hit=False)])
    log.add(ev("dismissed", 30.0, drone=2, tid=9))
    log.add(ev("dispatch", 40.0, drone=4, tid=11))
    log.look([cam(50.0, drone=4, hit=False)])
    log.add(ev("timeout", 60.0, drone=4, tid=11))
    log.add(ev("dispatch", 70.0, drone=5, tid=12))
    log.add(ev("outbid", 71.0, drone=5, tid=12))  # carried on by another drone: no entry
    es = log.entries()
    assert [(e["status"], e["label"], e["cls"]) for e in es] == [
        ("dismissed", "elephant herd, white rhino", "nothing"), ("dismissed", "nothing", "nothing")]


def test_the_same_poacher_confirmed_again_nearby_is_one_entry():
    log = ThreatLog("VEHICLE")
    log.add(ev("dispatch", 1.0))
    log.add(ev("confirmed", 10.0))
    log.add(ev("lost", 30.0))
    log.add(ev("dispatch", 40.0, drone=9, tid=50))
    log.look([cam(55.0, drone=9, label="poacher pickup")])
    log.add(ev("confirmed", 60.0, drone=9, tid=50, x=250.0))  # 150 m on, 30 s after it was lost
    [e] = log.entries()
    assert (e["status"], e["drone"], e["label"], e["x"]) == ("confirmed", 9, "poacher pickup", 250.0)
    log.add(ev("dispatch", 400.0, drone=1, tid=60, x=3000.0))
    log.add(ev("confirmed", 410.0, drone=1, tid=60, x=3000.0))  # far away: a second poacher
    assert [e["id"] for e in log.entries()] == ["T1", "T2"]


def test_without_follow_a_confirmation_is_logged_and_the_next_sortie_starts_clean():
    log = ThreatLog()  # the airport: drones
    log.add(ev("dispatch", 1.0))
    log.add(ev("confirmed", 10.0))
    log.add(ev("dispatch", 100.0, x=5000.0, tid=77))
    log.look([cam(101.0, label="drone")])  # the new sortie's sighting, not T1's
    [e] = log.entries()
    assert (e["cls"], e["label"], e["status"]) == ("drone", "drone", "confirmed")

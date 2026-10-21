"""Agent radio contract: a sensor window's readings share their Observations, and every
reading is fused exactly once per (origin, seq)."""

from pathlib import Path

from earshot.agent import Agent
from earshot.city import City
from earshot.messages import (
    Mode,
    MsgType,
    Observation,
    Sensor,
    SensorReading,
    decode,
    peek_key,
    with_hops,
)
from earshot.state import DroneSpec, DroneState, Sensed, SwarmConfig

CITY = City.load(Path(__file__).resolve().parents[1] / "data/cities/synthetic.json")
KEY = b"k" * 16
SWARM = SwarmConfig.from_specs([DroneSpec(0, -300.0, -300.0), DroneSpec(1, 300.0, 300.0)])
DRONE_P = (0.9, 0.0333, 0.0333, 0.0334)


def _sensed(me, t, readings):
    at = SWARM.homes[me]
    return Sensed(DroneState(me, t, *at, 40.0, 0.0, Mode.LISTEN, None, 1.0, 0.0, False),
                  readings)


def _observations(outbox):
    return [f for f in outbox if peek_key(f)[0] == MsgType.OBSERVATION]


def test_a_windows_readings_share_their_observations():
    sounds = [SensorReading(Sensor.SOUND, b, 5.0, DRONE_P, 50.0) for b in (10.0, 20.0, 30.0)]
    uhf = SensorReading(Sensor.RF_UHF, float("nan"), 0.0, DRONE_P, -90.0)
    me = Agent(1, CITY, KEY, SWARM)
    out = me.step(1.0, _sensed(1, 1.0, [*sounds, uhf]), [])
    frames = [decode(f, KEY) for f in _observations(out.outbox)]
    assert [len(o.readings) for o in frames] == [Observation.MAX_READINGS, 2]
    assert [r.sensor for o in frames for r in o.readings] == [Sensor.SOUND] * 3 + [Sensor.RF_UHF]
    assert me.belief.n_evidence == 4  # its own map has every reading, once
    # the next window, 0.5 s on, is too soon for those sensors: nothing goes out
    assert _observations(me.step(1.5, _sensed(1, 1.5, sounds), []).outbox) == []


def test_an_observation_is_fused_once_with_all_its_readings():
    sender, rx = Agent(1, CITY, KEY, SWARM), Agent(0, CITY, KEY, SWARM)
    readings = [SensorReading(Sensor.SOUND, 225.0, 5.0, DRONE_P, 50.0),
                SensorReading(Sensor.RF_2G4, float("nan"), 0.0, DRONE_P, -85.0)]
    (frame,) = _observations(sender.step(1.0, _sensed(1, 1.0, readings), []).outbox)
    relayed = with_hops(frame, 0)  # the same (origin, seq), arriving again by another path
    rx.step(1.1, _sensed(0, 1.1, []), [(frame, 10.0), (relayed, 25.0)])
    rx.step(1.2, _sensed(0, 1.2, []), [(frame, 10.0)])
    assert rx.belief.n_evidence == 2
    assert set(rx.belief._newest) == {(1, "sound"), (1, "rf")}

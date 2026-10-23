"""Which trained models a run uses: the Scenario's `audio:` and `camera:`, the command line over
them, and a stand-in (never a crash) when a model cannot load."""

from pathlib import Path

from earshot.backend.twin2d import Kinematics
from earshot.messages import Mode
from earshot.scenario import Scenario
from earshot.sensors import build_perception

ROOT = Path(__file__).resolve().parents[1]


def load(name, **over):
    sc = Scenario.load(ROOT / "scenarios" / name)
    for k, v in over.items():
        setattr(sc, k, v)
    return sc


def test_plain_scenario_uses_the_stand_ins():
    assert build_perception(load("demo.yaml")) == (None, None)


def test_muc_full_asks_for_both_models():
    sc = load("muc_full.yaml")
    assert sc.audio["classifier"] == "earshot.audio.classifier:efficientat"
    assert sc.camera["detector"] == "yolo"
    assert Kinematics.for_scenario(sc).standoff == 25.0
    assert Kinematics.for_scenario(load("muc.yaml")).standoff == Kinematics.standoff


def test_a_kinematics_block_flies_faster_and_keeps_what_it_leaves_out():
    assert Kinematics.for_scenario(load("muc.yaml")) == Kinematics()
    kin = Kinematics.for_scenario(load("demo.yaml", kinematics={
        "speed": {"verify": 22, "listen": 15}, "climb_rate": 8}))
    assert kin.v(Mode.VERIFY) == 22.0 and kin.v(Mode.LISTEN) == 15.0
    assert kin.v(Mode.PATROL) == Kinematics().v(Mode.PATROL)
    assert kin.climb_rate == 8.0 and kin.max_speed == Kinematics.max_speed
    assert Kinematics(speed={Mode.VERIFY: 40.0}).v(Mode.VERIFY) == Kinematics.max_speed


def test_a_classifier_that_cannot_load_falls_back_with_a_message():
    said = []
    audio, cam = build_perception(load("demo.yaml", audio={"classifier": "no_such_module:x"}),
                                  log=said.append)
    assert audio is None and cam is None
    assert "CONFUSION stand-in" in said[0]


def test_command_line_turns_the_scenario_models_off():
    sc = load("demo.yaml", audio={"classifier": "no_such_module:x"},
              camera={"detector": "yolo"})
    said = []
    assert build_perception(sc, classifier="none", camera="truth", renders=True,
                            log=said.append) == (None, None)
    assert said == []


def test_camera_detector_only_with_rendered_images():
    sc = load("demo.yaml", camera={"detector": "yolo"})
    assert build_perception(sc, renders=False)[1] is None


def test_a_detector_without_weights_falls_back_with_a_message():
    sc = load("demo.yaml", camera={"detector": "yolo", "weights": "/nonexistent/drone.pt"})
    said = []
    _, cam = build_perception(sc, renders=True, log=said.append)
    assert cam is None and "ground-truth stand-in" in said[0]


def test_audio_flag_renders_waveforms_without_a_classifier():
    audio, _ = build_perception(load("demo.yaml"), audio=True)
    assert audio is not None and audio.classifier is None

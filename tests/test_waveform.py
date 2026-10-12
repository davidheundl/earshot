"""Waveforms must agree with the level model, so their SNRs are valid training labels, and
a trained classifier plugged into the hook must decide the class probabilities."""

import json
import math
import sys
import types
import wave

import numpy as np

from earshot.messages import TargetClass
from earshot.runners.lockstep import run
from earshot.scenario import Scenario
from earshot.sensors import Audio
from earshot.sensors import waveform as wf
from earshot.sensors.export import AudioExport

BANK = wf.ClipBank(None)  # synthetic clips only, whatever data/sounds holds


def _db(x):
    return wf.FULL_SCALE_DB + 10 * math.log10(float(np.mean(np.square(x, dtype=np.float64))))


def _part(src="d1", cls=TargetClass.DRONE, level=60.0, d=200.0, snr=10.0):
    return wf.Part(src, cls, level, d, 0, 90.0, snr)


def test_each_part_has_the_level_the_model_computes():
    win = wf.Mixer(BANK).window(0, 12.3, [_part(level=62.0), _part("t1", TargetClass.TRAIN, 70.0)],
                                ego_db=-math.inf, background_db=40.0)
    assert abs(_db(win.signals[0]) - 62.0) < 0.1
    assert abs(_db(win.signals[1]) - 70.0) < 0.1
    assert abs(_db(win.noise) - 40.0) < 0.1
    assert abs(_db(win.mix) - 10 * math.log10(10**6.2 + 10**7 + 10**4)) < 1.0


def test_beam_suppresses_the_other_sources_by_the_masking_offset():
    mixer = wf.Mixer(BANK)
    win = mixer.window(0, 5.0, [_part(level=60.0), _part("v1", TargetClass.VEHICLE, 60.0)],
                       ego_db=-math.inf, background_db=-math.inf)
    other = mixer.beam(win, 0) - win.signals[0]
    assert abs(_db(other) - (60.0 - wf.MASKING_OFFSET_DB)) < 0.1


def test_consecutive_windows_continue_the_clip():
    clip = BANK.clip("drone", "d1")
    two = np.concatenate([wf._slice(clip, 1.0, wf.SR), wf._slice(clip, 2.0, wf.SR)])
    assert np.array_equal(two, wf._slice(clip, 2.0, 2 * wf.SR))


def test_wav_round_trip(tmp_path):
    x = (np.sin(np.arange(wf.SR) / 10.0) * 0.1).astype(np.float32)
    wf.write_wav(tmp_path / "x.wav", x)
    with wave.open(str(tmp_path / "x.wav")) as w:
        y = np.frombuffer(w.readframes(w.getnframes()), "<i4") / 2**31 * 10.0
    assert np.allclose(x, y, atol=1e-6)
    assert np.allclose(wf.read_wav(tmp_path / "x.wav"), wf._unit(x), atol=1e-4)


def test_a_plugged_in_classifier_decides_the_class(tmp_path, monkeypatch):
    mod = types.ModuleType("fake_clf")
    mod.calls = 0

    def classify(audio, sr):
        mod.calls += 1
        assert sr == wf.SR and audio.shape == (wf.SR,)
        return (0.0, 0.0, 1.0, 0.0)  # everything is a train

    mod.always_train = classify
    monkeypatch.setitem(sys.modules, "fake_clf", mod)
    clf = wf.load_classifier("fake_clf:always_train")

    sc = Scenario.load("scenarios/demo.yaml", seed=0)
    sc.duration = 40.0
    mixer = wf.Mixer(BANK)
    out = AudioExport(tmp_path, mixer, beams=True)
    det_log = tmp_path / "run.jsonl"
    run(sc, log=det_log, audio=Audio(mixer, clf), audio_out=out)
    out.close()

    sound = [json.loads(line) for line in (tmp_path / "run.detections.jsonl").read_text().splitlines()]
    sound = [d for d in sound if d["sensor"] == "SOUND"]
    assert sound and mod.calls == len(sound)
    assert all(d["probs"] == [0.0, 0.0, 1.0, 0.0] for d in sound)
    labels = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert 40 <= len(labels) / len(sc.drones) <= 41  # one window per drone and second
    assert any(lab["beams"] for lab in labels)


def test_audio_is_deterministic(tmp_path):
    def once(d):
        sc = Scenario.load("scenarios/demo.yaml", seed=3)
        sc.duration = 15.0
        mixer = wf.Mixer(BANK)
        out = AudioExport(d, mixer)
        run(sc, audio=Audio(mixer), audio_out=out)
        out.close()
        return (d / "drone0.wav").read_bytes()

    assert once(tmp_path / "a") == once(tmp_path / "b")

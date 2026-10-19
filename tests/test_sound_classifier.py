"""The EfficientAT sound classifiers (earshot.audio.classifier): fine classes fold onto the
radio's 4 Target Classes, and the in-sim hook stays deterministic. Needs the `ml` group; the
model tests also need the weights (tools/fetch_efficientat.py) and skip without them."""

from pathlib import Path

import numpy as np
import pytest

from earshot.messages import TargetClass

torch = pytest.importorskip("torch")

from earshot.audio import classifier as clf
from earshot.audio import efficientat

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / clf.BACKBONE
needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="no EfficientAT weights")


def test_planes_and_fans_fold_to_nothing_and_cars_to_vehicle():
    def one_hot(name):
        p = np.zeros(len(clf.FINE))
        p[clf.FINE.index(name)] = 1.0
        return clf.fold(p)

    assert np.argmax(one_hot("drone")) == TargetClass.DRONE
    for harmless in ("plane", "helicopter", "machine", "insect", "ambient", "other"):
        assert np.argmax(one_hot(harmless)) == TargetClass.NOTHING
    assert np.argmax(one_hot("vehicle")) == TargetClass.VEHICLE
    assert np.argmax(one_hot("train")) == TargetClass.TRAIN
    assert sum(clf.fold(np.full(len(clf.FINE), 1 / len(clf.FINE)))) == pytest.approx(1.0)


def test_every_zeroshot_label_is_an_audioset_label():
    names = set(clf.audioset_labels())
    assert len(clf.audioset_labels()) == clf.N_LABELS
    assert {n for labels in clf.ZEROSHOT.values() for n in labels} <= names
    assert set(clf.ZEROSHOT) == set(clf.FINE)


def test_mel_banks_are_triangles_over_the_band():
    b = efficientat.kaldi_mel_banks()
    assert b.shape == (128, 513)
    assert (b >= 0).all() and b.max() <= 1.0
    assert b[:, -1].sum() == 0  # the padded Nyquist bin
    assert (b.sum(axis=1) > 0).all()


@needs_weights
def test_zeroshot_is_a_deterministic_classifier(monkeypatch):
    monkeypatch.chdir(ROOT)
    from earshot.sensors import waveform as wf

    c = wf.load_classifier("earshot.audio.classifier:zeroshot")
    x = wf.synth("train", 0)[:wf.SR] * 0.01  # a beam's scale: RMS 1 is 110 dB
    p = c.classify(x, wf.SR)
    assert len(p) == 4 and sum(p) == pytest.approx(1.0) and min(p) >= 0
    assert c.classify(x, wf.SR) == p
    assert c.classify(x * 100, wf.SR) == pytest.approx(p, abs=1e-5)  # level-independent


@needs_weights
def test_same_seed_same_run_with_a_classifier(monkeypatch):
    monkeypatch.chdir(ROOT)
    from earshot.eval import sound_audio
    from earshot.runners.lockstep import run
    from earshot.scenario import Scenario

    audio = sound_audio("earshot.audio.classifier:zeroshot")

    def short():
        sc = Scenario.load(ROOT / "scenarios/demo.yaml", seed=7)
        sc.duration = 4.0
        return sc

    a, b = run(short(), root=ROOT, audio=audio), run(short(), root=ROOT, audio=audio)
    assert a.radio == b.radio
    assert a.agents == b.agents

"""Audio export: a run's sound windows as training data for a sound classifier.

    mixer = Mixer(ClipBank())
    out = AudioExport(Path("runs/audio-demo-seed0"), mixer, beams=True)
    run(scenario, audio=Audio(mixer), audio_out=out); out.close()

Writes, per run directory:
    drone<k>.wav    everything drone k's microphone got, its 1 s windows back to back
    labels.jsonl    one line per window: file, offset_s in it, and every Source with its true
                    class, received level, SNR, distance, buildings, true bearing
    beams/          with beams=True, one file per Source and window with SNR >= BEAM_MIN_SNR:
                    what the in-sim classifier hook is given (sensors.waveform.Mixer.beam)
    meta.json       sample rate and the level calibration (see waveform.write_wav)
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

from earshot.sensors import waveform as wf

BEAM_MIN_SNR = -20.0  # dB; quieter beams are noise only, labelled by an inaudible source


class AudioExport:
    def __init__(self, root: Path, mixer: wf.Mixer, beams: bool = False):
        self.root = root
        self.mixer = mixer
        self.beams = beams
        root.mkdir(parents=True, exist_ok=True)
        if beams:
            (root / "beams").mkdir(exist_ok=True)
        self._files: dict[int, wave.Wave_write] = {}
        self._frames: dict[int, int] = {}
        self._labels = open(root / "labels.jsonl", "w")  # noqa: SIM115 - closed in close()
        (root / "meta.json").write_text(json.dumps({
            "sr": wf.SR, "sample_format": "PCM 32-bit mono",
            "full_scale_db": wf.FULL_SCALE_DB + 20.0,
            "note": "dB(A) = full_scale_db + 20 log10(|sample| / 2^31); RMS 1.0 in memory = "
                    f"{wf.FULL_SCALE_DB} dB",
            "classes": list(wf.CLASS_DIRS.values()),
            "recorded_clips": mixer.bank.recorded,
        }, indent=1))

    def add(self, windows: list[wf.AudioWindow]) -> None:
        for win in windows:
            f = self._file(win.drone)
            offset = self._frames[win.drone] / wf.SR
            f.writeframes(wf.pcm32(win.mix))
            self._frames[win.drone] += len(win.mix)
            line = {"file": f"drone{win.drone}.wav", "offset_s": round(offset, 3), **win.label()}
            if self.beams:
                line["beams"] = {}
                for i, p in enumerate(win.parts):
                    if p.snr_db < BEAM_MIN_SNR:
                        continue
                    name = f"beams/d{win.drone}_t{win.t:08.2f}_{p.source}.wav"
                    wf.write_wav(self.root / name, self.mixer.beam(win, i))
                    line["beams"][p.source] = name
            self._labels.write(json.dumps(line) + "\n")

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._labels.close()

    def _file(self, drone: int) -> wave.Wave_write:
        if drone not in self._files:
            w = wave.open(str(self.root / f"drone{drone}.wav"), "wb")  # noqa: SIM115 - closed in close()
            w.setnchannels(1)
            w.setsampwidth(4)
            w.setframerate(wf.SR)
            self._files[drone] = w
            self._frames[drone] = 0
        return self._files[drone]


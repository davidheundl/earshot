# Sound classifier spike (2026-10-18)

Question: can a learned model tell a drone from planes, trains, cars and other noise well
enough to replace the `CONFUSION` stand-in, and is it worth rebuilding and retraining on the
day? Built on `prototype`, declared in DECLARATION.md.

## What was built

- `src/earshot/audio/classifier.py`, hooked in through the existing `load_classifier` as
  `earshot.audio.classifier:zeroshot` and `earshot.audio.classifier:efficientat`.
- **Backbone:** EfficientAT mn10_as, an AudioSet-pretrained MobileNet (MIT). Its model code is
  vendored in `audio/efficientat/`, with a torchaudio-free mel frontend. The backbone is frozen.
- **Head:** a 128-unit MLP over the backbone's 960-d embedding plus its 527 AudioSet label
  probabilities.
- **Classes:** the head predicts fine classes, `drone plane helicopter vehicle train machine
  insect ambient other`. These fold onto the 4 Target Classes the radio carries: a plane,
  helicopter, fan or insect counts as NOTHING. That needs no frame or belief change.
- **Zero-shot baseline:** no training. Each class scores its strongest AudioSet label.
- **Data:** `tools/fetch_sound_data.py`.
  - Train: ESC-50 folds 1-4 and DroneAudioSet *drone2*.
  - Test: ESC-50 fold 5 and *drone1*, a drone the head never heard.
  - The test recordings also stock `data/sounds/`, so the simulator's beams are real sound
    the head never trained on.
- **Training:** `tools/train_sound_head.py`. Each training window is used clean plus twice
  as a beam-like mix: −10 to +20 dB SNR over background and ego-noise, a distance lowpass,
  and in a third of cases a second source 6 dB down.
- **Evaluation:** `tools/eval_sound_classifier.py`, plus `earshot eval --classifier` side by
  side. The live arrays take `--classifier` too.

### Pitfall found on the way

The AudioSet models were trained on 10 s clips. On a 1 s window every label saturates at
p = 1, because edge effects dominate the pooled features. On the upstream example the model
gives Train 0.81 at 10 s and nonsense at 1 s. The classifier therefore tiles each 1 s beam to
5 s, which brings the logits back in range.

## Results: held-out real recordings

Per row: ESC-50 fold 5 (8 clips of 5 s per ESC-50 category; vehicle merges 3 categories, machine 4) and drone1 at 25/50 cm, as clean windows
and as beam-like mixes at fixed SNR. "Drone recall" is the share of drone windows called
drone. "False drone" is the share of everything else called drone. Full numbers are in
`sound-classifier-results.json`.

| SNR | head acc | head drone recall | head false drone | worst false-drone class |
|---|---|---|---|---|
| clean | 0.966 | 0.970 | 0.000 | — |
| +20 dB | 0.946 | 0.939 | 0.003 | machine 0.01 |
| +10 dB | 0.940 | 0.942 | 0.004 | helicopter 0.03 |
| +5 dB | 0.913 | 0.924 | 0.010 | helicopter, insect 0.03 |
| 0 dB | 0.888 | 0.916 | 0.020 | helicopter, ambient 0.05 |
| −5 dB | 0.825 | 0.828 | 0.037 | vehicle 0.06 |
| −10 dB | 0.682 | 0.599 | 0.065 | plane 0.12, vehicle 0.11 |

- **Zero-shot:** accuracy 0.38-0.39 at every SNR, and it **never** calls a drone. AudioSet
  has no UAV label, and Propeller or Light engine never win. The trained head is what makes
  this work.
- **Planes:** on clean audio, half of the plane windows are labelled "ambient", but none are
  called a drone. For the swarm, only the drone-or-not decision matters.

## Results: simulator beams (`earshot fast --audio --audio-beams`, demo seed 0, 300 s)

In this seed the intruder never comes closer than about −14 dB SNR at any drone; the median
is −15 dB. At those SNRs the head gives the drone p(drone) ≈ 0.3, against 0.0-0.2 for
everything else, and calls it a drone in 17-21 % of windows. The Nuisances in
`data/sounds/nothing/` (planes, helicopters, machines, insects) are never called a drone.
Trucks are recognised in 81 % of windows from −5 dB up and in 94 % above 0 dB.

The simulator detects a sound at −8 dB SNR half the time (`acoustics.snr50_db`), and the
beam the classifier gets has exactly that SNR. At the SNRs where the sim detects a drone, a
real recording is hard to classify. According to `acoustics.py`, the −8 dB threshold
models "tonal / long-integration detectors" that work below the broadband noise floor. The
classifier gets no such integration: one 1 s beam at that SNR. That gap is either a sim
assumption to revisit or the honest limit of 1 s sound classification.

## Results: downstream (`earshot eval --scenario random --runs 30`)

`EARSHOT_SOUND_DEVICE=mps uv run earshot eval --scenario random --runs 30 --classifier none
earshot.audio.classifier:efficientat`. Real clips are in `data/sounds/`, and the run took 33 min.
Medians, with [min-max] over seeds:

| | confirmed | t_dispatch s | t_confirm s | loc_err m | false_verify | verify_flights | missed |
|---|---|---|---|---|---|---|---|
| none (CONFUSION) | 20/30 | 14.2 [3-228] | 31.8 [17-127] | 102.5 | 18.5 [0-48] | 26.0 | 1.0 [0-2] |
| efficientat (head) | 22/30 | 10.0 [5-208] | 28.1 [2-105] | 97.3 | 18.5 [0-54] | 24.0 | 0.0 [0-2] |

Paired per seed (head minus none):

| metric | median difference | head better / worse |
|---|---|---|
| t_dispatch | −2.0 s | 13 / 11 seeds |
| t_confirm | −2.2 s | 10 / 9 seeds |
| false_verify | +1 | 11 / 15 seeds |
| missed | — | 3 / 1 seeds |

The head confirmed 3 drones that CONFUSION did not; CONFUSION confirmed 1 the head did not.
Downstream this is a wash with a slight edge for the head. A real classifier does about as
well as the hand-set placeholder, and is no worse. That's the useful finding: the placeholder
(drone recognised 85 %, fan→drone 35 %) was not wildly optimistic overall. The real model is
much better on fans and worse on faint drones.

`live` with the head on CPU kept real time on the demo: 120 sim s took 133 s of wall time,
startup included, while another `earshot tune` was running.

## Caveats

- **The "unseen drone" isn't far from the training data.** drone1 was recorded in the same
  lab, with the same microphones and at the same 25/50 cm as drone2. The head may partly
  recognise the DroneAudioSet recording setup. Nothing here tests a drone recorded outdoors
  at range, and DroneAudioSet has only these two drones.
- **Small test sets for planes and helicopters:** 8 ESC-50 clips each, 40 windows. A
  helicopter→drone rate of 0.05 means 2 windows.
- **Ego-noise and background in training are the simulator's synthetic loops**, not real
  motor noise.
- **Different Nuisance sounds:** the simulator's "fan" now sometimes sounds like a plane or a
  vacuum cleaner, because `data/sounds/nothing/` mixes them. That is intended.
- **Tuning bias:** `PolicyParams` were tuned against `CONFUSION` (fan→drone 35 %). A
  classifier that rarely calls a fan a drone moves the evidence the thresholds were tuned
  for, so some of any eval difference is the tuning target moving.
- **Never run on a real microphone.** The arrays take `--classifier`, but no drone, plane or
  car sound has been played at a ReSpeaker yet, because there was no hardware here.
- **The committed head (`data/models/sound_head.pt`, 0.8 MB) was trained partly on ESC-50,
  which is CC BY-NC.** That's fine for a hackathon demo; it is declared in DECLARATION.md.
- **Unpaired comparison:** `none` vs `efficientat` runs aren't paired, because CONFUSION draws
  random numbers and the classifier doesn't. Read medians and ranges over seeds.
- **Cost:** about 170 ms per beam on Apple-silicon CPU (torch has no fast depthwise
  convolution there) and about 70 ms on MPS, or 18 ms batched. `EARSHOT_SOUND_DEVICE=mps`
  selects MPS. Only beams of detected Sources are classified, so `live` keeps real time on
  CPU (see above). A 30-seed eval takes about 33 min instead of a few.

## Go / no-go for the day

**Go, with the effort put where the gaps are, not into retraining.**

- **Use the head.** It replaces a made-up table with a measured model. Drone vs. everything
  else is clean above 0 dB: drone recall ≥ 0.92 and false drone ≤ 0.02. At −5 dB and above, no
  harmless class is called a drone in more than 5 % of windows. It costs nothing downstream, and
  it's a much better story for the judges.
- **Retraining is cheap.** Features are cached, so re-fitting the head takes seconds. Fetching
  the data and extracting features is about 1 h, on MPS. The vendored backbone and the hook
  count as infrastructure.
- **What would actually move the numbers on the day:**
  1. Real drone recordings at range, outdoors. Both drones in the data were recorded at
     25/50 cm in one lab. Record the team's own drone at 20-100 m on the day, add it to
     training, and re-fit.
  2. Faint drones. The sim detects them at −15 dB SNR, where 1 s of audio says little. Give
     the classifier the integration the detection threshold assumes: 2-3 s of beam history
     per Source instead of 1 s, or a real array's beamforming gain (about 6 dB for the
     ReSpeaker's 4 mics).
  3. More plane and helicopter negatives. ESC-50 has 40 clips each, so add FSD50K
     (CC BY) aircraft clips if planes become part of the demo.

## Reproduce

    uv sync --group ml
    uv run python tools/fetch_efficientat.py           # backbone, 20 MB: all the head needs
    # the trained head is committed; to retrain or evaluate:
    uv run python tools/fetch_sound_data.py            # ~1 GB download, ~7 min
    uv run python tools/train_sound_head.py            # ~25 min features on MPS, then seconds
    uv run earshot fast --audio runs/audio --audio-beams --duration 300
    uv run python tools/eval_sound_classifier.py --beams runs/audio/demo-seed0
    EARSHOT_SOUND_DEVICE=mps uv run earshot eval --scenario random --runs 30 \
        --classifier none earshot.audio.classifier:efficientat

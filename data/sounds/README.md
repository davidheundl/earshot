# Sound recordings

Drop WAV recordings here to replace the synthetic stand-in sounds (`src/earshot/sensors/waveform.py`):

    data/sounds/drone/*.wav        small drones, rotor noise
    data/sounds/vehicle/*.wav      trucks, cars
    data/sounds/train/*.wav        trains
    data/sounds/nothing/*.wav      Nuisances: fans, AC units (drone-like, but no threat)
    data/sounds/ego/*.wav          our own drone's motors, heard at its microphone
    data/sounds/background/*.wav   ambient city noise

Any PCM WAV works (8/16/24/32-bit, mono or stereo, any rate); it is resampled to 16 kHz mono and
normalised, because the level comes from the sound model. Clips must be at least 1 s long; each
Source keeps one clip for the whole run. A class without recordings is synthesised. Check each
dataset's licence and list it in DECLARATION.md.

`tools/fetch_sound_data.py` stocks `drone/`, `vehicle/`, `train/` and `nothing/` with real recordings
(DroneAudioSet's drone1, ESC-50 fold 5: engines, trains, planes, helicopters, machines, insects): the
ones the sound classifier never trained on, so simulated beams test it fairly. They are gitignored
(dataset licences); `ego/` and `background/` stay synthetic.

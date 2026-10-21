from earshot.agent.store import MessageStore
from earshot.messages import Claim, Header, Request, encode

KEY = b"k" * 16


def test_own_messages_appear_in_my_digest():
    store = MessageStore(2, KEY)
    for seq in (1, 2, 7):
        store.add_own(encode(Claim(Header(2, seq, 3), 1.0, 5, 600.0, 10.0, 0.0, 0.0), KEY))
    assert store.digest(t=2.0) == [(2, 7)]


def _claim(origin, seq):
    return encode(Claim(Header(origin, seq, 1), 1.0, 5, 600.0, 10.0, 0.0, 0.0), KEY)


def test_a_wrapped_seq_is_a_new_frame_once_the_old_one_is_forgotten():
    store = MessageStore(0, KEY)
    assert store.receive(_claim(2, 7), 0.0, t=0.0) is not None
    assert store.receive(_claim(2, 7), 0.0, t=1.0) is None  # duplicate
    # 65536 frames later the same seq comes round again: by then the first is forgotten
    assert store.receive(_claim(2, 7), 0.0, t=700.0) is not None


def test_the_digest_follows_seqs_across_the_wrap():
    store = MessageStore(0, KEY)
    store.receive(_claim(2, 65530), 0.0, t=0.0)
    store.receive(_claim(2, 3), 0.0, t=1.0)  # after the wrap: newer, though smaller
    assert store.digest(t=2.0) == [(2, 3)]
    store.receive(_claim(2, 65531), 0.0, t=2.0)  # a late old one does not move it back
    assert store.digest(t=2.0) == [(2, 3)]


def test_one_hop_frames_are_not_healed():
    from earshot.messages import (
        Mode,
        Observation,
        Sensor,
        SensorReading,
        TargetClass,
        Track,
        TrackStatus,
    )

    def track(seq, status):
        return encode(Track(Header(2, seq, 1), 1.0, 9, 0.0, 0.0, 50.0, 20.0, 0.0,
                            TargetClass.VEHICLE, 0.9, status, 5.0), KEY)

    obs = encode(Observation(Header(2, 5, 1), 1.0, 0.0, 0.0, Mode.LISTEN,
                             (SensorReading(Sensor.SOUND, 10.0, 5.0, (0.9, 0.0, 0.0, 0.1), 40.0),)),
                 KEY)
    store = MessageStore(0, KEY)
    store.receive(track(3, TrackStatus.SUSPECTED), 0.0, t=0.0)
    store.receive(obs, 0.0, t=0.0)
    store.receive(track(6, TrackStatus.EXPLAINED), 0.0, t=0.0)
    assert store.digest(t=2.0) == [(2, 3)]  # the newest frame worth healing, not the newest heard
    store.on_request(Request(Header(1, 0, 1), 2, 1, 6), t=1.0)
    resent = store.due(5.0)
    assert len(resent) == 1 and resent[0][1] == 2  # the Suspicion only


def _track(origin, seq, t, track_id=9):
    from earshot.messages import TargetClass, Track, TrackStatus

    return encode(Track(Header(origin, seq, 1), t, track_id, 0.0, 0.0, 50.0, 20.0, 0.0,
                        TargetClass.VEHICLE, 0.9, TrackStatus.SUSPECTED, 5.0), KEY)


def test_a_track_is_healed_only_while_a_receiver_still_counts_it():
    store = MessageStore(0, KEY, track_life=15.0, release_life=10.0)
    store.receive(_track(2, 3, t=0.0), 0.0, t=0.0)
    assert store.digest(t=10.0) == [(2, 3)]
    store.on_request(Request(Header(1, 0, 1), 2, 1, 3), t=10.0)
    assert len(store.due(13.0)) == 1
    # 15 s after it was sent nobody counts it any more: out of the Digest, not resent
    assert store.digest(t=16.0) == []
    store.on_request(Request(Header(4, 0, 1), 2, 1, 3), t=30.0)
    assert store.due(35.0) == []


def test_a_resend_that_goes_stale_while_it_waits_is_dropped():
    store = MessageStore(0, KEY, track_life=15.0)
    store.receive(_track(2, 3, t=0.0), 0.0, t=0.0)
    store.on_request(Request(Header(1, 0, 1), 2, 1, 3), t=14.9)
    assert store.due(16.0) == []
    assert store.stats.resent == 0


def test_only_the_newest_frame_of_an_id_is_healed():
    store = MessageStore(0, KEY, track_life=15.0)
    store.receive(_track(2, 3, t=0.0), 0.0, t=0.0)
    store.receive(_track(2, 5, t=4.0, track_id=7), 0.0, t=4.0)  # another Track
    store.receive(_track(2, 8, t=5.0), 0.0, t=5.0)  # replaces seq 3: same track id
    store.on_request(Request(Header(1, 0, 1), 2, 1, 8), t=6.0)
    assert sorted(f[2] | f[3] << 8 for f in store.due(10.0)) == [5, 8]


def test_a_claim_is_healed_while_its_lease_runs_and_a_release_for_one_lease():
    store = MessageStore(0, KEY, track_life=15.0, release_life=10.0)
    store.receive(encode(Claim(Header(2, 1, 1), 0.0, 5, 600.0, 4.0, 0.0, 0.0), KEY), 0.0, t=0.0)
    assert store.digest(t=3.0) == [(2, 1)]
    assert store.digest(t=5.0) == []  # its 4 s lease ran out
    store.receive(encode(Claim(Header(2, 2, 1), 6.0, 5, 0.0, 0.0, 0.0, 0.0), KEY), 0.0, t=6.0)
    assert store.digest(t=15.0) == [(2, 2)]  # a release (lease 0) counts for release_life
    assert store.digest(t=16.5) == []


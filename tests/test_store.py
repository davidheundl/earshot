from earshot.agent.store import MessageStore
from earshot.messages import Claim, Header, encode

KEY = b"k" * 16


def test_own_messages_appear_in_my_digest():
    store = MessageStore(2, KEY)
    for seq in (1, 2, 7):
        store.add_own(encode(Claim(Header(2, seq, 3), 1.0, 5, 600.0, 10.0, 0.0, 0.0), KEY))
    assert store.digest() == [(2, 7)]


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
    assert store.digest() == [(2, 3)]
    store.receive(_claim(2, 65531), 0.0, t=2.0)  # a late old one does not move it back
    assert store.digest() == [(2, 3)]

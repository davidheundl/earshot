from earshot.agent.store import MessageStore
from earshot.messages import Claim, Header, encode

KEY = b"k" * 16


def test_own_messages_appear_in_my_digest():
    store = MessageStore(2, KEY)
    for seq in (1, 2, 7):
        store.add_own(encode(Claim(Header(2, seq, 3), 1.0, 5, 600.0, 10.0, 0.0, 0.0), KEY))
    assert store.digest() == [(2, 7)]

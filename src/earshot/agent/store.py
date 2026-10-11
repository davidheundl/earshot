"""Message Store: every frame a drone has accepted, keyed (type, origin, seq), at most once.

Doubles as the duplicate filter, the forwarding queue and the buffer used to heal after a
partition (Digest / Request). Pure: no clock, no sockets.

PRE-BUILT: duplicate filter + plain hop-limited flooding.
ON THE DAY: relevance radius for Observations, contention-based forwarding (far receivers relay
first, cancel on overhearing), Digest/Request healing. Hooks: `should_forward`, `digest`,
`missing`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from earshot.messages import BadFrame, Message, MsgType, decode, peek_key, with_hops

STORED_TYPES = {MsgType.OBSERVATION, MsgType.TRACK, MsgType.CLAIM}


@dataclass
class StoreStats:
    accepted: int = 0
    duplicates: int = 0
    bad: int = 0
    forwarded: int = 0
    by_origin: dict = field(default_factory=dict)  # origin -> highest seq seen


class MessageStore:
    def __init__(self, me: int, key: bytes):
        self.me = me
        self.key = key
        self._frames: dict[tuple[int, int, int], bytes] = {}
        self.stats = StoreStats()

    def __len__(self) -> int:
        return len(self._frames)

    def add_own(self, frame: bytes) -> None:
        self._frames[peek_key(frame)] = frame

    def receive(self, frame: bytes) -> tuple[Message | None, bytes | None]:
        """Accept one frame from the radio.

        Returns (message, frame_to_forward). message is None for duplicates and bad frames;
        Digest/Request are returned but never stored or forwarded.
        """
        try:
            msg = decode(frame, self.key)
        except BadFrame:
            self.stats.bad += 1
            return None, None
        mtype, origin, seq = peek_key(frame)
        if origin == self.me:
            return None, None
        if MsgType(mtype) not in STORED_TYPES:
            return msg, None
        k = (mtype, origin, seq)
        if k in self._frames:
            self.stats.duplicates += 1
            return None, None
        self._frames[k] = frame
        self.stats.accepted += 1
        self.stats.by_origin[origin] = max(self.stats.by_origin.get(origin, 0), seq)
        fwd = None
        if self.should_forward(msg):
            fwd = with_hops(frame, msg.header.hops_left - 1)
            self.stats.forwarded += 1
        return msg, fwd

    def should_forward(self, msg: Message) -> bool:
        # PRE-BUILT placeholder: everyone forwards while hops remain (the broadcast-storm
        # baseline from concept 8.3). Replace with contention-based relaying on the day.
        return msg.header.hops_left > 1

    def digest(self) -> list[tuple[int, int]]:
        return sorted(self.stats.by_origin.items())

    def missing(self, peer_digest: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
        """(origin, from_seq, to_seq) ranges a peer has that we lack. ON THE DAY."""
        return []

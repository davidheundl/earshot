"""Message Store: every frame a drone has accepted, keyed (type, origin, seq), at most once.

Doubles as the duplicate filter, the forwarding queue and the buffer used to heal after a
Partition. Pure: time comes in as an argument.

Interface:
    store = MessageStore(me, key, scheme="ours", relay_radius=inf)
    msg = store.receive(frame, margin_db, t, here)   accepted message, or None (duplicate / bad / own)
    frames = store.due(t)                       forwards and healing resends whose time has come
    reqs = store.on_digest(digest, t)           Requests for ranges a neighbour has and we lack
    store.on_request(request, t)                schedule resends of the Tracks/Claims asked for
    store.digest()                              [(origin, highest healable seq held)]

Healing covers only what every drone should hold: Claims and Tracks that are not Explained.
Observations go one hop and an Explained Track too (agent/core.py): a drone out of their
range never gets them and must not keep asking for them. So the Digest names, per origin,
the newest healable frame held, and a Request is answered with healable frames only.

Forwarding schemes (concept 8.1 / 8.3):
    "ours"      contention-based relaying: a receiver waits longer the stronger it heard the
                frame (it is close to the sender and adds little range); far receivers relay
                first, and anyone who overhears the relay cancels its own (as in Meshtastic).
    "flood"     everyone forwards everything at once (the broadcast-storm baseline).
    "unicast3"  like flood, but every transmission goes to three neighbours one by one.

Relevance-limited forwarding (`relay_radius`, a Scenario's `radio.relay_radius_m`): a Track
or Claim is relayed only by a drone within that distance of the place it is about (`here`
is the receiver's position). Nobody further off would fly there, and in a twenty-drone
reserve a three-hop flood of every Track filled the channel. Off by default. Healing still
resends what a neighbour asks for: holding that back too quieted the channel (68 -> 59 %)
but confirmed the reserve's first poacher on 2 of 4 seeds instead of 4 of 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from earshot.messages import (
    BadFrame,
    Claim,
    Digest,
    Message,
    MsgType,
    Request,
    Track,
    TrackStatus,
    decode,
    peek_key,
    with_hops,
)

STORED_TYPES = {MsgType.OBSERVATION, MsgType.TRACK, MsgType.CLAIM}
FWD_MIN_DELAY = 0.05  # s
FWD_SPAN = 1.2  # s added for the strongest receivers
FWD_FULL_MARGIN_DB = 30.0  # margin at which the full delay applies
FWD_SKIP_MARGIN_DB = 20.0  # heard this strongly: the sender is close, my neighbours heard it too
REQUEST_WINDOW = 20  # seqs asked for at most per origin
REQUEST_COOLDOWN = 30.0  # s before the same origin is asked for again
RESEND_COOLDOWN = 10.0  # s before the same frame is resent again
MAX_RESEND_PER_REQUEST = 5
HORIZON = 600.0  # s a stored frame is kept: seqs wrap after 65536 frames (~55 min at 20/s)
SEQ_MOD = 0x10000


def healable(msg: Message) -> bool:
    """Worth healing after a Partition: a Claim, or a Track that is not Explained."""
    return isinstance(msg, Claim) or (isinstance(msg, Track)
                                      and msg.status != TrackStatus.EXPLAINED)


def seq_newer(a: int, b: int) -> bool:
    """a was sent after b, across the 16-bit wrap (serial number arithmetic)."""
    return 0 < (a - b) % SEQ_MOD < SEQ_MOD // 2


@dataclass
class StoreStats:
    accepted: int = 0
    duplicates: int = 0
    bad: int = 0
    forwarded: int = 0
    cancelled: int = 0
    requested: int = 0
    resent: int = 0
    by_origin: dict = field(default_factory=dict)  # origin -> highest seq seen


class MessageStore:
    def __init__(self, me: int, key: bytes, scheme: str = "ours", relay_radius: float = math.inf):
        self.me = me
        self.key = key
        self.scheme = scheme
        self.relay_radius = relay_radius
        self._frames: dict[tuple[int, int, int], bytes] = {}
        self._pending: dict[tuple[int, int, int], tuple[float, bytes, str]] = {}
        self._asked: dict[int, float] = {}  # origin -> last Request time
        self._resent: dict[tuple[int, int, int], float] = {}
        self._kept: dict[tuple[int, int, int], float] = {}  # when each stored frame came
        self._healable: set[tuple[int, int, int]] = set()  # stored frames a Request may resend
        self._now = 0.0  # the latest time receive() or due() was called with
        self._pruned = 0.0
        self.stats = StoreStats()

    def __len__(self) -> int:
        return len(self._frames)

    def add_own(self, frame: bytes) -> None:
        k = peek_key(frame)
        self._frames[k] = frame
        self._kept[k] = self._now
        if healable(decode(frame, self.key)):
            self._healable.add(k)
            self._latest(k[1], k[2])  # in my own Digest too

    def _latest(self, origin: int, seq: int) -> None:
        mine = self.stats.by_origin.get(origin)
        if mine is None or seq_newer(seq, mine):
            self.stats.by_origin[origin] = seq

    def _prune(self, t: float) -> None:
        """Forget frames older than HORIZON, so a wrapped seq is a new frame, not a duplicate,
        and the store stays bounded in a run that goes on for hours."""
        self._now = max(self._now, t)
        if t - self._pruned < 10.0:
            return
        self._pruned = t
        old = [k for k, at in self._kept.items() if t - at > HORIZON]
        for k in old:
            del self._kept[k]
            self._frames.pop(k, None)
            self._resent.pop(k, None)
            self._healable.discard(k)

    # -- receiving ------------------------------------------------------------------------

    def receive(self, frame: bytes, margin_db: float, t: float,
                here: tuple[float, float] | None = None) -> Message | None:
        try:
            msg = decode(frame, self.key)
        except BadFrame:
            self.stats.bad += 1
            return None
        self._prune(t)
        k = peek_key(frame)
        mtype, origin, seq = k
        if k in self._pending:  # someone else relayed or resent it first: stand down
            del self._pending[k]
            self.stats.cancelled += 1
        if origin == self.me:
            return None
        if MsgType(mtype) not in STORED_TYPES:
            return msg  # Digest / Request: handled by the Agent, never stored
        if k in self._frames:
            self.stats.duplicates += 1
            return None
        self._frames[k] = frame
        self._kept[k] = t
        self.stats.accepted += 1
        if healable(msg):
            self._healable.add(k)
            self._latest(origin, seq)
        if msg.header.hops_left > 1 and self._relevant(msg, here):
            self._schedule_forward(k, with_hops(frame, msg.header.hops_left - 1), margin_db, t)
        return msg

    def _relevant(self, msg: Message, here: tuple[float, float] | None) -> bool:
        """Worth relaying from here: anything without a place, or a place within relay_radius."""
        if here is None or math.isinf(self.relay_radius) or not isinstance(msg, Track | Claim):
            return True
        return math.hypot(msg.x - here[0], msg.y - here[1]) <= self.relay_radius

    def _schedule_forward(self, k, frame: bytes, margin_db: float, t: float) -> None:
        if self.scheme == "ours" and margin_db > FWD_SKIP_MARGIN_DB:
            return
        if self.scheme == "ours":
            share = min(max(margin_db, 0.0), FWD_FULL_MARGIN_DB) / FWD_FULL_MARGIN_DB
            due = t + FWD_MIN_DELAY + FWD_SPAN * share + 0.01 * self.me  # id breaks ties
        else:
            due = t
        self._pending[k] = (due, frame, "fwd")

    def due(self, t: float) -> list[bytes]:
        self._prune(t)
        out = []
        for k, (when, frame, why) in sorted(self._pending.items(), key=lambda kv: kv[1][0]):
            if when <= t:
                del self._pending[k]
                out.append(frame)
                if why == "fwd":
                    self.stats.forwarded += 1
                else:
                    self.stats.resent += 1
        if self.scheme == "unicast3":
            out = [f for f in out for _ in range(3)]
        return out

    # -- healing --------------------------------------------------------------------------

    def digest(self) -> list[tuple[int, int]]:
        return sorted(self.stats.by_origin.items())

    def on_digest(self, d: Digest, t: float) -> list[tuple[int, int, int]]:
        """(origin, from_seq, to_seq) ranges to request from the neighbour that sent `d`."""
        out = []
        for origin, theirs in d.entries:
            if origin == self.me:
                continue
            mine = self.stats.by_origin.get(origin)
            newer = mine is None or seq_newer(theirs, mine)
            if newer and t - self._asked.get(origin, -1e9) >= REQUEST_COOLDOWN:
                self._asked[origin] = t
                gap = REQUEST_WINDOW if mine is None else (theirs - mine) % SEQ_MOD
                lo = (theirs - min(gap, REQUEST_WINDOW) + 1) % SEQ_MOD
                out.append((origin, lo, theirs))
                self.stats.requested += 1
        return out

    def on_request(self, r: Request, t: float) -> None:
        """Schedule resends (1 hop) of stored Tracks/Claims in the asked range, newest first."""
        hits = sorted(
            (k for k in self._healable
             if k[1] == r.want_origin
             and (k[2] - r.from_seq) % SEQ_MOD <= (r.to_seq - r.from_seq) % SEQ_MOD
             and t - self._resent.get(k, -1e9) >= RESEND_COOLDOWN),
            key=lambda k: -((k[2] - r.from_seq) % SEQ_MOD),
        )[:MAX_RESEND_PER_REQUEST]
        for i, k in enumerate(hits):
            self._resent[k] = t
            # small, id-staggered delay so one neighbour answers and the others overhear it
            self._pending[k] = (t + 0.2 + 0.15 * self.me + 0.05 * i,
                                with_hops(self._frames[k], 1), "resend")

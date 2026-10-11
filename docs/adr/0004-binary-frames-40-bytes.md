# Radio frames are fixed binary layouts of at most 40 bytes

Every message is `struct`-packed (header 5 B, payload, 10 B truncated HMAC-SHA256) so the same frames could go over real LoRa (concept 8.3: the design must survive a 500 kHz LoRa channel). `hops_left` is excluded from the tag so forwarders can decrement it without the key. `earshot/messages.py` is the single source of truth; `tests/test_messages.py` fails if any layout exceeds 40 B. Process-to-process payloads that never cross the radio (sensing, commands, telemetry) use msgpack instead.

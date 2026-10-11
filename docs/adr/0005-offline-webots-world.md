# The generated Webots world uses base nodes only

Webots R2025a ships almost no PROTO files locally; worlds that use `EXTERNPROTO` download them from GitHub on first load. The venue may be offline, so `tools/build_webots_world.py` emits only base nodes (Pose, Shape, Box, Solid, Robot, Camera), and drones are drawn as simple scaled shapes. Anything that adds an EXTERNPROTO must be opened once online before the event so it lands in Webots' asset cache, and must pass the Wi-Fi-off rehearsal.

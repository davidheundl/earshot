---
status: experiment (branch `gazebo`, 2026-10-13)
---

# Gazebo Sim as a second puppet renderer, behind a bridge process

Branch `gazebo` tries Gazebo Sim (Harmonic, gz-sim 8) as the 3D backend next to Webots, without changing anything above the Backend seam. It keeps ADR 0002's shape: the 2D twin flies the drones, Gazebo only shows them and renders the cameras; drones, Sources and cameras are static models (`DRONE_<id>`, `SRC_<id>[_w<k>]`, `CAMV`, `CAM_<k>`) whose poses `GazeboBackend` sends every 20 ms basic step.

**Bridge process.** Gazebo's Python bindings exist only for Homebrew's Python (3.12+); the project is pinned to 3.11. So `src/earshot/backend/gz_bridge.py` runs in its own venv (`gazebo/.venv`, Homebrew Python with system site-packages plus `gazebo/requirements.txt`) and embeds the Gazebo server (`gz.sim8.TestFixture`): it sets poses in a pre-update callback, steps one iteration per basic step and paces each step to its wall-clock deadline. `GazeboBackend` talks to it over one ZeroMQ REQ/REP socket (`:5554`, msgpack), one round trip per 0.1 s tick. `gz sim -g` is a separate process (macOS cannot run server and GUI in one) and joins over gz-transport.

**Cameras.** The Verifier's camera is a `boundingbox_camera`; each Source's body visuals carry a `Label` per class, so the boxes stand in for a detector the way Webots' recogniser did (out to 100 m, its far clip). A plain camera beside it feeds the dashboard preview. Gazebo renders a sensor only while someone subscribes to it, so the bridge subscribes exactly to the cameras that are on: that is how "only the Verifier sees" and "view cameras only while a screen asks" carry over.

**Isolation.** gz-transport binds the LAN interface and discovers by multicast, so every Gazebo process gets `GZ_IP=127.0.0.1` and `GZ_PARTITION=earshot` (the analog of our Webots port 1240).

## Considered Options

- `gz sim -s` on its own clock, poses through the `set_pose_vector` service: works (under 1 ms per call once warm), but Gazebo then runs beside us instead of in lockstep, and its first calls timed out while the world loaded.
- Moving the whole project to Python 3.14 to import gz directly: touches every environment for one backend.
- A plain camera plus a ground-truth visibility check instead of the box camera: cheaper, but no longer a rendered detector.

## Consequences

- The label render pass costs per visual in the scene (about 0.2 ms each on an M-series laptop: 87 ms per frame with 370 visuals, 13 ms with 43). So the world builder merges each body (drone, truck, wagon, fan) into one OBJ with MTL colours, and trees into one mesh per species. Keep the visual count low (`tests/test_gazebo.py` caps it).
- Gazebo draws an OBJ without normals unlit white: the builder adds vertex normals to every mesh.
- The first camera subscription after start-up takes a few seconds (the renderer starts lazily).

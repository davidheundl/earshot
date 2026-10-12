"""Backend seam: where drone bodies live. Two adapters: the 2D twin and Webots.

Interface (everything a caller must know):

    backend.step(dt, commands) -> {drone_id: DroneState}
        Advance the world by dt seconds. `commands` holds the latest Command per drone;
        drones without a new command keep their previous one. Kinematics, landing on
        Perch Sites, battery drain and motor speed are the backend's business.
    backend.camera_frame(drone_id) -> object | None
        What the camera recognises, only for drones whose last Command had camera_on=True,
        else None. Rendering is the expensive part; nobody else should trigger it. A
        rendering backend may also offer camera_image() for the dashboard preview.
    backend.close()

Both adapters read the same City, so Perch Site ids and coordinates agree.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from earshot.state import Command, DroneState


class Backend(Protocol):
    def step(self, dt: float, commands: Mapping[int, Command]) -> dict[int, DroneState]: ...

    def camera_frame(self, drone_id: int) -> object | None: ...

    def close(self) -> None: ...

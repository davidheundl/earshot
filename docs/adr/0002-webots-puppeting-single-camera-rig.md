# Webots drones are puppeted; the Supervisor carries the only camera

Drones in Webots are plain Pose nodes without physics or controllers. One Supervisor (DEF RIG) moves them every basic time step from the 2D twin's kinematics, so both backends move drones identically and "perch" is just "stop at the roof". A Supervisor cannot read devices mounted on other robots, so the RIG itself carries the single Camera (with Recognition and segmentation) and flies along with whichever Drone is the Verifier. This keeps rendering to one camera on a laptop GPU and makes "only the Verifier sees" literally true.

## Considered Options

- Physics-flown Mavic2Pro per drone with motor PID: realistic but six extern controllers, tuning, and much more CPU. Rejected for a one-day build.
- One extern controller per drone to toggle its own camera: works, but N processes talking to Webots for one image.

## Update 2026-10-11

Segmentation is no longer enabled: nothing read it, and it cost a render pass on every Verifier frame. The Verifier camera renders once per 0.1 s tick, not every 20 ms step.

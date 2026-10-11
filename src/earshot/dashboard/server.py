"""Dashboard server: operator console for a live run.

    python -m earshot.dashboard.server --scenario scenarios/demo.yaml   -> http://127.0.0.1:8000

Subscribes to world "state." and radio "radio." and fans them out over WebSocket. Operator
hotkeys arrive as POST /control and are forwarded to the world as scenario events; Agents
never see the dashboard, and a crash here never stops a run.
The page is a single static file with no external JS (works offline).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from earshot import config
from earshot.runners.wire import pack, topic, unpack
from earshot.scenario import Scenario

STATIC = Path(__file__).parent / "static"


LANDUSE_RGB = {"meadow": (118, 150, 84), "field": (178, 168, 104), "forest": (52, 84, 52),
               "urban": (128, 128, 122), "water": (64, 104, 140), "road": (70, 70, 72),
               "rail": (110, 96, 80), "industrial": (140, 140, 138)}


def basemap(city_json: dict, city_dir: Path) -> bytes | None:
    """Hill-shaded land-use map (PNG, north up) for cities with terrain or land use."""
    import io

    import numpy as np
    from PIL import Image

    if "terrain" not in city_json and "landuse" not in city_json:
        return None
    n = 1024
    if "landuse" in city_json:
        lu = np.array(Image.open(city_dir / city_json["landuse"]["image"]).resize((n, n), Image.NEAREST))
        names = city_json["landuse"]["classes"]
        rgb = np.zeros((n, n, 3))
        for i, name in enumerate(names):
            rgb[lu == i] = LANDUSE_RGB.get(name, (120, 140, 90))
    else:
        rgb = np.full((n, n, 3), 110.0)
    if "terrain" in city_json:
        t = city_json["terrain"]
        H = np.asarray(t["heights"], float).reshape(t["n"], t["n"])[::-1]  # north up
        gy, gx = np.gradient(H, t["spacing"])
        shade = np.clip(1.0 + 1.6 * (-gx - gy) / np.sqrt(2), 0.55, 1.35)  # light from north-west
        shade = np.array(Image.fromarray((shade * 100).astype(np.uint8)).resize((n, n), Image.BILINEAR)) / 100
        rgb *= shade[..., None]
    buf = io.BytesIO()
    Image.fromarray(np.clip(rgb * 0.75, 0, 255).astype(np.uint8)).save(buf, "PNG")  # dimmed
    return buf.getvalue()


def create_app(scenario_path: str) -> FastAPI:
    scenario = Scenario.load(scenario_path)
    city_path = Path(scenario.city)
    city_json = json.loads(city_path.read_text())
    base_png = basemap(city_json, city_path.parent)
    for k in ("terrain", "terrain_outer"):  # big; the basemap carries them
        city_json.pop(k, None)
    latest: dict = {"state": None, "radio": None, "camera": None}
    cams: dict = {}  # view camera k -> {drone, mode, w, h, jpeg (base64)}
    readings: list = []  # (seq, reading): every reading, so no client misses one between frames
    seq = [0]

    async def pump():
        ctx = zmq.asyncio.Context.instance()
        ctx.setsockopt(zmq.LINGER, 0)
        sub = ctx.socket(zmq.SUB)
        sub.connect(config.WORLD_PUB)
        sub.connect(config.RADIO_PUB)
        sub.setsockopt(zmq.SUBSCRIBE, topic("state"))
        sub.setsockopt(zmq.SUBSCRIBE, topic("radio"))
        sub.setsockopt(zmq.SUBSCRIBE, topic("cams"))
        while True:
            t, payload = await sub.recv_multipart()
            msg = unpack(payload)
            if t == topic("cams"):
                for k, f in msg["frames"].items():
                    f["jpeg"] = base64.b64encode(f["jpeg"]).decode()
                    cams[int(k)] = f
                continue
            if t == topic("state"):
                for r in msg.pop("readings", []):
                    seq[0] += 1
                    readings.append((seq[0], r))
                del readings[:-2000]
                if "camera" in msg:
                    cam = msg.pop("camera")
                    if cam:
                        cam["rgba"] = base64.b64encode(cam["rgba"]).decode()
                    latest["camera"] = cam
                for d in msg["drones"]:
                    hm = d.get("telemetry", {}).get("heatmap")
                    if hm:
                        hm["p"] = base64.b64encode(hm["p"]).decode()
                        hm["cls"] = base64.b64encode(hm["cls"]).decode()
            latest["state" if t == topic("state") else "radio"] = msg

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(pump())
        yield
        task.cancel()

    app = FastAPI(title="Earshot dashboard", lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "hud.html")

    to_world = zmq.Context.instance().socket(zmq.PUSH)
    to_world.setsockopt(zmq.LINGER, 0)
    to_world.connect(config.WORLD_PULL)

    @app.post("/control")
    async def control(request: Request):
        op = await request.json()
        if not isinstance(op, dict) or "op" not in op:
            return JSONResponse({"error": "missing op"}, status_code=400)
        to_world.send(pack(op))
        return {"queued": op["op"]}

    scenario_json = {
        "drones": [{"id": d.id, "home": d.home or (d.x, d.y), "default": d.default.name}
                   for d in scenario.drones],
        "bounds": city_json["bounds"],
    }

    @app.get("/scenario")
    async def scenario_info():
        return JSONResponse(scenario_json)

    @app.get("/basemap.png")
    async def basemap_png():
        if base_png is None:
            return Response(status_code=404)
        return Response(base_png, media_type="image/png")

    @app.get("/city")
    async def city():
        return JSONResponse(city_json)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        sent = seq[0] - 200  # a new screen starts with the recent history
        try:
            while True:
                if latest["state"] is not None:
                    new = [r for s, r in readings if s > sent]
                    sent = seq[0]
                    await sock.send_text(json.dumps({**latest, "readings": new}))
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/cams")
    async def ws_cams(sock: WebSocket):
        """Drone camera frames: only frames newer than the last one this client got."""
        await sock.accept()
        last: dict[int, str] = {}
        try:
            while True:
                for k, f in list(cams.items()):
                    if last.get(k) is not f["jpeg"]:
                        last[k] = f["jpeg"]
                        await sock.send_text(json.dumps({"k": k, **f}))
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            pass

    @app.post("/view")
    async def view(request: Request):
        """{"views": [[drone, "pilot" | "chase"], ...]} for the four view cameras."""
        body = await request.json()
        to_world.send(pack({"op": "view", "views": body["views"]}))
        return {"ok": True}

    @app.get("/classic")
    async def classic():
        return FileResponse(STATIC / "index.html")

    return app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenarios/demo.yaml")
    ap.add_argument("--port", type=int, default=config.DASHBOARD_PORT)
    a = ap.parse_args(argv)
    uvicorn.run(create_app(a.scenario), host=config.HOST, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

"""Dashboard server: operator console for a live run.

    python -m earshot.dashboard.server --scenario scenarios/demo.yaml   -> http://127.0.0.1:8000

Set EARSHOT_DASHBOARD_PORT to use a different local port.

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
import logging
import math
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
NO_CACHE = {"Cache-Control": "no-cache"}  # pages change between runs; always revalidate
log = logging.getLogger(__name__)


LANDUSE_RGB = {"meadow": (118, 150, 84), "field": (178, 168, 104), "forest": (52, 84, 52),
               "urban": (128, 128, 122), "water": (64, 104, 140), "road": (70, 70, 72),
               "rail": (110, 96, 80), "industrial": (140, 140, 138)}


def basemap(city_json: dict, city_dir: Path) -> bytes | None:
    """The City's own photo basemap (dimmed JPEG), else a hill-shaded land-use map (PNG),
    north up, for cities with terrain or land use."""
    import io

    import numpy as np
    from PIL import Image

    if city_json.get("basemap"):  # an aerial photo (tools/make_muc.py)
        img = Image.open(city_dir / city_json["basemap"]).convert("RGB")
        buf = io.BytesIO()
        Image.eval(img, lambda v: int(v * 0.8)).save(buf, "JPEG", quality=85)  # dimmed
        return buf.getvalue()
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


QUEUE = 20  # messages per screen (2 s); a screen further behind gets a fresh snapshot
HISTORY = 200  # readings a new screen starts with


def finite(x):
    """x with every NaN and infinity replaced by None: JSON has no NaN, and a single one in
    a state would make the screens' JSON.parse fail on every frame. Bytes pass untouched."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [finite(v) for v in x]
    return x


def dumps(msg) -> str:
    try:
        return json.dumps(msg, allow_nan=False)
    except ValueError:  # rare: pay for the walk only then
        return json.dumps(finite(msg), allow_nan=False)


class Hub:
    """Latest world state for the screens, serialized once per tick for all of them.

    A delta carries the full state except the big parts that change about once per second:
    each drone's telemetry.heatmap only when it changed (else the key is absent, the pages
    keep the last one) and "camera" only when it changed (null when it went away).
    A snapshot carries everything and the recent readings, for a new or lagging screen.
    The Verifier preview ("camera") is big and only the classic page shows it, so it goes
    out only while `want_camera` (set while a classic page is connected).
    """

    def __init__(self):
        self.state: dict | None = None
        self.radio: dict | None = None
        self.camera: dict | None = None
        self.pump_alive = True
        self.readings: list = []  # (seq, reading): every reading, so no screen misses one
        self.seq = 0
        self.version = 0  # bumped per state message
        self._heat: dict = {}  # drone -> (raw p, raw cls, w, h)
        self._heat_json: dict = {}  # drone -> encoded heatmap or None
        self._heat_v: dict = {}  # drone -> version it last changed
        self._camera_v = 0
        self._sent = (0, 0)  # (version, seq) of the last delta
        self.want_camera = False

    def ingest_state(self, msg: dict) -> None:
        self.version += 1
        if self.state is not None and msg["t"] < self.state["t"] - 1.0:
            # a new run: last run's readings are not history, and the screens forget the
            # heatmaps and camera they kept, so send them all again
            self.readings.clear()
            self._heat_v = dict.fromkeys(self._heat_v, self.version)
            self._camera_v = self.version
        for r in msg.pop("readings", []):
            self.seq += 1
            self.readings.append((self.seq, r))
        del self.readings[:-2000]
        if "camera" in msg:
            cam = msg.pop("camera")
            if cam:
                cam["rgba"] = base64.b64encode(cam["rgba"]).decode()
            if self._camera_v == 0 or cam != self.camera:  # the first one counts, even None
                self.camera = cam
                self._camera_v = self.version
        for d in msg["drones"]:
            tm = d.get("telemetry")
            if not isinstance(tm, dict) or "heatmap" not in tm:
                continue
            hm = tm.pop("heatmap")
            raw = (hm["p"], hm["cls"], hm.get("w"), hm.get("h")) if hm else None
            if d["id"] not in self._heat or raw != self._heat[d["id"]]:
                self._heat[d["id"]] = raw
                self._heat_json[d["id"]] = None if hm is None else {
                    **hm, "p": base64.b64encode(hm["p"]).decode(),
                    "cls": base64.b64encode(hm["cls"]).decode()}
                self._heat_v[d["id"]] = self.version
        self.state = msg

    def _message(self, since_version: int | None, since_seq: int) -> dict:
        drones = []
        for d in self.state["drones"]:
            i = d["id"]
            if i in self._heat_json and (since_version is None
                                         or self._heat_v[i] > since_version):
                d = {**d, "telemetry": {**d.get("telemetry", {}), "heatmap": self._heat_json[i]}}
            drones.append(d)
        out = {"state": {**self.state, "drones": drones}, "radio": self.radio,
               "readings": [r for s, r in self.readings if s > since_seq]}
        if self.want_camera and (since_version is None or self._camera_v > since_version):
            out["camera"] = self.camera
        return out

    def delta_text(self) -> str | None:
        """What changed since the last delta, or None if no new state arrived."""
        if self.state is None or self.version == self._sent[0]:
            return None
        msg = self._message(*self._sent)
        self._sent = (self.version, self.seq)
        return dumps(msg)

    def snapshot_text(self) -> str:
        """Everything as of the last delta, for a screen that joins now."""
        s = self._sent[1]
        msg = self._message(None, s - HISTORY)
        # pending changes go out with the next delta; readings must not arrive twice
        msg["readings"] = [r for q, r in self.readings if s - HISTORY < q <= s]
        return dumps(msg)


def offer(q: asyncio.Queue, text: str, hub: Hub) -> None:
    """Queue a delta; a screen too far behind drops its backlog and gets a fresh snapshot."""
    try:
        q.put_nowait(text)
    except asyncio.QueueFull:
        while not q.empty():
            q.get_nowait()
        q.put_nowait(hub.snapshot_text())


def create_app(scenario_path: str) -> FastAPI:
    scenario = Scenario.load(scenario_path)
    city_path = Path(scenario.city)
    city_json = json.loads(city_path.read_text())
    base_png = basemap(city_json, city_path.parent)
    for k in ("terrain", "terrain_outer"):  # big; the basemap carries them
        city_json.pop(k, None)
    hub = Hub()
    cams: dict = {}  # view camera k -> {drone, mode, w, h, jpeg (base64)}
    listeners: dict[asyncio.Queue, dict] = {}  # /ws/audio screen -> {"drone": the one it hears}

    async def pump():
        try:
            ctx = zmq.asyncio.Context.instance()
            ctx.setsockopt(zmq.LINGER, 0)
            sub = ctx.socket(zmq.SUB)
            sub.connect(config.WORLD_PUB)
            sub.connect(config.RADIO_PUB)
            sub.setsockopt(zmq.SUBSCRIBE, topic("state"))
            sub.setsockopt(zmq.SUBSCRIBE, topic("radio"))
            sub.setsockopt(zmq.SUBSCRIBE, topic("cams"))
            sub.setsockopt(zmq.SUBSCRIBE, topic("audio"))
            while True:
                t, payload = await sub.recv_multipart()
                try:  # one bad message must not freeze the screen
                    msg = unpack(payload)
                    if t == topic("cams"):
                        for k, f in msg["frames"].items():
                            f["jpeg"] = base64.b64encode(f["jpeg"]).decode()
                            cams[int(k)] = f
                    elif t == topic("audio"):
                        for q, want in list(listeners.items()):
                            pcm = msg["windows"].get(str(want["drone"]))
                            if pcm is not None and not q.full():  # a slow screen skips a second
                                q.put_nowait(pcm)
                    elif t == topic("state"):
                        hub.ingest_state(msg)
                    else:
                        hub.radio = finite(msg)
                except Exception:
                    log.exception("dashboard: skipped a bad %r message", t)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("dashboard: the pump stopped; screens are told the data is stale")
        finally:
            hub.pump_alive = False

    clients: set[asyncio.Queue] = set()

    async def broadcast():
        """Serialize each new state once for all screens; per-screen queues, so a slow one
        only delays itself."""
        idle = 0
        while True:
            await asyncio.sleep(0.1)
            try:
                text = hub.delta_text()
                idle = 0 if text is not None else idle + 1
                if text is None and idle >= 10:  # nothing new for 1 s: say why
                    text, idle = dumps({"heartbeat": True, "pump": hub.pump_alive}), 0
                if text is None:
                    continue
                for q in list(clients):
                    offer(q, text, hub)
            except Exception:
                log.exception("dashboard: broadcast failed")

    @asynccontextmanager
    async def lifespan(_app):
        tasks = [asyncio.create_task(pump()), asyncio.create_task(broadcast())]
        yield
        for task in tasks:
            task.cancel()

    app = FastAPI(title="Earshot dashboard", lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "hud.html", headers=NO_CACHE)

    to_world = zmq.Context.instance().socket(zmq.PUSH)
    to_world.setsockopt(zmq.LINGER, 0)
    to_world.setsockopt(zmq.SNDHWM, 100)
    to_world.connect(config.WORLD_PULL)

    def send_world(msg: dict) -> bool:
        """Never block the event loop: with no world running, drop instead of queueing."""
        try:
            to_world.send(pack(msg), zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False

    async def json_body(request: Request):
        try:
            return await request.json()
        except ValueError:
            return None

    @app.post("/control")
    async def control(request: Request):
        op = await json_body(request)
        if not isinstance(op, dict) or "op" not in op:
            return JSONResponse({"error": "missing op"}, status_code=400)
        if not send_world(op):
            return JSONResponse({"error": "no world running"}, status_code=503)
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
        jpeg = base_png[:3] == b"\xff\xd8\xff"
        return Response(base_png, media_type="image/jpeg" if jpeg else "image/png")

    @app.get("/city")
    async def city():
        return JSONResponse(city_json)

    camera_screens: set[asyncio.Queue] = set()  # /ws?camera=1: the classic page

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE)
        if sock.query_params.get("camera"):
            camera_screens.add(q)
            hub.want_camera = True
        if hub.state is not None:  # a new screen starts with everything and the recent history
            q.put_nowait(hub.snapshot_text())
        clients.add(q)  # no await since the snapshot: the next delta continues from it
        try:
            while True:
                await sock.send_text(await q.get())
        except (WebSocketDisconnect, RuntimeError, OSError):  # closed; the type depends on the server
            pass
        finally:
            clients.discard(q)
            camera_screens.discard(q)
            hub.want_camera = bool(camera_screens)

    @app.websocket("/ws/cams")
    async def ws_cams(sock: WebSocket):
        """Drone camera frames: only frames newer than the last one this client got."""
        await sock.accept()
        last: dict[int, str] = {}
        closed = asyncio.Event()

        async def watch():  # without a receive, a closed client is noticed only on a send
            try:
                while True:
                    await sock.receive_text()
            except Exception:  # noqa: BLE001 - closed: stop the sender
                closed.set()

        watcher = asyncio.create_task(watch())
        try:
            while not closed.is_set():
                for k, f in list(cams.items()):
                    if last.get(k) is not f["jpeg"]:
                        last[k] = f["jpeg"]
                        await sock.send_text(json.dumps({"k": k, **f}))
                await asyncio.sleep(0.05)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        finally:
            watcher.cancel()

    @app.websocket("/ws/audio")
    async def ws_audio(sock: WebSocket):
        """The microphone sound of one drone (earshot live --audio): binary 1 s windows,
        float32 mono at 16 kHz, RMS 1.0 = sensors.waveform.FULL_SCALE_DB. The screen picks
        the drone by sending {"drone": id}."""
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        want = {"drone": 0}
        listeners[q] = want

        async def choose():
            try:
                while True:
                    want["drone"] = int(json.loads(await sock.receive_text())["drone"])
            except Exception:  # noqa: BLE001 - closed or garbage: stop sending
                listeners.pop(q, None)
                await q.put(None)

        chooser = asyncio.create_task(choose())
        try:
            while (pcm := await q.get()) is not None:
                await sock.send_bytes(pcm)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        finally:
            chooser.cancel()
            listeners.pop(q, None)

    @app.post("/view")
    async def view(request: Request):
        """{"views": [[drone, "pilot" | "chase"], ...]} for the four view cameras, or
        {"views": null}: the screen still shows them (keepalive), assignment unchanged."""
        body = await json_body(request)
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected {\"views\": ...}"}, status_code=400)
        return {"ok": send_world({"op": "view", "views": body.get("views")})}

    @app.get("/classic")
    async def classic():
        return FileResponse(STATIC / "index.html", headers=NO_CACHE)

    return app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenarios/demo.yaml")
    ap.add_argument("--port", type=int, default=config.DASHBOARD_PORT)
    a = ap.parse_args(argv)
    uvicorn.run(create_app(a.scenario), host=config.HOST, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

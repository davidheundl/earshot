"""Dashboard server: subscribes to world "state." and radio "radio.", fans out over WebSocket.

    python -m earshot.dashboard.server --scenario scenarios/demo.yaml   -> http://127.0.0.1:8000

Read-only observer: it never talks back to agents, so it can crash without hurting a run.
The page is a single static file with no external JS (works offline).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from earshot import config
from earshot.runners.wire import topic, unpack
from earshot.scenario import Scenario

STATIC = Path(__file__).parent / "static"


def create_app(scenario_path: str) -> FastAPI:
    scenario = Scenario.load(scenario_path)
    city_json = json.loads(Path(scenario.city).read_text())
    latest: dict = {"state": None, "radio": None}

    async def pump():
        ctx = zmq.asyncio.Context.instance()
        ctx.setsockopt(zmq.LINGER, 0)
        sub = ctx.socket(zmq.SUB)
        sub.connect(config.WORLD_PUB)
        sub.connect(config.RADIO_PUB)
        sub.setsockopt(zmq.SUBSCRIBE, topic("state"))
        sub.setsockopt(zmq.SUBSCRIBE, topic("radio"))
        while True:
            t, payload = await sub.recv_multipart()
            latest["state" if t == topic("state") else "radio"] = unpack(payload)

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(pump())
        yield
        task.cancel()

    app = FastAPI(title="Earshot dashboard", lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/city")
    async def city():
        return JSONResponse(city_json)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                if latest["state"] is not None:
                    await sock.send_text(json.dumps(latest))
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            pass

    return app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenarios/demo.yaml")
    ap.add_argument("--port", type=int, default=config.DASHBOARD_PORT)
    a = ap.parse_args(argv)
    uvicorn.run(create_app(a.scenario), host=config.HOST, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

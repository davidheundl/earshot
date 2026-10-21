"""Start Webots on a private port (not the live 1240) with the test world, run capture.py."""
import os
import subprocess
import sys
import time
from pathlib import Path

repo, world, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
extra = sys.argv[4:]
port = 1247
WH = "/Applications/Webots.app"
env = dict(os.environ)
env.update(WEBOTS_HOME=WH, PYTHONPATH=f"{WH}/Contents/lib/controller/python:{repo / 'src'}",
           DYLD_LIBRARY_PATH=f"{WH}/Contents/lib/controller",
           WEBOTS_CONTROLLER_URL=f"ipc://{port}/supervisor", PYTHONIOENCODING="UTF-8")
out.mkdir(parents=True, exist_ok=True)
with (out / "webots.log").open("w") as log:
    proc = subprocess.Popen([f"{WH}/Contents/MacOS/webots", "--mode=fast", "--batch", f"--port={port}",
                             "--no-rendering", "--minimize", "--stdout", "--stderr", str(world)],
                            stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        time.sleep(8)
        subprocess.run([str(repo / ".venv/bin/python"), str(Path(__file__).parent / os.environ.get("CAPTURE", "capture.py")),
                        str(repo), str(out), *extra], env=env, check=True, timeout=3000)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

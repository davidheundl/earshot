import os
import subprocess
import time
from pathlib import Path

root = Path('/Users/nrieger/Documents/Projects/earshot')
out = Path('/tmp/earshot-cv-next')
source = root/'webots/worlds/demo.wbt'
world = source.read_text()
old = 'name "camera" width 400 height 240'
assert world.count(old)==1
world = world.replace(old,'name "camera" width 1280 height 768')
world = world.replace('"demo_assets/', '"'+str(source.parent/'demo_assets')+'/')
(out/'highres.wbt').write_text(world)
env = dict(os.environ)
env.update(WEBOTS_HOME='/Applications/Webots.app',
           PYTHONPATH='/Applications/Webots.app/Contents/lib/controller/python:'+str(root/'src'),
           DYLD_LIBRARY_PATH='/Applications/Webots.app/Contents/lib/controller',
           WEBOTS_CONTROLLER_URL='ipc://1247/supervisor', PYTHONIOENCODING='UTF-8')
with (out/'webots.log').open('w') as log:
    proc = subprocess.Popen(['/Applications/Webots.app/Contents/MacOS/webots','--mode=fast',
                             '--batch','--port=1247','--no-rendering','--stdout','--stderr',
                             str(out/'highres.wbt')],stdout=log,stderr=subprocess.STDOUT,env=env)
    try:
        time.sleep(5)
        subprocess.run([str(root/'.venv/bin/python'),str(out/'capture_highres.py')],
                       env=env,check=True,timeout=240)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

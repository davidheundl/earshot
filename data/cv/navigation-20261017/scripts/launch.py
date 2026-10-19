import os
import socket
import subprocess
import time
from pathlib import Path

root=Path('/Users/nrieger/Documents/Projects/earshot')
out=Path('/tmp/earshot-nav')
source=root/'webots/worlds/demo.wbt'
world=source.read_text()
needle='children [ Camera { name "camera" width 400 height 240'
replacement=('children [ RangeFinder { name "depth" width 640 height 384 '
             'fieldOfView 1.0 minRange 0.1 maxRange 1000 } '
             'Camera { name "camera" width 640 height 384')
assert world.count(needle)==1
world=world.replace(needle,replacement)
world=world.replace('"demo_assets/','"'+str(source.parent/'demo_assets')+'/')
(out/'nav.wbt').write_text(world)
with socket.socket() as s:
    assert s.connect_ex(('127.0.0.1',1247))!=0, 'Capture port already occupied'
env=dict(os.environ)
env.update(WEBOTS_HOME='/Applications/Webots.app',
           PYTHONPATH='/Applications/Webots.app/Contents/lib/controller/python:'+str(root/'src'),
           DYLD_LIBRARY_PATH='/Applications/Webots.app/Contents/lib/controller',
           WEBOTS_CONTROLLER_URL='ipc://1247/supervisor',NAV_OUTPUT=str(out/'dataset'))
with (out/'webots.log').open('w') as log:
    proc=subprocess.Popen(['/Applications/Webots.app/Contents/MacOS/webots','--mode=fast',
                           '--batch','--port=1247','--no-rendering','--stdout','--stderr',
                           str(out/'nav.wbt')],env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        time.sleep(5)
        subprocess.run([str(root/'.venv/bin/python'),str(out/'capture.py')],env=env,
                       check=True,timeout=180)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

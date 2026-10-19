import os
import subprocess
import time
from pathlib import Path

root = Path('/Users/nrieger/Documents/Projects/earshot')
env = dict(os.environ)
env.update(WEBOTS_HOME='/Applications/Webots.app',
           PYTHONPATH='/Applications/Webots.app/Contents/lib/controller/python:'+str(root/'src'),
           DYLD_LIBRARY_PATH='/Applications/Webots.app/Contents/lib/controller',
           WEBOTS_CONTROLLER_URL='ipc://1247/supervisor', PYTHONIOENCODING='UTF-8')
with open('/tmp/earshot-aircraft-check/webots-capture.log','w') as log:
    proc = subprocess.Popen(['/Applications/Webots.app/Contents/MacOS/webots',
                             '--mode=fast','--batch','--port=1247','--no-rendering',
                             '--stdout','--stderr',str(root/'webots/worlds/demo.wbt')],
                            stdout=log, stderr=subprocess.STDOUT,env=env)
    try:
        time.sleep(5)
        subprocess.run([str(root/'.venv/bin/python'),'/tmp/earshot-aircraft-check/collect.py'],
                       env=env, check=True, timeout=180)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

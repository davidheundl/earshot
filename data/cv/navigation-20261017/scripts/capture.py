import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from controller import Supervisor
from PIL import Image
from earshot.backend.webots import look_rotation

out = Path(os.environ['NAV_OUTPUT'])
(out/'images').mkdir(parents=True,exist_ok=True)
(out/'truth').mkdir(exist_ok=True)
sup = Supervisor()
cam, depth = sup.getDevice('camera'), sup.getDevice('depth')
cam.enable(100)
depth.enable(100)
rig = sup.getFromDef('CAMV')
w,h = cam.getWidth(),cam.getHeight()
f = w/(2*math.tan(cam.getFov()/2))
K = [[f,0,w/2],[0,f,h/2],[0,0,1]]
# Webots ENU optical axes: +x forward, +y left, +z up.
A = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
poses = []
for heading in [0,90,180,270]:
    for distance in [15,35,65]:
        theta=math.radians(heading)
        poses.append(('reference',[-340-distance*math.sin(theta),-520-distance*math.cos(theta),45],heading))
    for distance,offset in [(25,3),(50,-3)]:
        theta=math.radians(heading)
        poses.append(('query',[-338-distance*math.sin(theta),-519-distance*math.cos(theta),46],heading+offset))
# Additional obstacle views toward a building, outside the localization queries.
for distance in [8,15,30]:
    poses.append(('depth',[-500,-400-distance,12],0))
records=[]
for i,(split,pos,heading) in enumerate(poses):
    rot=look_rotation(heading,14)
    rig.getField('translation').setSFVec3f(pos)
    rig.getField('rotation').setSFRotation(rot)
    for _ in range(15):
        if sup.step(20)==-1:
            raise RuntimeError('Webots stopped')
    name=f'{split}_{i:02d}'
    im=Image.frombytes('RGBA',(w,h),bytes(cam.getImage()),'raw','BGRA').convert('RGB')
    im.save(out/'images'/f'{name}.png')
    z=np.asarray(depth.getRangeImage(),dtype=np.float32).reshape(h,w)
    np.save(out/'truth'/f'{name}.npy',z)
    R=cv2.Rodrigues(np.array(rot[:3])*rot[3])[0]@A
    record={'id':name,'split':split,'image':f'images/{name}.png',
            'depth':f'truth/{name}.npy','position':pos,'R_camera_to_world':R.tolist(),'K':K,
            'heading':heading,'width':w,'height':h}
    records.append(record)
    print(name,'finite depth fraction',round(float(np.isfinite(z).mean()),3),flush=True)
(out/'manifest.json').write_text(json.dumps(records,indent=2))
# Separate inference inputs from held-out query truth.
(out/'map.json').write_text(json.dumps([r for r in records if r['split']=='reference'],indent=2))
(out/'queries.json').write_text(json.dumps([
    {k:r[k] for k in ['id','image','K','width','height']} for r in records if r['split']=='query'],indent=2))
sup.simulationQuit(0)

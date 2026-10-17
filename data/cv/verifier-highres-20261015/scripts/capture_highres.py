"""Matched 1280x768 camera stills and a 10 Hz moving-source sequence."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from controller import Supervisor
from PIL import Image
from earshot.backend.webots import look_rotation

out = Path('/tmp/earshot-cv-next/dataset')
(out/'images').mkdir(parents=True, exist_ok=True)
sup = Supervisor()
cam = sup.getDevice('camera')
cam.enable(100)
cam.recognitionEnable(100)
rig = sup.getFromDef('CAMV')
src = sup.getFromDef('SRC_drone1')
children = src.getField('children')
while children.getCount() > 1:
    children.removeMF(children.getCount()-1)
w, h = cam.getWidth(), cam.getHeight()
assert (w,h) == (1280,768)


def capture(steps=15):
    for _ in range(steps):
        if sup.step(20) == -1:
            raise RuntimeError('Webots quit')
    return Image.frombytes('RGBA', (w,h), bytes(cam.getImage()), 'raw','BGRA').convert('RGB')


def boxes():
    result = []
    for obj in cam.getRecognitionObjects():
        node = sup.getFromId(obj.getId())
        while node is not None and not (node.getDef() or '').startswith('SRC_'):
            node = node.getParentNode()
        if node is None or node.getDef() != 'SRC_drone1':
            continue
        x,y = obj.getPositionOnImage()
        bw,bh = obj.getSizeOnImage()
        result.append({'class':'drone','bbox_xyxy':[max(0,x-bw/2),max(0,y-bh/2),
                                                  min(w,x+bw/2),min(h,y+bh/2)]})
    return result


rows = []
for i in range(30):
    distance = [5,15,30,60,90][i%5]
    heading = [0,90,180,270][i//5%4]
    theta = math.radians(heading)
    pos = [-340-distance*math.sin(theta),-520-distance*math.cos(theta),
           30+distance*math.tan(math.radians(14))]
    rig.getField('translation').setSFVec3f(pos)
    rig.getField('rotation').setSFRotation(look_rotation(heading,14))
    src.getField('rotation').setSFRotation([0,0,1,(i%4)*math.pi/4])
    src.getField('translation').setSFVec3f([-340,-520,30] if i<20 else [0,0,-100])
    im = capture()
    objs = boxes()
    # Independent visible-pixel reference from a same-pose target-absent render.
    tight = []
    if i<20:
        src.getField('translation').setSFVec3f([0,0,-100])
        bg = capture()
        diff = np.max(np.abs(np.asarray(im).astype(int)-np.asarray(bg).astype(int)),axis=2)>20
        for obj in objs:
            a,b,c,d = map(int,obj['bbox_xyxy'])
            ys,xs = np.where(diff[b:d,a:c])
            if len(xs):
                tight.append([int(xs.min()+a),int(ys.min()+b),
                              int(xs.max()+a+1),int(ys.max()+b+1)])
    name = f'images/frame_{i:02d}.png'
    im.save(out/name)
    rows.append({'image':name,'source_present':i<20,'horizontal_distance_m':distance,
                 'heading_deg':heading,'objects':objs,'visible_pixel_boxes':tight,
                 'visible_box_method':'paired-render RGB difference >20 inside geometry box'})
(out/'labels.json').write_text(json.dumps(rows,indent=2))
print('Captured 30 matched stills',flush=True)

# Fixed camera; lateral motion at 15 m, then a controlled 1 s disappearance.
rig.getField('translation').setSFVec3f([-340,-535,30+15*math.tan(math.radians(14))])
rig.getField('rotation').setSFRotation(look_rotation(0,14))
src.getField('rotation').setSFRotation([0,0,1,math.pi/4])
video = cv2.VideoWriter(str(out/'moving.avi'),cv2.VideoWriter_fourcc(*'MJPG'),10,(w,h))
assert video.isOpened()
sequence = []
for i in range(100):
    present = not 40<=i<50
    x = -340+4*math.sin(2*math.pi*i/100)
    src.getField('translation').setSFVec3f([x,-520,30] if present else [0,0,-100])
    im = capture(5 if i else 15)
    video.write(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR))
    sequence.append({'frame':i+1,'time_s':i/10,'source_present':present,'objects':boxes()})
video.release()
(out/'moving-labels.json').write_text(json.dumps(sequence,indent=2))
print('Captured 100 moving frames, frames 41–50 target absent',flush=True)
sup.simulationQuit(0)

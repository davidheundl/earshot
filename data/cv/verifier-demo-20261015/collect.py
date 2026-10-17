"""Controlled sampling of Earshot's unmodified demo-world Verifier camera."""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw
from controller import Supervisor
from earshot.backend.webots import look_rotation

out = Path('/tmp/earshot-verifier-frames')
out.mkdir(exist_ok=True)
(out / 'images').mkdir(exist_ok=True)
sup = Supervisor()
cam = sup.getDevice('camera')
cam.enable(100)
cam.recognitionEnable(100)
rig = sup.getFromDef('CAMV')
src = sup.getFromDef('SRC_drone1')
# Source child 0 is the recognizable drone body; later children are demo markers.
children = src.getField('children')
while children.getCount() > 1:
    children.removeMF(children.getCount()-1)
w, h = cam.getWidth(), cam.getHeight()
rows = []
for i in range(30):
    present = i < 20
    distance = [5, 15, 30, 60, 90][i % 5]
    heading = [0, 90, 180, 270][(i // 5) % 4]
    # Preserve backend's 14 degree downwards pitch; move camera around a 30 m Source.
    tx, ty, tz = -340.0, -520.0, 30.0
    theta = math.radians(heading)
    pos = [tx-distance*math.sin(theta), ty-distance*math.cos(theta),
           tz+distance*math.tan(math.radians(14))]
    src.getField('translation').setSFVec3f([tx, ty, tz] if present else [0, 0, -100])
    src.getField('rotation').setSFRotation([0, 0, 1, (i % 4)*math.pi/4])
    rig.getField('translation').setSFVec3f(pos)
    rig.getField('rotation').setSFRotation(look_rotation(heading, 14))
    for _ in range(15):
        if sup.step(20) == -1:
            raise RuntimeError('Webots stopped before capture finished')
    im = Image.frombytes('RGBA', (w,h), bytes(cam.getImage()), 'raw', 'BGRA').convert('RGB')
    objects = []
    for obj in cam.getRecognitionObjects():
        node = sup.getFromId(obj.getId())
        while node is not None and not (node.getDef() or '').startswith('SRC_'):
            node = node.getParentNode()
        if node is None or node.getDef() != 'SRC_drone1':
            continue
        cx, cy = obj.getPositionOnImage()
        bw, bh = obj.getSizeOnImage()
        objects.append({'class':'drone', 'source':'SRC_drone1',
                        'bbox_xyxy':[max(0,cx-bw/2),max(0,cy-bh/2),
                                     min(w,cx+bw/2),min(h,cy+bh/2)]})
    name = f'frame_{i:02d}.png'
    im.save(out / 'images' / name)
    rows.append({'image':'images/'+name, 'width':w, 'height':h,
                 'source_present':present, 'horizontal_distance_m':distance,
                 'camera_position':pos, 'heading_deg':heading, 'pitch_down_deg':14,
                 'objects':objects, 'label_origin':'Webots recognition geometry',
                 'status':'recognized_drone' if objects else ('needs_review' if present else 'empty')})
    print(name, rows[-1]['status'], len(objects), flush=True)
(out/'labels.json').write_text(json.dumps(rows, indent=2))
sheet = Image.new('RGB',(w*5,(h+25)*6),'#222222')
draw = ImageDraw.Draw(sheet)
for i,row in enumerate(rows):
    im = Image.open(out/row['image'])
    d = ImageDraw.Draw(im)
    for obj in row['objects']:
        d.rectangle(obj['bbox_xyxy'], outline='lime', width=1)
    x,y = (i%5)*w,(i//5)*(h+25)
    sheet.paste(im,(x,y))
    draw.text((x+4,y+h+4), f'{i:02d} {row["status"]} {row["horizontal_distance_m"]}m',fill='white')
sheet.save(out/'contact-sheet.jpg')
sup.simulationQuit(0)

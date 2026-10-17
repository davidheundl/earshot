import json
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

root = Path('/Users/nrieger/Documents/Projects/earshot')
base = root/'data/cv/verifier-demo-20261015'
out = Path('/tmp/earshot-cv-next/dataset')
rows = json.loads((out/'labels.json').read_text())
model = YOLO(str(base/'evaluation/drone-best.pt'))


def iou(a,b):
    inter = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0


summary = []
all_predictions = []
for setting, size, highres in [('low_640',640,False),('high_640',640,True),('high_1280',1280,True)]:
    predictions = []
    sheet = Image.new('RGB',(2000,1590),'#222222')
    for n,row in enumerate(rows):
        im = Image.open((out if highres else base)/row['image']).convert('RGB')
        r = model.predict(im,imgsz=size,conf=0.25,device='cpu',verbose=False)[0]
        boxes = [b.xyxy[0].tolist() for b in r.boxes if int(b.cls[0])==0]
        scores = [float(b.conf[0]) for b in r.boxes if int(b.cls[0])==0]
        scale = 1 if highres else 400/1280
        refs = [[v*scale for v in b] for b in row['visible_pixel_boxes']]
        geom = [[v*scale for v in o['bbox_xyxy']] for o in row['objects']]
        best = max((iou(b,g) for b in boxes for g in refs),default=0)
        centered = any(g[0]<=(b[0]+b[2])/2<=g[2] and g[1]<=(b[1]+b[3])/2<=g[3]
                       for b in boxes for g in geom)
        predictions.append({'frame':n,'present':row['source_present'],
                            'distance':row['horizontal_distance_m'],'boxes':boxes,'scores':scores,
                            'iou':best,'centered':centered,'inference_ms':r.speed['inference']})
        draw = ImageDraw.Draw(im)
        for g in refs:
            draw.rectangle(g,outline='lime',width=2)
        for b in boxes:
            draw.rectangle(b,outline='red',width=2)
        im.thumbnail((400,240))
        x,y = n%5*400,n//5*265
        sheet.paste(im,(x,y))
        ImageDraw.Draw(sheet).text((x+3,y+242),f'{n:02d} {row["horizontal_distance_m"]}m IoU {best:.2f}',fill='white')
    sheet.save(out/f'{setting}.jpg')
    totals = []
    for distance in [5,15,30,60,90]:
        group = [p for p in predictions if p['present'] and p['distance']==distance]
        totals.append({'distance':distance,'total':len(group),
                       'iou50_hits':sum(p['iou']>=0.5 for p in group),
                       'center_hits':sum(p['centered'] for p in group)})
    entry = {'setting':setting,'distance_results':totals,
             'empty_false_frames':sum(bool(p['boxes']) for p in predictions if not p['present'])}
    summary.append(entry)
    all_predictions.append({'setting':setting,'frames':predictions})
    print(json.dumps(entry),flush=True)
(out/'comparison.json').write_text(json.dumps(summary,indent=2))
(out/'predictions.json').write_text(json.dumps(all_predictions,indent=2))

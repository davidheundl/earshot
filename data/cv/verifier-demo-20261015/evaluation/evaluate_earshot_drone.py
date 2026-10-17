"""Evaluate a drone checkpoint on controlled Earshot frames; no training or tuning."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO


def iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area / union if union else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m = YOLO(str(a.model))
    print('Classes:', m.names, flush=True)
    target = {k for k,v in m.names.items() if v.lower() == 'quadcopter'}
    assert target, 'Expected quadcopter class'
    rows = json.loads((a.data/'labels.json').read_text())
    records = []
    sheet = Image.new('RGB', (2000, 1590), '#222222')
    for n,row in enumerate(rows):
        im = Image.open(a.data/row['image']).convert('RGB')
        r = m.predict(im, conf=0.25, imgsz=640, device='cpu', verbose=False)[0]
        preds = [{'class':m.names[int(b.cls[0])], 'score':float(b.conf[0]),
                  'box':b.xyxy[0].tolist(), 'target':int(b.cls[0]) in target} for b in r.boxes]
        gt = [o['bbox_xyxy'] for o in row['objects']]
        quad = [d for d in preds if d['target']]
        best = max((iou(d['box'],g) for d in quad for g in gt), default=0)
        # Supplementary diagnostic only: reference boxes are broad geometry bounds.
        center = any(g[0] <= (d['box'][0]+d['box'][2])/2 <= g[2] and
                     g[1] <= (d['box'][1]+d['box'][3])/2 <= g[3] for d in quad for g in gt)
        rec = dict(row, predictions=preds, best_iou=best, matched_iou50=best>=0.5,
                   center_inside_reference=center, inference_ms=r.speed['inference'])
        records.append(rec)
        draw = ImageDraw.Draw(im)
        for g in gt:
            draw.rectangle(g, outline='lime', width=1)
        for d in preds:
            draw.rectangle(d['box'],outline='red',width=1)
            draw.text((d['box'][0], max(0,d['box'][1]-12)),f'{d["class"]} {d["score"]:.2f}',fill='red')
        im.save(a.out/Path(row['image']).name)
        x,y = (n%5)*400,(n//5)*265
        sheet.paste(im,(x,y))
        ImageDraw.Draw(sheet).text((x+3,y+242),f'{n:02d} {row["horizontal_distance_m"]}m IoU {best:.2f} preds {len(preds)}',fill='white')
    summary = {'model_source':'https://huggingface.co/TomSmail/drone-yolo-v1',
               'checkpoint_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
               'classes':m.names, 'confidence':0.25, 'imgsz':640, 'device':'cpu',
               'matching':'quadcopter class and IoU >= 0.5', 'by_distance':[]}
    for distance in [5,15,30,60,90]:
        group = [r for r in records if r['source_present'] and r['horizontal_distance_m']==distance]
        summary['by_distance'].append({'distance_m':distance,'frames':len(group),
            'matched':sum(r['matched_iou50'] for r in group),
            'missed':sum(not r['matched_iou50'] for r in group),
            'center_inside_reference':sum(r['center_inside_reference'] for r in group)})
    empty = [r for r in records if not r['source_present']]
    summary['empty_frames'] = len(empty)
    summary['empty_frames_with_any_drone_prediction'] = sum(bool(r['predictions']) for r in empty)
    summary['empty_false_boxes'] = sum(len(r['predictions']) for r in empty)
    (a.out/'predictions.json').write_text(json.dumps(records,indent=2))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    sheet.save(a.out/'contact-sheet.jpg')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

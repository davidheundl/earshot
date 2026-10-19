import json
import subprocess
from pathlib import Path

root = Path('/Users/nrieger/Documents/Projects/earshot')
here = Path('/tmp/earshot-cv-next')
out = here/'dataset'
model = root/'data/cv/verifier-demo-20261015/evaluation/drone-best.pt'
common = ['--source',str(out/'moving.avi'),'--model',str(model),'--imgsz','1280',
          '--max-frames','0']
for mode in ['tracking','verify']:
    cmd = [str(root/'.venv/bin/python'),str(here/'cv_demos.py'),mode,*common]
    if mode=='tracking':
        cmd += ['--save']
    run = subprocess.run(cmd,cwd=out,text=True,capture_output=True,check=True)
    (out/f'{mode}.log').write_text(run.stdout+'\n'+run.stderr)
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
    assert len(rows)==100
    (out/f'{mode}.json').write_text(json.dumps(rows,indent=2))
    if mode=='tracking':
        for start,stop in [(1,40),(41,50),(51,100)]:
            part = rows[start-1:stop]
            print(mode,start,stop,'frames with tracks',sum(bool(r['tracks']) for r in part),
                  'ids',sorted({t['id'] for r in part for t in r['tracks'] if t['id'] is not None}))
    else:
        for start,stop in [(1,40),(41,50),(51,100)]:
            part = rows[start-1:stop]
            print(mode,start,stop,'confirmed',sum(r['camera_evidence']=='CONFIRM' for r in part))
        assert all(r['camera_evidence']=='INCONCLUSIVE' for r in rows[40:50])
        assert all(r['camera_evidence']!='DISMISS' for r in rows)
# An empty still must remain inconclusive with the actual loaded model.
cmd = [str(root/'.venv/bin/python'),str(here/'cv_demos.py'),'verify','--model',str(model),
       '--source',str(out/'images/frame_20.png')]
run = subprocess.run(cmd,text=True,capture_output=True,check=True)
assert 'INCONCLUSIVE' in run.stdout and 'DISMISS' not in run.stdout
print('Empty still verification passed')

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('/tmp/earshot-nav')
rows = json.loads((root/'results/localization.json').read_text())
truth = {r['id']:r for r in json.loads((root/'dataset/manifest.json').read_text())}
fig, axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
refs = np.array([r['position'] for r in truth.values() if r['split']=='reference'])
axes[0].scatter(refs[:,0],refs[:,1],marker='s',c='#aaaaaa',label='Reference map cameras')
for i,r in enumerate(rows):
    gt=np.array(truth[r['query']]['position'])
    est=np.array(r['estimate']['position'])
    axes[0].scatter(gt[0],gt[1],c='#168a75',label='Query truth' if i==0 else None)
    axes[0].scatter(est[0],est[1],marker='x',c='#d54a37',label='Estimated pose' if i==0 else None)
    axes[0].plot([gt[0],est[0]],[gt[1],est[1]],color='#d54a37')
    axes[0].annotate(r['query'].replace('query_',''),gt[:2],xytext=(4,4),textcoords='offset points')
axes[0].set(xlabel='East (m)',ylabel='North (m)',title='Localization on held-out camera poses')
axes[0].axis('equal')
axes[0].legend(fontsize=8)
axes[1].bar(np.arange(8)-.18,[r['retrieval_error_m'] for r in rows],width=.36,
            color='#aaa',label='Nearest retrieved pose')
axes[1].bar(np.arange(8)+.18,[r['position_error_m'] for r in rows],width=.36,
            color='#168a75',label='SIFT + depth-map PnP')
axes[1].set(xticks=np.arange(8),xticklabels=[r['query'].replace('query_','') for r in rows],
            xlabel='Query ID',ylabel='3D position error (m)',title='Six accurate estimates; two false matches')
axes[1].legend(fontsize=8)
fig.savefig(root/'results/localization-overview.png',dpi=150)
print('median pose error',np.median([r['position_error_m'] for r in rows]))

"""Plot saved metric-depth errors; no new inference or scale fitting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
rows = json.loads((args.data / 'manifest.json').read_text())
metrics = {r['id']: r for r in json.loads((args.data / 'results/depth.json').read_text())}
truths, predictions, names, maes, excluded = [], [], [], [], []
for row in rows:
    if row['split'] == 'reference':
        continue
    gt = np.load(args.data / row['depth'])
    pred = np.load(args.data / 'results' / f"{row['id']}_depth.npy")
    assert gt.shape == pred.shape
    valid = np.isfinite(gt) & (gt > .5) & (gt < 80) & np.isfinite(pred) & (pred > 0)
    if not valid.any():
        excluded.append(row['id'])
        continue
    actual, estimate = gt[valid], pred[valid]
    mae = float(np.abs(estimate - actual).mean())
    assert np.isclose(mae, metrics[row['id']]['mae_m'], atol=1e-4)
    truths.append(actual)
    predictions.append(estimate)
    names.append(row['id'])
    maes.append(mae)
gt, pred = np.concatenate(truths), np.concatenate(predictions)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
fig.subplots_adjust(left=.06, right=.98, bottom=.28, top=.79, wspace=.36)
fig.suptitle('Depth model error on Earshot simulator images', fontsize=19, y=.97)
fig.text(.5, .895, f'Mean frame MAE: {np.mean(maes):.1f} m  |  {len(maes)} evaluated frames  |  '
         f'{len(gt):,} valid pixels  |  No scale correction', ha='center', fontsize=12)
ax = axes[0]
h = ax.hexbin(gt, pred, gridsize=65, extent=(0,80,0,80), mincnt=1, bins='log', cmap='viridis')
ax.plot([0,80],[0,80], '--', color='#d44532', lw=1.8, label='Perfect prediction')
ax.set(xlim=(0,80), ylim=(0,80), xlabel='True depth (m)', ylabel='Predicted depth (m)',
       title='Predictions are mostly too short')
ax.legend(loc='upper left', fontsize=9)
fig.colorbar(h, ax=ax, label='Pixels per hexagon', fraction=.045, pad=.025)
ax = axes[1]
edges = np.arange(0, 81, 10)
centers, medians, lows, highs = [], [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    error = (pred - gt)[(gt >= lo) & (gt < hi)]
    if len(error):
        centers.append((lo+hi)/2)
        low, med, high = np.percentile(error, [10,50,90])
        lows.append(low)
        medians.append(med)
        highs.append(high)
ax.fill_between(centers, lows, highs, alpha=.2, color='#247b91', label='10th–90th percentiles')
ax.plot(centers, medians, 'o-', color='#247b91', label='Median pixel error')
ax.axhline(0, color='#d44532', linestyle='--', lw=1.5)
ax.set(xlim=(0,80), xlabel='True depth (m), 10 m bins', ylabel='Predicted minus true depth (m)',
       title='Negative error = distance underestimated')
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=.2)
ax = axes[2]
ax.barh(names, maes, color='#247b91')
ax.invert_yaxis()
ax.set(xlim=(0,max(maes)*1.22), xlabel='Mean absolute error (m)', title='Error varies between views')
for i, value in enumerate(maes):
    ax.text(value+.7, i, f'{value:.1f}', va='center', fontsize=9)
ax.grid(axis='x', alpha=.2)
fig.text(.06, .12, 'Evaluation uses finite, positive predictions and simulator axial depth between 0.5 and 80 m; sky is excluded.', fontsize=11)
fig.text(.06, .075, f"Excluded frames: {', '.join(excluded)} (no ground-truth pixels in that range). Density and bins pool pixels; headline averages frames.", fontsize=10)
fig.text(.06, .035, 'Depth Anything V2 Metric Outdoor Small • Same simulator scene and lighting • Not real-world validation', fontsize=10, color='#555555')
args.out.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out / 'depth-error-plot.png', dpi=170)
print(json.dumps({'mean_frame_mae_m':float(np.mean(maes)), 'evaluated_frames':len(maes),
                  'excluded':excluded, 'valid_pixels':len(gt)}, indent=2))

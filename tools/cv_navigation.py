"""Offline localization and monocular depth evaluation on mapped simulator views.

uv sync --group cv --group cv-navigation
python tools/cv_navigation.py --data data/cv/navigation-20261017 --out /tmp/nav-results
Reference map depth is permitted localization input; query depth/pose are evaluation only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation

DINO = 'facebook/dinov2-small'
DEPTH = 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf'
REVISIONS = {DINO:'ed25f3a31f01632728cabb09d1542f84ab7b0056',
             DEPTH:'fd2c22027eaf20374204f14099b8341e1925ad39'}


def load_model(repo, cls, cache):
    processor = AutoImageProcessor.from_pretrained(repo, revision=REVISIONS[repo],
                                                  cache_dir=cache, local_files_only=True,
                                                  use_fast=False)
    model = cls.from_pretrained(repo, revision=REVISIONS[repo], cache_dir=cache,
                               local_files_only=True).eval()
    return processor, model


def backproject(uv, depth, K):
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(K).T
    return rays * depth[:, None]


def estimate_pose(ref, kp_ref, des_ref, kp_query, des_query, z, K):
    if des_ref is None or des_query is None or len(des_ref) == 0 or len(des_query) < 2:
        return None
    pairs = cv2.BFMatcher().knnMatch(des_ref, des_query, k=2)
    matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .7*p[1].distance]
    # A query landmark can only support one correspondence.
    unique = {}
    for m in sorted(matches, key=lambda m: m.distance):
        unique.setdefault(m.trainIdx, m)
    matches = list(unique.values())
    if len(matches) < 12:
        return None
    uv = np.array([kp_ref[m.queryIdx].pt for m in matches])
    pixels = np.rint(uv).astype(int)
    d = z[np.clip(pixels[:, 1], 0, z.shape[0]-1),
          np.clip(pixels[:, 0], 0, z.shape[1]-1)]
    valid = np.isfinite(d) & (d > .5) & (d < 200)
    matches = [m for m, v in zip(matches, valid, strict=True) if v]
    if len(matches) < 12:
        return None
    xyz = backproject(uv[valid], d[valid], np.asarray(ref['K']))
    # Coordinates in world axes relative to the reference camera, for conditioning.
    xyz = xyz @ np.asarray(ref['R_camera_to_world']).T
    xy = np.array([kp_query[m.trainIdx].pt for m in matches], dtype=np.float64)
    cv2.setRNGSeed(42)
    ok, r, t, inliers = cv2.solvePnPRansac(
        xyz, xy, K, None, iterationsCount=2000, reprojectionError=3., confidence=.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 12 or len(inliers)/len(matches) < .3:
        return None
    keep = inliers.ravel()
    r, t = cv2.solvePnPRefineLM(xyz[keep], xy[keep], K, None, r, t)
    R = cv2.Rodrigues(r)[0]
    reproj = cv2.projectPoints(xyz[keep], r, t, K, None)[0].reshape(-1, 2)
    err = float(np.median(np.linalg.norm(reproj-xy[keep], axis=1)))
    if err > 3 or np.mean((xyz[keep]@R.T+t.ravel())[:, 2] > 0) < .9:
        return None
    pos = np.asarray(ref['position']) - R.T@t.ravel()
    return {'position':pos.tolist(), 'R_camera_to_world':R.T.tolist(),
            'reference':ref['id'], 'inliers':len(keep), 'matches':len(matches),
            'median_reprojection_px':err, '_matches':[matches[k] for k in keep]}


def pose_hypotheses(ref, kp_ref, des_ref, kp_query, des_query, z, K, limit=4):
    """Peel off dominant match groups so a repeated facade cannot monopolize PnP."""
    if des_ref is None or des_query is None:
        return []
    remaining = np.arange(len(kp_ref))
    candidates = []
    for _ in range(limit):
        candidate = estimate_pose(ref, [kp_ref[i] for i in remaining], des_ref[remaining],
                                  kp_query, des_query, z, K)
        if candidate is None:
            break
        used = [m.queryIdx for m in candidate['_matches']]
        for m in candidate['_matches']:
            m.queryIdx = int(remaining[m.queryIdx])
        candidates.append(candidate)
        remaining = np.delete(remaining, used)
        if len(remaining) < 12:
            break
    return candidates


def select_consistent_pose(candidates, max_distance=2., max_angle_deg=3.):
    """Require support from at least two distinct reference cameras; otherwise abstain."""
    ranked = []
    for candidate in candidates:
        support = {}
        for other in candidates:
            distance = np.linalg.norm(np.asarray(candidate['position']) - other['position'])
            rel = np.asarray(candidate['R_camera_to_world']).T @ other['R_camera_to_world']
            angle = math.degrees(math.acos(np.clip((np.trace(rel)-1)/2, -1, 1)))
            if distance <= max_distance and angle <= max_angle_deg:
                rid = other['reference']
                if rid not in support or other['inliers'] > support[rid]['inliers']:
                    support[rid] = other
        candidate['supporting_references'] = sorted(support)
        ranked.append((len(support), sum(c['inliers'] for c in support.values()),
                       candidate['inliers'], candidate))
    if not ranked:
        return None
    winner = max(ranked, key=lambda item:item[:3])
    if winner[0] < 2:
        return None
    # Equal cross-camera support for a different location is ambiguous, regardless
    # of how many repeated windows voted for either location.
    for count, _, _, candidate in ranked:
        if count >= winner[0] and np.linalg.norm(
                np.asarray(candidate['position']) - winner[3]['position']) > max_distance:
            return None
    return winner[3]


def localization(data, out, cache, strategy='consensus'):
    refs = json.loads((data/'map.json').read_text())
    queries = json.loads((data/'queries.json').read_text())  # no pose or query depth here
    processor, model = load_model(DINO, AutoModel, cache)
    feats = []
    for row in refs+queries:
        inp = processor(images=Image.open(data/row['image']).convert('RGB'), return_tensors='pt')
        with torch.inference_mode():
            f = model(**inp).last_hidden_state[:, 0].numpy()[0]
        feats.append(f/np.linalg.norm(f))
    feats = np.asarray(feats)
    sift = cv2.SIFT_create(nfeatures=3000)
    features = {}
    images = {}
    for row in refs+queries:
        im = cv2.imread(str(data/row['image']))
        images[row['id']] = im
        features[row['id']] = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
    results = []
    for i, query in enumerate(queries):
        sims = feats[:len(refs)] @ feats[len(refs)+i]
        nearest = np.argsort(-sims)[:3]
        candidates = []
        qkp, qdes = features[query['id']]
        for idx in nearest:
            ref = refs[idx]
            kp, des = features[ref['id']]
            candidates.extend(pose_hypotheses(
                ref, kp, des, qkp, qdes, np.load(data/ref['depth']),
                np.asarray(query['K'], dtype=float), limit=4 if strategy == 'consensus' else 1))
        best = (select_consistent_pose(candidates) if strategy == 'consensus'
                else max(candidates, key=lambda c:c['inliers'], default=None))
        result = {'query':query['id'], 'retrieved':[refs[j]['id'] for j in nearest],
                  'retrieval_pose':refs[nearest[0]]['position'], 'estimate':None,
                  'status':('accepted' if best else 'insufficient_cross_reference_agreement'
                            if strategy == 'consensus' else 'no_geometric_pose'),
                  'candidates':[{k:v for k,v in c.items() if not k.startswith('_')}
                                for c in candidates]}
        if best:
            matched = best.pop('_matches')
            refid = best['reference']
            vis = cv2.drawMatches(images[refid], features[refid][0], images[query['id']], qkp,
                                  matched[:60], None, flags=2)
            cv2.imwrite(str(out/f'{query["id"]}_matches.jpg'), vis)
            result['estimate'] = best
        results.append(result)
    # Query truth is loaded only after all estimates have been produced.
    truth = {r['id']:r for r in json.loads((data/'manifest.json').read_text())}
    for result in results:
        gt = truth[result['query']]
        result['retrieval_error_m'] = float(np.linalg.norm(
            np.asarray(result['retrieval_pose'])-gt['position']))
        if result['estimate']:
            est = result['estimate']
            result['position_error_m'] = float(np.linalg.norm(np.asarray(est['position'])-gt['position']))
            rel = np.asarray(est['R_camera_to_world']).T@np.asarray(gt['R_camera_to_world'])
            result['orientation_error_deg'] = math.degrees(math.acos(np.clip((np.trace(rel)-1)/2,-1,1)))
    (out/'localization.json').write_text(json.dumps(results, indent=2))
    summary = {'strategy':strategy,
               'queries':len(results), 'accepted':sum(r['estimate'] is not None for r in results),
               'median_retrieval_error_m':float(np.median([r['retrieval_error_m'] for r in results])),
               'errors_m':[r.get('position_error_m') for r in results],
               'orientation_errors_deg':[r.get('orientation_error_deg') for r in results]}
    print('LOCALIZATION', json.dumps(summary), flush=True)
    return summary


def depth_demo(data, out, cache):
    processor, model = load_model(DEPTH, AutoModelForDepthEstimation, cache)
    rows = [r for r in json.loads((data/'manifest.json').read_text()) if r['split'] != 'reference']
    results = []
    for row in rows:
        im = Image.open(data/row['image']).convert('RGB')
        inp = processor(images=im, return_tensors='pt')
        with torch.inference_mode():
            pred = model(**inp).predicted_depth
            pred = torch.nn.functional.interpolate(pred[:, None], size=(im.height, im.width),
                                                   mode='bicubic', align_corners=False)[0, 0].numpy()
        np.save(out/f'{row["id"]}_depth.npy', pred)
        gt = np.load(data/row['depth'])  # evaluation only; never passed to network
        valid = np.isfinite(gt) & (gt > .5) & (gt < 80) & np.isfinite(pred) & (pred > 0)
        # Some aerial views are entirely beyond the model's 80 m range.
        # Keep those views and report unavailable in-range accuracy explicitly.
        near_valid = np.isfinite(gt) & (gt > .5) & np.isfinite(pred) & (pred > 0)
        near_pred, near_true = pred < 20, gt < 20
        tp = int((near_pred & near_true & near_valid).sum())
        fp = int((near_pred & ~near_true & near_valid).sum())
        fn = int((~near_pred & near_true & near_valid).sum())
        sectors = []
        for a,b in [(0,1/3),(1/3,2/3),(2/3,1)]:
            sl = np.s_[int(im.height*.25):int(im.height*.75),int(im.width*a):int(im.width*b)]
            # Fixed pixel regions; depth quantiles are proximity cues, not safe corridors.
            sectors.append({'predicted_p10_m':float(np.percentile(pred[sl],10)),
                            'true_p10_m':float(np.percentile(np.minimum(gt[sl],1000),10))})
        record = {'id':row['id'],'valid_pixels':int(valid.sum()),
                  'mae_m':float(np.abs(pred[valid]-gt[valid]).mean()) if valid.any() else None,
                  'absrel':float((np.abs(pred[valid]-gt[valid])/gt[valid]).mean()) if valid.any() else None,
                  'near20_tp':tp,'near20_fp':fp,'near20_fn':fn,'sectors_left_center_right':sectors}
        results.append(record)
        def colored(z):
            z = np.nan_to_num(z,nan=80,posinf=80,neginf=0)
            heat = cv2.applyColorMap(np.uint8(np.clip(z/80,0,1)*255),cv2.COLORMAP_TURBO)
            return Image.fromarray(cv2.cvtColor(heat,cv2.COLOR_BGR2RGB))
        overlay = np.asarray(im).copy()
        overlay[near_pred] = (.45*overlay[near_pred]+.55*np.array([255,30,30])).astype('uint8')
        panels = [im, colored(pred), colored(gt), Image.fromarray(overlay)]
        labels = ['RGB', 'Predicted metric depth (0-80m)', 'Simulator depth (0-80m)',
                  'Predicted surface depth <20m (red)']
        sheet = Image.new('RGB',(im.width*2,(im.height+25)*2),'#222222')
        for i,(panel,label) in enumerate(zip(panels,labels,strict=True)):
            x,y = i%2*im.width,i//2*(im.height+25)
            sheet.paste(panel,(x,y))
            ImageDraw.Draw(sheet).text((x+4,y+im.height+5),label,fill='white')
        sheet.save(out/f'{row["id"]}_depth.jpg')
        print('DEPTH', json.dumps(record), flush=True)
    (out/'depth.json').write_text(json.dumps(results,indent=2))
    eligible = [r for r in results if r['mae_m'] is not None]
    return {'frames':len(results), 'frames_with_in_range_truth':len(eligible),
            'mean_frame_mae_m':float(np.mean([r['mae_m'] for r in eligible])) if eligible else None,
            'mean_frame_absrel':float(np.mean([r['absrel'] for r in eligible])) if eligible else None,
            'near20_tp':sum(r['near20_tp'] for r in results),
            'near20_fp':sum(r['near20_fp'] for r in results),
            'near20_fn':sum(r['near20_fn'] for r in results)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',default='/tmp/earshot-nav/models')
    p.add_argument('--download',action='store_true',help='download pinned pretrained weights first')
    p.add_argument('--mode',choices=['all','localization','depth'],default='all')
    p.add_argument('--localization-strategy', choices=['consensus', 'max-inliers'],
                   default='consensus', help='cross-reference hypotheses or original baseline')
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.download:
        from huggingface_hub import snapshot_download
        repos = ([DINO] if args.mode == 'localization' else [DEPTH] if args.mode == 'depth'
                 else [DINO, DEPTH])
        for repo in repos:
            snapshot_download(repo, revision=REVISIONS[repo], cache_dir=args.cache,
                              allow_patterns=['*.json','*.safetensors'])
    cv2.setRNGSeed(42)
    torch.set_num_threads(4)
    summary = {}
    if args.mode in ('all','localization'):
        summary['localization'] = localization(args.data,args.out,args.cache,args.localization_strategy)
    if args.mode in ('all','depth'):
        summary['depth'] = depth_demo(args.data,args.out,args.cache)
    (args.out/f'summary-{args.mode}.json').write_text(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

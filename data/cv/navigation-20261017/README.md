# Visual localization and depth/proximity demos

Captured and evaluated 2026-10-17. This is a controlled Webots experiment with Earshot's
existing demo world. It does not modify or control the swarm's navigation policy.

## Try the results

- [Localization overview](results/localization-overview.png): mapped references, query
  truth, estimated positions and errors.
- [Successful landmark matches](results/query_03_matches.jpg).
- [Confident but incorrect window matches](results/query_19_matches.jpg).
- [Near-obstacle depth comparison](results/depth_22_depth.jpg).
- [Aerial depth comparison](results/query_03_depth.jpg).

The depth images contain four panels: RGB, model depth, simulator depth and red areas
where predicted depth is below 20 m. Both depth color panels use the same 0–80 m scale
(blue near, red far); out-of-range/sky is clipped for display only. Red proximity overlays
include ground and vegetation, not just discrete obstacles. They are not safe-flight paths.

## Visual localization: original baseline

12 reference RGB-D images form a prebuilt map. Eight query images use different positions
(25/50 m offsets versus 15/35/65 m references), changed height and ±3 degree heading.
640×384 camera resolution, 1-radian horizontal FOV and 14-degree downward pitch.
No query is an identical reference frame. Scene, lighting and textures are shared.

Pipeline: pretrained DINOv2-small retrieves three map views; SIFT descriptors match
landmarks; known reference depths and camera poses back-project them into world points;
PnP-RANSAC estimates the query pose. Input queries contain RGB and intrinsics only.
Reference depth is an idealized mapped-geometry input, not monocular predicted depth.
Query truth is loaded only after estimates are generated, for scoring.

| Query | Position error (m) |
|---|---:|
| 03 | 0.063 |
| 04 | 0.272 |
| 08 | 0.449 |
| 09 | 0.485 |
| 13 | 0.266 |
| 14 | 0.089 |
| 18 | 8.016 |
| 19 | 15.993 |

Median position error is 0.361 m, compared with 12.580 m from using the nearest retrieved
reference pose alone. However, **two of eight estimates are grossly wrong**. All eight pass
the original geometric gates (12 inliers, 30% inlier ratio, median reprojection <=3 px,
positive camera depth). Repeated facade windows yield wrong correspondences with very
small reprojection error; high match count is not proof of localization. Orientation errors
are below 0.21 degrees, including the wrongly translated estimates.

This demonstrates promising local map-based localization, not general GPS replacement
or SLAM. It has not been tested across lighting, weather, unseen buildings or map changes.
### Improvement: multiple hypotheses and cross-reference agreement

The default evaluator now removes each dominant RANSAC match group and estimates up
to four hypotheses per reference. This exposes less numerous matches that were hidden
behind repeated-window matches. A pose needs agreement from at least two distinct
reference cameras within 2 m and 3 degrees. Reference count ranks before inlier count;
competing locations with equal reference support cause abstention. Multiple hypotheses
from the same camera count as one vote. PnP uses a fixed seed for each hypothesis.
These are experimental tolerances, not calibrated uncertainty bounds.

On the original eight development queries, six estimates are unchanged and the two
failures improve from 8.016 to 0.057 m and from 15.993 to 0.217 m. All eight are now
within 0.5 m, with median error 0.242 m. Results and all candidate diagnostics are in
`results-consensus/`; the original `results/` and example images remain historical.
No query position or depth is used during estimation or candidate selection.

An additional split removes reference_01, reference_06, reference_11 and reference_16
from the map entirely and uses their RGB/intrinsics as four queries. This tests different
camera poses with a sparser eight-image map, but the same scene and lighting, not an
independent real-world dataset. With the strategy fixed before evaluating this split:

| Withheld reference | Baseline error (m) | Consensus outcome |
|---|---:|---|
| 01 | 0.170 | 0.170 m |
| 06 | 0.418 | Abstain |
| 11 | 15.917 | Abstain |
| 16 | 16.050 | Abstain |

The conservative check rejects both large errors but also rejects one accurate estimate.
Coverage is only 1/4 on this sparser split: do not report rejected cases as zero error.
Cross-reference agreement is not a guarantee; repeated structures shared by references
can still agree on a wrong pose. Next: collect new trajectories and lighting conditions,
then add motion consistency or additional well-spaced reference cameras. No live flight
control has been changed.

Reproduce the default and baseline separately:

```sh
uv run --group cv --group cv-navigation python tools/cv_navigation.py \
  --data data/cv/navigation-20261017 --out /tmp/nav-consensus --mode localization
uv run --group cv --group cv-navigation python tools/cv_navigation.py \
  --data data/cv/navigation-20261017 --out /tmp/nav-baseline --mode localization \
  --localization-strategy max-inliers
```

For the extra split, use `--data data/cv/navigation-20261017/validation-split`.
Its image/depth symlinks point back to the original captures. Baseline and consensus
outputs are saved there in `results-baseline/` and `results-consensus/`.

## Depth and obstacle proximity

Pretrained Depth Anything V2 Metric Outdoor Small runs on the eight queries plus three
lower-altitude scenes. It receives RGB only. A colocated Webots RangeFinder supplies
independent axial-depth truth for evaluation. No per-image scale fitting is applied.
The model config caps depth at 80 m. Two aerial views contain no ground-truth pixels
inside this range, so their in-range error is explicitly null, not silently omitted.

For the nine views with valid 0.5–80 m truth, the mean of per-frame absolute errors is
**42.10 m**, and mean per-frame absolute relative error is **85.3%**. Values are much too
short: an aerial central-region 10th-percentile distance of about 125 m is predicted as
about 9 m. Images show recognizable depth structure, but metric scale does not transfer
well to this scene/camera setup.

Across finite rendered geometry in all eleven views, a predicted-depth <20 m pixel mask
has 63,175 true positives and 1,644,685 false positives: **3.7% precision**. It misses no
near pixels in this tiny set, because it labels enormous areas as near; that is not a
useful obstacle detector. Sky/out-of-range pixels are excluded from these quantitative
counts. Quantiles for fixed left/center/right regions are saved in depth.json; they are
surface-proximity cues, not collision-free corridors.

Do not use the current metric-depth output for clearance decisions. Next: stereo or
multi-view geometry with known camera motion for metric scale, or depth training/calibration
on a separate set of simulator flights followed by held-out evaluation.

## Files and reproducibility

- images/: 23 original RGB renders; truth/: aligned float32 axial depths in metres.
- map.json: known reference geometry and poses.
- queries.json: query RGB/intrinsics only; manifest.json: complete evaluation truth.
- results/: raw predictions, per-frame metrics, matched-landmark images and depth panels.
- scripts/: capture/launch helpers, plotting script and original model-download helper.
  These helpers have local-machine paths and use /tmp/earshot-nav and Webots port 1247.
- tools/cv_navigation.py: reusable command-line evaluator, deterministic RANSAC seed 42.

From Earshot:

```sh
uv sync --group cv --group cv-navigation
uv run --group cv --group cv-navigation python tools/cv_navigation.py \
  --data data/cv/navigation-20261017 --out /tmp/navigation-results --download
```

Use `--mode localization` or `--mode depth` to run one demo. The optional `--download`
fetches pinned safetensors weights; omit it after the models are cached. `--cache` sets
the model cache (default /tmp/earshot-nav/models). No remote model code is enabled.

Model sources:
- [DINOv2-small, Meta](https://huggingface.co/facebook/dinov2-small), revision
  ed25f3a31f01632728cabb09d1542f84ab7b0056.
- [Depth Anything V2 Metric Outdoor Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf),
  revision fd2c22027eaf20374204f14099b8341e1925ad39; trained for metric outdoor depth using
  Virtual KITTI, not this Earshot scene. Credit: Lihe Yang and coauthors.

Geometry tests verify metric back-projection, known-pose recovery and rejection without
features. Full inference results above are measured on these captures, not advertised
model benchmark scores.

# Earshot Verifier camera sample

30 controlled Webots captures of Earshot's existing demo world, captured 2026-10-15.

- 20 frames with one drone Source, at horizontal distances 5, 15, 30, 60 and 90 m,
  from headings 0, 90, 180 and 270 degrees; Source orientation also varies.
- 10 matched empty views: same camera poses as frames 00–09 with the Source parked
  below the ground. Other scene geometry remains unchanged.
- Original Verifier camera: 400 x 240, horizontal FOV 1 radian, pitch down 14 degrees.
- Red intruder marker children removed in memory before capture; world file unmodified.
- Images are simulator camera renders, not aerial-map crops or generated illustrations.

`labels.json` contains pixel xyxy boxes from Webots recognition geometry, camera pose,
Source presence and distance. These are simulator annotations, not predictions from YOLO.
Recognition boxes can differ from tight visible-pixel boxes. Contact sheet was visually
reviewed for rendered content and the absence of red markers; distant drones can be only
a few pixels. Empty means no drone Source in these controlled views, not an empty scene.

`images/` contains raw images without annotation overlays. `contact-sheet.jpg` shows
all 30 images with green reference boxes. `collect.py` and `launch_capture.py` reproduce
the collection on this machine (paths are local and output defaults to /tmp).

Limitations: one synthetic city, one drone mesh, one lighting setup, intentionally centered
Source, controlled placements rather than autonomous flight. No aircraft examples are
included: this demo world has a drone Source. This set is for initial evaluation, not model
training or claims of real-world accuracy. Keep matched/near-duplicate views together when
splitting any larger dataset. Add aircraft and realistic operational camera poses next.

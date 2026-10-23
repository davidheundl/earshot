"""Test world: the built savanna.wbt with absolute asset paths and candidate vehicle models
parked under the ground (DEF TEST_*), for offline Verifier camera renders."""
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
src = repo / "webots/worlds/savanna.wbt"
w = src.read_text()
w = w.replace('"savanna_assets/', '"' + str(src.parent / "savanna_assets") + "/")
P = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/vehicles/protos"
extra = [f'EXTERNPROTO "{P}/range_rover/RangeRoverSportSVRSimple.proto"',
         f'EXTERNPROTO "{P}/tesla/TeslaModel3Simple.proto"']
lines = w.split("\n")
last = max(i for i, l in enumerate(lines) if l.startswith("EXTERNPROTO"))
lines[last + 1:last + 1] = extra
nodes = [
    'DEF TEST_TRUCK TruckSimple { translation 0 0 -100 name "test_truck" color 0.55 0.3 0.12 }',
    'DEF TEST_CAB TruckSimple { translation 0 0 -110 name "test_cab" color 0.55 0.3 0.12 trailer NULL }',
    'DEF TEST_RR RangeRoverSportSVRSimple { translation 0 0 -120 name "test_rr" color 0.32 0.34 0.24 }',
    'DEF TEST_RRW RangeRoverSportSVRSimple { translation 0 0 -130 name "test_rrw" color 0.9 0.9 0.88 }',
    'DEF TEST_TESLA TeslaModel3Simple { translation 0 0 -140 name "test_tesla" color 0.5 0.5 0.5 }',
]
SH = ('Pose {{ translation {x} 0 {z} children [ Shape {{ appearance PBRAppearance {{ baseColor 0 0 0 '
      'transparency 0.35 roughness 1 metalness 0 }} geometry Box {{ size {l} {w} 0.02 }} castShadows FALSE }} ] }}')
rr_sh = SH.format(x=1.25, z=-0.38, l=5.2, w=2.4)
nodes += [
    'DEF TEST_RRS RangeRoverSportSVRSimple { translation 0 0 -150 name "test_rrs" color 0.32 0.34 0.24 '
    f'sensorsSlotCenter [ {rr_sh} ] }}',
    'DEF TEST_RRWS RangeRoverSportSVRSimple { translation 0 0 -160 name "test_rrws" color 0.9 0.9 0.88 '
    f'sensorsSlotCenter [ {rr_sh} ] }}',
    'DEF TEST_TESLAS TeslaModel3Simple { translation 0 0 -170 name "test_teslas" color 0.5 0.5 0.5 '
    f'sensorsSlotCenter [ {SH.format(x=1.4, z=-0.38, l=5.0, w=2.3)} ] }}',
]
pk = next(l for l in lines if l.startswith("DEF SRC_poacher1 Solid"))
pk = pk.replace("DEF SRC_poacher1", "DEF TEST_PKS").replace('"src_poacher1"', '"test_pks"')
pk = pk.replace("children [ Pose { translation 0 0 -0.5 children [ ",
                "children [ Pose { translation 0 0 -0.5 children [ " + SH.format(x=0, z=0.01, l=5.6, w=2.5) + " ", 1)
nodes.append(pk)
if len(sys.argv) > 3 and sys.argv[3] == "shadows":
    lines = [l.replace("TexturedBackgroundLight { texture \"mountains\" castShadows FALSE }",
                       "TexturedBackgroundLight { texture \"mountains\" castShadows TRUE }") for l in lines]
lines += nodes
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print(out)

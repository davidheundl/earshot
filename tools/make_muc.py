"""Build the 'muc' City, Munich Airport 1:1, from open data cached by tools/muc_fetch.py.

    uv run python tools/muc_fetch.py           # once, with internet -> data/raw/muc/
    uv run python tools/make_muc.py            # -> data/cities/muc.json + muc_* assets

Sources (all real, nothing invented):
- buildings: Bavarian LoD2 (measured height, roof type, wall and roof polygons)
- runways, taxiways, aprons, roads, rail, land use: OpenStreetMap
- ground: Bavarian aerial photo (DOP), the Webots ground texture and the dashboard basemap

Frame: EPSG:25832 (UTM 32N) minus ORIGIN, metres, x = east, y = north, so LoD2 needs no
reprojection and the 2 km photo tiles fall on whole kilometres. The ground is taken as flat
(the airfield varies by a few metres); every building stands on z = 0 with its LoD2 height.

Licences: (c) OpenStreetMap contributors, ODbL. LoD2 and DOP: Bayerische
Vermessungsverwaltung, www.geodaten.bayern.de, CC BY 4.0.
"""

from __future__ import annotations

import glob
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely import LineString, MultiPolygon, Polygon, box, make_valid
from shapely.ops import linemerge, unary_union

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "muc"
OUT = ROOT / "data" / "cities"

ORIGIN = (706000.0, 5358000.0)  # EPSG:25832; the airport reference point is at ~(395, 1374)
BOUNDS = (-3000.0, -1000.0, 4000.0, 4000.0)  # both runways + terminals + the S-Bahn approach
LANDUSE_M_PER_PX = 4.0

# land-use classes: the valley's eight (same ids), then the airfield's
LANDUSE = ["meadow", "field", "forest", "urban", "water", "road", "rail", "industrial",
           "runway", "taxiway", "apron"]
ID = {name: i for i, name in enumerate(LANDUSE)}

# roads a truck can use (the vehicle prior); no taxiways, no footpaths, nothing underground
ROAD_CLASSES = {"motorway": 22.0, "trunk": 16.0, "primary": 12.0, "secondary": 10.0,
                "tertiary": 8.0, "unclassified": 6.0, "residential": 6.0, "motorway_link": 7.0,
                "trunk_link": 7.0, "primary_link": 7.0}
LANDUSE_TAGS = {  # OSM (key, value) -> class, painted in this order (later wins)
    ("landuse", "farmland"): "field", ("landuse", "meadow"): "meadow",
    ("landuse", "grass"): "meadow", ("landuse", "forest"): "forest",
    ("natural", "wood"): "forest", ("natural", "water"): "water", ("landuse", "basin"): "water",
    ("landuse", "residential"): "urban", ("landuse", "commercial"): "urban",
    ("landuse", "retail"): "urban", ("landuse", "industrial"): "industrial",
    ("aeroway", "apron"): "apron",
}

GML = "{http://www.opengis.net/gml}"
BLDG = "{http://www.opengis.net/citygml/building/1.0}"
GEN = "{http://www.opengis.net/citygml/generics/1.0}"
LOD2_FLAT = "1000"  # bldg:roofType
LOD2_ROOFS = {"1000": "flat roof", "2100": "pyramidal roof", "2200": "gable roof",
              "3100": "gable roof", "3200": "hip roof", "3300": "hip roof", "3400": "hip roof",
              "3500": "gable roof", "3600": "gable roof", "3700": "gable roof",
              "3800": "gable roof", "3900": "flat roof", "4000": "flat roof", "5000": "flat roof"}

PERCH_MIN_AREA = 600.0  # m2 of flat roof
PERCH_SPACING = 120.0  # m between sites on one large roof (terminals are 1 km long)


# -- OpenStreetMap -----------------------------------------------------------------------------


def load_osm():
    """All cached tiles merged: nodes id -> (x, y) local, ways id -> (node ids, tags),
    relations id -> (members, tags)."""
    tf = Transformer.from_crs(4326, 25832, always_xy=True)
    lonlat: dict[int, tuple[float, float]] = {}
    ways: dict[int, tuple[list[int], dict]] = {}
    rels: dict[int, tuple[list[tuple[str, int, str]], dict]] = {}
    for f in sorted(glob.glob(str(RAW / "osm" / "*.osm"))):
        for _, el in ET.iterparse(f):
            if el.tag == "node":
                lonlat[int(el.get("id"))] = (float(el.get("lon")), float(el.get("lat")))
            elif el.tag == "way":
                ways[int(el.get("id"))] = ([int(n.get("ref")) for n in el.findall("nd")],
                                           {t.get("k"): t.get("v") for t in el.findall("tag")})
                el.clear()
            elif el.tag == "relation":
                rels[int(el.get("id"))] = (
                    [(m.get("type"), int(m.get("ref")), m.get("role")) for m in el.findall("member")],
                    {t.get("k"): t.get("v") for t in el.findall("tag")})
                el.clear()
    ids = list(lonlat)
    xs, ys = tf.transform([lonlat[i][0] for i in ids], [lonlat[i][1] for i in ids])
    nodes = {i: (x - ORIGIN[0], y - ORIGIN[1]) for i, x, y in zip(ids, xs, ys, strict=True)}
    return nodes, ways, rels


def way_coords(nodes, ways, wid) -> list[tuple[float, float]]:
    return [nodes[n] for n in ways[wid][0] if n in nodes]


def areas(nodes, ways, rels, match) -> list[tuple[Polygon | MultiPolygon, dict]]:
    """Closed ways and multipolygon relations whose tags satisfy `match`."""
    out = []
    for wid, (refs, tags) in ways.items():
        if match(tags) and len(refs) > 3 and refs[0] == refs[-1]:
            c = way_coords(nodes, ways, wid)
            if len(c) > 3:
                out.append((make_valid(Polygon(c)), tags))
    for members, tags in rels.values():
        if tags.get("type") != "multipolygon" or not match(tags):
            continue
        outer = [LineString(way_coords(nodes, ways, ref)) for t, ref, role in members
                 if t == "way" and role in ("outer", "") and ref in ways
                 and len(way_coords(nodes, ways, ref)) > 1]
        inner = [LineString(way_coords(nodes, ways, ref)) for t, ref, role in members
                 if t == "way" and role == "inner" and ref in ways
                 and len(way_coords(nodes, ways, ref)) > 1]
        rings = [Polygon(r.coords) for r in _rings(outer)]
        holes = [Polygon(r.coords) for r in _rings(inner)]
        if rings:
            g = unary_union(rings)
            if holes:
                g = g.difference(unary_union(holes))
            out.append((make_valid(g), tags))
    return out


def _rings(lines: list[LineString]) -> list[LineString]:
    merged = linemerge(lines) if lines else None
    if merged is None or merged.is_empty:
        return []
    return [g for g in getattr(merged, "geoms", [merged]) if g.is_ring and len(g.coords) > 3]


def lines(nodes, ways, match) -> list[tuple[LineString, dict]]:
    out = []
    for wid, (_, tags) in ways.items():
        if match(tags):
            c = way_coords(nodes, ways, wid)
            if len(c) > 1:
                out.append((LineString(c), tags))
    return out


def clip_lines(geoms: list[LineString], bounds) -> list[LineString]:
    b = box(*bounds)
    out = []
    for g in geoms:
        c = g.intersection(b)
        for part in getattr(c, "geoms", [c]):
            if isinstance(part, LineString) and part.length > 1.0:
                out.append(part)
    return out


def merged(geoms: list[LineString]) -> list[LineString]:
    if not geoms:
        return []
    m = linemerge(geoms)
    return list(getattr(m, "geoms", [m]))


# -- LoD2 -------------------------------------------------------------------------------------


def _poslist(poly_el) -> np.ndarray:
    txt = poly_el.find(f".//{GML}exterior//{GML}posList")
    v = np.array(txt.text.split(), float).reshape(-1, 3)
    v[:, 0] -= ORIGIN[0]
    v[:, 1] -= ORIGIN[1]
    return v


def load_lod2(bounds) -> list[dict]:
    """Every LoD2 building whose footprint lies in `bounds`: id, footprint (local 2D), height,
    roof type, ground height (ASL), and its wall and roof polygons (local x, y; z above its
    ground) for the Webots meshes."""
    b = box(*bounds)
    out = []
    for f in sorted(glob.glob(str(RAW / "lod2" / "*.gml"))):
        for _, el in ET.iterparse(f):
            if el.tag != f"{BLDG}Building":
                continue
            ground, walls, roofs = [], [], []
            for kind, store in (("GroundSurface", ground), ("WallSurface", walls),
                                ("RoofSurface", roofs)):
                for s in el.iter(f"{BLDG}{kind}"):
                    for p in s.iter(f"{GML}Polygon"):
                        store.append(_poslist(p))
            attrs = {a.get("name"): a.findtext(f"{GEN}value") for a in el.iter(f"{GEN}stringAttribute")}
            h = el.findtext(f"{BLDG}measuredHeight")
            if not ground or h is None:
                el.clear()
                continue
            fp = make_valid(unary_union([Polygon(g[:, :2]) for g in ground if len(g) > 3]))
            if fp.is_empty or not b.contains(fp.centroid):
                el.clear()
                continue
            if isinstance(fp, MultiPolygon):  # keep the main part; LoD2 footprints are one ring
                fp = max(fp.geoms, key=lambda g: g.area)
            if not isinstance(fp, Polygon):
                el.clear()
                continue
            z0 = float(attrs.get("HoeheGrund") or min(g[:, 2].min() for g in ground))
            out.append({
                "id": el.get(f"{GML}id").removeprefix("DEBY_LOD2_"),
                "footprint": fp, "height": float(h), "z0": z0,
                "roof_type": el.findtext(f"{BLDG}roofType") or "",
                "function": el.findtext(f"{BLDG}function") or "",
                "walls": [w - [0, 0, z0] for w in walls], "roofs": [r - [0, 0, z0] for r in roofs],
            })
            el.clear()
    return out


def perch_sites(blds: list[dict]) -> list[dict]:
    """Flat roofs of at least PERCH_MIN_AREA; big ones get a site every PERCH_SPACING m."""
    out = []
    for bd in blds:
        fp = bd["footprint"]
        if bd["roof_type"] != LOD2_FLAT or fp.area < PERCH_MIN_AREA or bd["height"] < 4.0:
            continue
        inner = fp.buffer(-6.0)  # keep off the parapet
        if inner.is_empty:
            continue
        x0, y0, x1, y1 = inner.bounds
        pts = []
        if max(x1 - x0, y1 - y0) > 2 * PERCH_SPACING:
            for x in np.arange(x0 + PERCH_SPACING / 2, x1, PERCH_SPACING):
                for y in np.arange(y0 + PERCH_SPACING / 2, y1, PERCH_SPACING):
                    if inner.contains(_pt(x, y)):
                        pts.append((float(x), float(y)))
        if not pts:
            p = inner.representative_point()
            pts = [(p.x, p.y)]
        for k, (x, y) in enumerate(pts):
            out.append({"id": f"p{bd['id']}_{k}", "building": bd["id"],
                        "x": round(x, 1), "y": round(y, 1), "z": round(bd["height"], 1)})
    return out


def _pt(x, y):
    from shapely import Point

    return Point(x, y)


# -- land use raster ---------------------------------------------------------------------------


def landuse_raster(bounds, osm_areas, runways, taxiways, roads, rails, blds) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    w = round((x1 - x0) / LANDUSE_M_PER_PX)
    h = round((y1 - y0) / LANDUSE_M_PER_PX)
    img = Image.new("L", (w, h), ID["meadow"])  # the airfield between runways is grass
    d = ImageDraw.Draw(img)

    def px(p):
        return ((p[0] - x0) / LANDUSE_M_PER_PX, (y1 - p[1]) / LANDUSE_M_PER_PX)

    def poly(g, cls):
        for part in getattr(g, "geoms", [g]):
            if isinstance(part, Polygon) and not part.is_empty:
                d.polygon([px(p) for p in part.exterior.coords], fill=ID[cls])
                for hole in part.interiors:
                    d.polygon([px(p) for p in hole.coords], fill=ID["meadow"])

    order = list(dict.fromkeys(LANDUSE_TAGS.values()))
    for cls in order:
        for g, tags in osm_areas:
            if any(tags.get(k) == v and LANDUSE_TAGS[(k, v)] == cls for k, v in LANDUSE_TAGS):
                poly(g, cls)
    for g, wdt in taxiways:
        d.line([px(p) for p in g.coords], fill=ID["taxiway"], width=max(1, round(wdt / LANDUSE_M_PER_PX)))
    for g, wdt in runways:
        d.line([px(p) for p in g.coords], fill=ID["runway"], width=max(1, round(wdt / LANDUSE_M_PER_PX)))
    for g, wdt in roads:
        d.line([px(p) for p in g.coords], fill=ID["road"], width=max(1, round(wdt / LANDUSE_M_PER_PX)))
    for g in rails:
        d.line([px(p) for p in g.coords], fill=ID["rail"], width=2)
    for bd in blds:
        poly(bd["footprint"], "industrial" if bd["footprint"].area > 2000 else "urban")
    return np.array(img)


# -- aircraft stands ---------------------------------------------------------------------------


def stands(nodes, ways, bounds) -> list[dict]:
    """OSM aeroway=parking_position lead-in lines, one per stand: where the Webots world parks
    aircraft. A stand split into A/B/... sub-stands for small aircraft keeps only its main line."""
    x0, y0, x1, y1 = bounds
    lines = {}
    for nd, t in ways.values():
        if t.get("aeroway") != "parking_position" or len(nd) < 2 or any(n not in nodes for n in nd):
            continue
        a, b = nodes[nd[0]], nodes[nd[-1]]
        if not all(x0 < x < x1 and y0 < y < y1 for x, y in (a, b)):
            continue
        ref = t.get("ref") or t.get("name") or str(len(lines))
        lines[ref] = [[round(a[0], 1), round(a[1], 1)], [round(b[0], 1), round(b[1], 1)]]
    out = [{"ref": r, "line": ln} for r, ln in sorted(lines.items())
           if not (r[-1:].isalpha() and r[:-1] in lines)]  # 607A, 607B -> 607 when it exists
    return out


# -- ground photo ------------------------------------------------------------------------------


def ground_tiles(bounds, m_per_px: float = 0.5) -> list[dict]:
    """The DOP chunks cropped to the bounds, rescaled to m_per_px, as committed JPEGs."""
    x0, y0, x1, y1 = bounds
    out_dir = OUT / "muc_ground"
    out_dir.mkdir(exist_ok=True)
    tiles = []
    for f in sorted((RAW / "dop").glob("*.jpg")):
        e, n = (int(v) for v in f.stem.split("_"))
        tx0, ty0 = e * 1000 - ORIGIN[0], n * 1000 - ORIGIN[1]
        cx0, cy0 = max(x0, tx0), max(y0, ty0)
        cx1, cy1 = min(x1, tx0 + 2000), min(y1, ty0 + 2000)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        im = Image.open(f)
        s = im.width / 2000.0  # px per m in the source
        crop = im.crop((round((cx0 - tx0) * s), round((ty0 + 2000 - cy1) * s),
                        round((cx1 - tx0) * s), round((ty0 + 2000 - cy0) * s)))
        crop = crop.resize((_pow2((cx1 - cx0) / m_per_px), _pow2((cy1 - cy0) / m_per_px)),
                           Image.LANCZOS)  # GPU textures want powers of two
        name = f"{round(cx0)}_{round(cy0)}.jpg"
        crop.save(out_dir / name, quality=82, optimize=True)
        tiles.append({"image": f"muc_ground/{name}", "bounds": [cx0, cy0, cx1, cy1]})
    return tiles


def _pow2(px: float) -> int:
    return 1 << max(0, round(math.log2(max(px, 1.0))))


def basemap(tiles: list[dict], bounds, width_px: int = 4096) -> str:
    """One downsampled photo of the whole City for the dashboard, north up."""
    x0, y0, x1, y1 = bounds
    s = width_px / (x1 - x0)
    canvas = Image.new("RGB", (width_px, round((y1 - y0) * s)), (110, 130, 90))
    for t in tiles:
        tx0, ty0, tx1, ty1 = t["bounds"]
        im = Image.open(OUT / t["image"]).resize((round((tx1 - tx0) * s), round((ty1 - ty0) * s)),
                                                  Image.LANCZOS)
        canvas.paste(im, (round((tx0 - x0) * s), round((y1 - ty1) * s)))
    canvas.save(OUT / "muc_basemap.jpg", quality=85, optimize=True)
    return "muc_basemap.jpg"


# -- meshes for Webots -------------------------------------------------------------------------


def triangles(poly3d: np.ndarray) -> list[np.ndarray]:
    """A planar 3D polygon (closed ring) -> triangles, via its best-fitting 2D projection."""
    import shapely

    v = poly3d[:-1] if np.allclose(poly3d[0], poly3d[-1]) else poly3d
    if len(v) < 3:
        return []
    nrm = np.zeros(3)  # Newell's method: robust for any planar polygon
    for a, b in zip(v, np.roll(v, -1, axis=0), strict=True):
        nrm += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]),
                (a[0] - b[0]) * (a[1] + b[1])]
    drop = int(np.argmax(np.abs(nrm)))
    keep = [i for i in range(3) if i != drop]
    p2 = Polygon(v[:, keep])
    if not p2.is_valid or p2.area < 0.05:
        return []
    tris = shapely.constrained_delaunay_triangles(p2)
    out = []
    lookup = {tuple(np.round(q, 3)): q3 for q, q3 in zip(v[:, keep], v, strict=True)}
    for t in getattr(tris, "geoms", []):
        c = list(t.exterior.coords)[:3]
        try:
            tri = np.array([lookup[tuple(np.round(q, 3))] for q in c])
        except KeyError:
            continue
        if np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), nrm) < 0:
            tri = tri[::-1]  # face outward like the LoD2 ring (CityGML: ccw seen from outside)
        out.append(tri)
    return out


def write_obj(path: Path, tris: list[np.ndarray]) -> None:
    lines_, k = [], 1
    for t in tris:
        for p in t:
            lines_.append(f"v {p[0]:.2f} {p[1]:.2f} {p[2]:.2f}")
        lines_.append(f"f {k} {k + 1} {k + 2}")
        k += 3
    path.write_text("\n".join(lines_) + "\n")


def building_meshes(blds: list[dict], bounds, tile_m: float = 1000.0) -> list[dict]:
    """Walls and roofs of every building, merged per tile_m square and surface kind, so a
    camera only draws the tiles in view. Returns [{"walls": file, "roofs": file}]."""
    x0, y0, _, _ = bounds
    groups: dict[tuple[int, int, str], list[np.ndarray]] = {}
    for bd in blds:
        c = bd["footprint"].centroid
        key = (int((c.x - x0) // tile_m), int((c.y - y0) // tile_m))
        for kind in ("walls", "roofs"):
            g = groups.setdefault((*key, kind), [])
            for poly in bd[kind]:
                g.extend(triangles(poly))
    out_dir = OUT / "muc_mesh"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.obj"):
        old.unlink()
    files: dict[tuple[int, int], dict] = {}
    for (i, j, kind), tris in sorted(groups.items()):
        if not tris:
            continue
        name = f"{kind}_{i}_{j}.obj"
        write_obj(out_dir / name, tris)
        files.setdefault((i, j), {})[kind] = f"muc_mesh/{name}"
    return list(files.values())


# -- main --------------------------------------------------------------------------------------


def main() -> None:
    bx0, by0, bx1, by1 = BOUNDS
    cells = math.ceil((bx1 - bx0) / 100) * math.ceil((by1 - by0) / 100)
    assert cells <= 4096, f"{cells} 100 m cells: task ids allow 4096 per kind"
    print(f"[muc] bounds {BOUNDS}: {cells} task cells")

    nodes, ways, rels = load_osm()
    print(f"[muc] osm: {len(nodes)} nodes, {len(ways)} ways, {len(rels)} relations")

    def underground(t):
        return t.get("tunnel", "no") != "no" or t.get("layer", "0").startswith("-")

    rw = lines(nodes, ways, lambda t: t.get("aeroway") == "runway")
    runways = [(g, float(t.get("width", 60))) for g, t in rw]
    tx = lines(nodes, ways, lambda t: t.get("aeroway") == "taxiway")
    taxiways = [(g, float(t.get("width", 23)) if t.get("width", "").replace(".", "").isdigit()
                 else 23.0) for g, t in tx]
    road_ls = lines(nodes, ways, lambda t: t.get("highway") in ROAD_CLASSES and not underground(t))
    rail_ls = lines(nodes, ways, lambda t: t.get("railway") in ("rail", "light_rail")
                    and t.get("service") is None and not underground(t))
    osm_areas = areas(nodes, ways, rels,
                      lambda t: any(t.get(k) == v for k, v in LANDUSE_TAGS))

    roads, widths = [], []
    for cls, wdt in ROAD_CLASSES.items():
        for g in merged(clip_lines([g for g, t in road_ls if t["highway"] == cls], BOUNDS)):
            if g.length >= 80.0:
                roads.append(g.simplify(1.0))
                widths.append(wdt)
    rails = [g.simplify(0.5) for g in merged(clip_lines([g for g, _ in rail_ls], BOUNDS))
             if g.length >= 300.0]
    rails.sort(key=lambda g: -g.length)
    print(f"[muc] {len(runways)} runways, {len(taxiways)} taxiways, {len(roads)} roads, "
          f"{len(rails)} surface rail lines (longest {rails[0].length:.0f} m)")

    blds = load_lod2(BOUNDS)
    sites = perch_sites(blds)
    print(f"[muc] lod2: {len(blds)} buildings, {len(sites)} Perch Sites")

    lu = landuse_raster(BOUNDS, osm_areas, runways, taxiways, list(zip(roads, widths, strict=True)),
                        rails, blds)
    Image.fromarray(lu).save(OUT / "muc_landuse.png", optimize=True)
    tiles = ground_tiles(BOUNDS)
    meshes = building_meshes(blds, BOUNDS)

    def ring(g):
        return [[round(x, 1), round(y, 1)] for x, y in g.exterior.coords]

    def pts(g):
        return [[round(x, 1), round(y, 1)] for x, y in g.coords]

    city = {
        "name": "muc",
        "origin": {"lat": 48.353783, "lon": 11.786086, "epsg": 25832,
                   "utm_origin": list(ORIGIN), "ground_asl": round(float(np.median(
                       [bd["z0"] for bd in blds])), 1) if blds else None},
        "bounds": list(BOUNDS),
        "landuse": {"image": "muc_landuse.png", "classes": LANDUSE},
        "ground_tiles": tiles,
        "basemap": basemap(tiles, BOUNDS) if tiles else None,
        "building_meshes": meshes,
        "buildings": [{"id": bd["id"], "footprint": ring(bd["footprint"].simplify(0.3)),
                       "height": round(bd["height"], 2),
                       "roof": LOD2_ROOFS.get(bd["roof_type"], "flat roof"),
                       "floors": max(1, round(bd["height"] / 4.0))} for bd in blds],
        "perch_sites": sites,
        "rails": [pts(g) for g in rails],
        "roads": [pts(g) for g in roads],
        "road_widths": widths,
        "runways": [{"ref": t.get("ref"), "width": wdt, "line": pts(g)}
                    for (g, t), (_, wdt) in zip(rw, runways, strict=True)],
        "stands": stands(nodes, ways, BOUNDS),
        "attribution": ["(c) OpenStreetMap contributors, ODbL",
                        "Bayerische Vermessungsverwaltung, www.geodaten.bayern.de, CC BY 4.0"],
    }
    (OUT / "muc.json").write_text(json.dumps(city, separators=(",", ":")))
    size = (OUT / "muc.json").stat().st_size / 1e6
    print(f"[muc] data/cities/muc.json: {size:.1f} MB, {len(tiles)} ground tiles, "
          f"{len(meshes)} mesh tiles")


if __name__ == "__main__":
    main()

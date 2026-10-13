"""Download the open data for the Munich Airport City into data/raw/muc/ (gitignored).

    uv run python tools/muc_fetch.py osm      OpenStreetMap, tiled API calls (polite, resumable)
    uv run python tools/muc_fetch.py lod2     Bavarian LoD2 CityGML tiles (2 x 2 km, EPSG:25832)
    uv run python tools/muc_fetch.py dop      Bavarian aerial photo (DOP) via WMS, ~1 m/px
    uv run python tools/muc_fetch.py ring     the surroundings: 4 km more of all three, photo at 2 m/px

Run once on a machine with internet; tools/make_muc.py builds data/cities/muc.json from the
cache, and that file is committed, so nothing here runs at the venue.

Licences: OSM (c) OpenStreetMap contributors, ODbL. LoD2 and DOP: Bayerische
Vermessungsverwaltung, www.geodaten.bayern.de, CC BY 4.0.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "muc"
UA = "earshot-hackathon/0.1 (city builder; github.com/davidheundl/earshot)"

# WGS84 box around the airfield (both runways, terminals, cargo, the S-Bahn approaches)
LON0, LAT0, LON1, LAT1 = 11.70, 48.33, 11.86, 48.38
TILE_LON, TILE_LAT = 0.02, 0.01  # ~1.5 x 1.1 km; the terminal tile has ~10k nodes (limit 50k)

# EPSG:25832 kilometre tiles that LoD2 is cut into (2 x 2 km, named by their lower-left km)
LOD2_URL = "https://download1.bayernwolke.de/a/lod2/citygml/{e}_{n}.gml"
DOP_WMS = "https://geoservices.bayern.de/od/wms/dop/v1/dop40"


def get(url: str, out: Path, timeout: float = 120.0) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} {url}", flush=True)
        return False
    except (urllib.error.URLError, TimeoutError) as e:  # resumable: the next run fetches it
        print(f"  skipped ({e}) {url}", flush=True)
        return False
    tmp = out.with_suffix(out.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(out)
    return True


def osm_split(e, lon0, lat0, lon1, lat1, depth) -> None:
    if e.code == 400 and depth < 3:  # too many nodes: split in four
        mx, my = (lon0 + lon1) / 2, (lat0 + lat1) / 2
        for a, b, c, d in ((lon0, lat0, mx, my), (mx, lat0, lon1, my),
                           (lon0, my, mx, lat1), (mx, my, lon1, lat1)):
            osm_tile(a, b, c, d, depth + 1)
        return
    raise e


def osm_tile(lon0: float, lat0: float, lon1: float, lat1: float, depth: int = 0) -> None:
    out = RAW / "osm" / f"{lon0:.4f}_{lat0:.4f}_{lon1:.4f}_{lat1:.4f}.osm"
    if out.exists():
        return
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={lon0},{lat0},{lon1},{lat1}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
    except (urllib.error.URLError, TimeoutError) as e:
        if isinstance(e, urllib.error.HTTPError):
            return osm_split(e, lon0, lat0, lon1, lat1, depth)
        print(f"  skipped ({e}) {out.name}", flush=True)  # resumable: the next run fetches it
        return
    out.write_bytes(data)
    print(f"  osm {out.name}: {len(data) / 1e6:.1f} MB", flush=True)
    time.sleep(1.0)  # the API's usage policy: no parallel or rapid-fire requests


def fetch_osm() -> None:
    (RAW / "osm").mkdir(parents=True, exist_ok=True)
    nx = round((LON1 - LON0) / TILE_LON)
    ny = round((LAT1 - LAT0) / TILE_LAT)
    for j in range(ny):
        for i in range(nx):
            lon0, lat0 = LON0 + i * TILE_LON, LAT0 + j * TILE_LAT
            osm_tile(round(lon0, 4), round(lat0, 4),
                     round(lon0 + TILE_LON, 4), round(lat0 + TILE_LAT, 4))


def lod2_tiles() -> list[tuple[int, int]]:
    from pyproj import Transformer

    tf = Transformer.from_crs(4326, 25832, always_xy=True)
    xs, ys = zip(*(tf.transform(lon, lat) for lon in (LON0, LON1) for lat in (LAT0, LAT1)),
                 strict=True)
    e0, e1 = int(min(xs) // 2000) * 2, int(max(xs) // 2000) * 2
    n0, n1 = int(min(ys) // 2000) * 2, int(max(ys) // 2000) * 2
    return [(e, n) for e in range(e0, e1 + 2, 2) for n in range(n0, n1 + 2, 2)]


def fetch_lod2() -> None:
    (RAW / "lod2").mkdir(parents=True, exist_ok=True)
    for e, n in lod2_tiles():
        out = RAW / "lod2" / f"{e}_{n}.gml"
        if out.exists():
            continue
        if get(LOD2_URL.format(e=e, n=n), out, timeout=300):
            print(f"  lod2 {out.name}: {out.stat().st_size / 1e6:.1f} MB", flush=True)


# the City's bounds in EPSG:25832 (make_muc.ORIGIN + BOUNDS): the photo is only needed there
DOP_BOUNDS = (703000, 5357000, 710000, 5362000)


def fetch_dop(m_per_px: float = 0.5, chunk_m: float = 2000.0) -> None:
    """The aerial photo in 2 x 2 km JPEG chunks (EPSG:25832), cropped and scaled by make_muc."""
    (RAW / "dop").mkdir(parents=True, exist_ok=True)
    px = int(chunk_m / m_per_px)
    x0b, y0b, x1b, y1b = DOP_BOUNDS
    tiles = [(e, n) for e, n in lod2_tiles()
             if e * 1000 < x1b and (e + 2) * 1000 > x0b and n * 1000 < y1b and (n + 2) * 1000 > y0b]
    for e, n in tiles:
        x0, y0 = e * 1000, n * 1000
        out = RAW / "dop" / f"{e}_{n}.jpg"
        if out.exists():
            continue
        url = (f"{DOP_WMS}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=by_dop40c&STYLES="
               f"&SRS=EPSG:25832&BBOX={x0},{y0},{x0 + chunk_m},{y0 + chunk_m}"
               f"&WIDTH={px}&HEIGHT={px}&FORMAT=image/jpeg")
        if get(url, out):
            print(f"  dop {out.name}: {out.stat().st_size / 1e6:.1f} MB", flush=True)
        time.sleep(0.5)


# the surroundings: the City's bounds plus 4 km on every side (EPSG:25832), and a WGS84 box
# that is whole OSM tiles beyond the airfield box, so cached tiles are reused by name
RING_BOUNDS = (699000, 5353000, 714000, 5366000)
RING_LON0, RING_LAT0, RING_LON1, RING_LAT1 = 11.64, 48.30, 11.92, 48.42


def fetch_ring() -> None:
    """Coarser photo, LoD2 and OSM for the land around the City (tools/make_muc.py surround)."""
    global LON0, LAT0, LON1, LAT1
    (RAW / "dop_ring").mkdir(parents=True, exist_ok=True)
    x0b, y0b, x1b, y1b = RING_BOUNDS
    chunk, px = 2000, 1000  # 2 m/px
    for e in range(x0b // 1000 - x0b // 1000 % 2, x1b // 1000, 2):
        for n in range(y0b // 1000 - y0b // 1000 % 2, y1b // 1000, 2):
            out = RAW / "dop_ring" / f"{e}_{n}.jpg"
            if out.exists():
                continue
            url = (f"{DOP_WMS}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=by_dop40c&STYLES="
                   f"&SRS=EPSG:25832&BBOX={e * 1000},{n * 1000},{e * 1000 + chunk},{n * 1000 + chunk}"
                   f"&WIDTH={px}&HEIGHT={px}&FORMAT=image/jpeg")
            if get(url, out):
                print(f"  dop_ring {out.name}: {out.stat().st_size / 1e6:.2f} MB", flush=True)
            time.sleep(0.5)
    (RAW / "lod2").mkdir(parents=True, exist_ok=True)
    for e in range(x0b // 1000 - x0b // 1000 % 2, x1b // 1000, 2):
        for n in range(y0b // 1000 - y0b // 1000 % 2, y1b // 1000, 2):
            out = RAW / "lod2" / f"{e}_{n}.gml"
            if not out.exists() and get(LOD2_URL.format(e=e, n=n), out, timeout=300):
                print(f"  lod2 {out.name}: {out.stat().st_size / 1e6:.1f} MB", flush=True)
    LON0, LAT0, LON1, LAT1 = RING_LON0, RING_LAT0, RING_LON1, RING_LAT1
    fetch_osm()


if __name__ == "__main__":
    what = sys.argv[1:] or ["osm", "lod2", "dop", "ring"]
    for w in what:
        print(f"[fetch] {w}", flush=True)
        {"osm": fetch_osm, "lod2": fetch_lod2, "dop": fetch_dop, "ring": fetch_ring}[w]()

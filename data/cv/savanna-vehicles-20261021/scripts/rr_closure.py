"""The Webots cache files a PROTO pulls in: the PROTO, its EXTERNPROTOs and the files its url
fields name, recursively (cache file name = sha1 of the URL). Prints `sha1 present url`."""
import hashlib
import re
import sys
from pathlib import Path

CACHE = Path.home() / "Library/Caches/Cyberbotics/Webots/assets"
BASE = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/"


def resolve(url: str, parent: str) -> str:
    if url.startswith("webots://"):
        return BASE + url[len("webots://"):]
    if url.startswith("https://"):
        return url
    return parent.rsplit("/", 1)[0] + "/" + url


seen: dict[str, bool] = {}


def visit(url: str) -> None:
    if url in seen:
        return
    h = hashlib.sha1(url.encode()).hexdigest()
    f = CACHE / h
    seen[url] = f.exists()
    if not url.endswith(".proto") or not f.exists():
        return
    text = f.read_text(errors="replace")
    for m in re.finditer(r'EXTERNPROTO\s+"([^"]+)"', text):
        visit(resolve(m.group(1), url))
    for block in re.finditer(r"url\s*\[([^\]]*)\]|url\s+\"([^\"]+)\"", text):
        for u in re.findall(r'"([^"]+)"', block.group(0)):
            if not u.startswith("%"):
                visit(resolve(u, url))


visit(sys.argv[1])
for url, present in seen.items():
    print(hashlib.sha1(url.encode()).hexdigest(), "present" if present else "MISSING", url)

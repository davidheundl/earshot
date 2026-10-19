"""Download EfficientAT's mn10_as AudioSet weights (MIT, ~20 MB) into data/models/.

    uv run --group ml python tools/fetch_efficientat.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt"
OUT = Path("data/models") / URL.rsplit("/", 1)[1]


def main() -> None:
    if OUT.exists():
        print(f"{OUT} already there")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.rename(OUT)
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())

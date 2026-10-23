"""Download Amazon Berkeley Objects (ABO) metadata from the public S3 bucket (CC BY 4.0).

    uv run python data/scripts/fetch_abo.py

Writes data/raw/abo/{listings_*.json.gz, images.csv.gz}. Images themselves are fetched
per product by build_abo_subset.py, only for the subset we keep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from renopilot import config  # noqa: E402

BUCKET = "https://amazon-berkeley-objects.s3.amazonaws.com"
# listing shards are numbered in hex: listings_0 ... listings_f
FILES = [f"listings/metadata/listings_{i:x}.json.gz" for i in range(16)] + ["images/metadata/images.csv.gz"]


def main() -> None:
    out = config.RAW / "abo"
    out.mkdir(parents=True, exist_ok=True)
    for key in FILES:
        dest = out / Path(key).name
        if dest.exists() and dest.stat().st_size > 0:
            print("have", dest.name)
            continue
        with requests.get(f"{BUCKET}/{key}", stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
        print("got ", dest.name, dest.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()

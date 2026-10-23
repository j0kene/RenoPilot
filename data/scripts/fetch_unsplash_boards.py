"""Build the style-eval mood boards from Unsplash (official API, Unsplash licence).

    set UNSPLASH_ACCESS_KEY in .env  (free: https://unsplash.com/oauth/applications, "demo" tier)
    uv run python data/scripts/fetch_unsplash_boards.py [--per-style 2] [--images 4]

Rate limit: the demo tier allows 50 API calls/hour. One search per style plus one download
ping per photo used = 14 + 112 calls for 28 boards, so a full run takes about 3 hours; it
throttles itself and resumes (cached searches, finished boards skipped).

Each board = 4 interior photos from one style query, from different photographers.
Writes data/boards/<board_id>/{01.jpg..., meta.json} with photographer attribution and the
query used (the query is NOT a label: the humans label what they see), plus an empty
labelling sheet per labeller in data/labels/. Boards are a small private eval set and are
not redistributed (proposal: datasets section). Follows the API guidelines: attribution
kept, and the download endpoint is pinged for each photo used.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from renopilot import config  # noqa: E402
from renopilot.vocab import STYLE_TAGS  # noqa: E402

API = "https://api.unsplash.com"
MAX_PER_HOUR = 45            # Unsplash demo apps: 50 requests/hour. Image CDN downloads do not count.
QUERIES = {                  # one search per style; its results are split into boards of --images photos
    "japandi": "japandi interior", "scandinavian": "scandinavian living room",
    "muji": "muji interior", "minimalist": "minimalist interior",
    "modern_contemporary": "modern contemporary living room", "modern_luxe": "luxury modern interior marble",
    "industrial": "industrial loft interior", "mid_century_modern": "mid century modern interior",
    "wabi_sabi": "wabi sabi interior", "bohemian": "bohemian interior",
    "coastal": "coastal interior", "tropical_resort": "tropical resort interior rattan",
    "vintage_retro": "vintage retro interior", "classic_traditional": "classic traditional interior",
}
LABEL_COLUMNS = ["board_id", "style_tags", "palette", "materials", "item_types", "notes"]


class Limiter:
    """Client-side: at most MAX_PER_HOUR API calls in any rolling hour; also obeys the
    X-Ratelimit-Remaining header if Unsplash says we are out."""

    def __init__(self, per_hour: int = MAX_PER_HOUR):
        self.per_hour, self.calls = per_hour, []

    def wait(self) -> None:
        now = time.time()
        self.calls = [t for t in self.calls if now - t < 3600]
        if len(self.calls) >= self.per_hour:
            pause = 3600 - (now - self.calls[0]) + 5
            print(f"  rate limit: sleeping {pause / 60:.0f} min", flush=True)
            time.sleep(pause)
        self.calls.append(time.time())

    def get(self, url: str, headers: dict, **kw) -> requests.Response:
        self.wait()
        r = requests.get(url, headers=headers, timeout=60, **kw)
        if r.status_code == 403 or r.headers.get("X-Ratelimit-Remaining") == "0":
            print("  Unsplash says the hourly quota is used up: sleeping 61 min", flush=True)
            time.sleep(61 * 60)
        return r


def write_label_sheets(board_ids: list[str]) -> None:
    """Create the two sheets, or append rows for new boards. Existing labels are never touched."""
    labels = config.DATA / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for who in ("A", "B"):
        f = labels / f"style_labels_{who}.csv"
        have = {r["board_id"] for r in csv.DictReader(f.open(encoding="utf-8"))} if f.exists() else set()
        new = [b for b in sorted(board_ids) if b not in have]
        with f.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, LABEL_COLUMNS)
            if not have:
                w.writeheader()
            w.writerows({"board_id": b} for b in new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-style", type=int, default=2, help="boards per style")
    ap.add_argument("--images", type=int, default=4, help="photos per board")
    args = ap.parse_args()
    key = os.environ.get("UNSPLASH_ACCESS_KEY")
    if not key:
        sys.exit("Set UNSPLASH_ACCESS_KEY in .env (free key from https://unsplash.com/oauth/applications)")
    headers = {"Authorization": f"Client-ID {key}", "Accept-Version": "v1"}
    assert set(QUERIES) <= set(STYLE_TAGS)
    lim = Limiter()
    cache = config.RAW / "unsplash"
    cache.mkdir(parents=True, exist_ok=True)

    board_ids = []
    for style, q in QUERIES.items():
        ids = [f"{style}_{i + 1:02d}" for i in range(args.per_style)]
        if all((config.BOARDS / b / "meta.json").exists() for b in ids):
            board_ids += ids
            continue
        hit = cache / f"{style}.json"                      # cached search: a resume costs no search call
        if not hit.exists():
            r = lim.get(f"{API}/search/photos", headers, params={"query": q, "per_page": 30,
                                                                  "orientation": "landscape", "content_filter": "high"})
            r.raise_for_status()
            hit.write_text(r.text, encoding="utf-8")
        seen, photos = set(), []
        for ph in json.loads(hit.read_text(encoding="utf-8"))["results"]:
            if ph["user"]["id"] not in seen:               # one photo per photographer
                seen.add(ph["user"]["id"])
                photos.append(ph)
        for bi, board_id in enumerate(ids):
            chunk = photos[bi * args.images:(bi + 1) * args.images]
            out = config.BOARDS / board_id
            if (out / "meta.json").exists() or len(chunk) < config.MIN_BOARD_IMAGES:
                board_ids += [board_id] if (out / "meta.json").exists() else []
                continue
            out.mkdir(parents=True, exist_ok=True)
            meta = {"board_id": board_id, "query": q, "source": "unsplash", "photos": []}
            for i, ph in enumerate(chunk, 1):
                img = requests.get(ph["urls"]["regular"], timeout=60)          # CDN: not rate limited
                img.raise_for_status()
                (out / f"{i:02d}.jpg").write_bytes(img.content)
                lim.get(ph["links"]["download_location"], headers)            # API guideline: counts
                meta["photos"].append({"file": f"{i:02d}.jpg", "id": ph["id"], "photographer": ph["user"]["name"],
                                       "profile": ph["user"]["links"]["html"], "page": ph["links"]["html"]})
            (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            board_ids.append(board_id)
            write_label_sheets(board_ids)
            print(f"{board_id}: {len(chunk)} photos  ({q})", flush=True)

    write_label_sheets(board_ids)
    print(f"{len(board_ids)} boards; label sheets in data/labels/ (see data/labels/LABELLING.md)")


if __name__ == "__main__":
    main()

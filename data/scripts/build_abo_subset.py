"""Pick the ABO furniture subset and fetch its images.

    uv run python data/scripts/build_abo_subset.py [--per-type 120] [--workers 16]

Keeps listings that (1) have an English title, (2) map to one of our loose-furniture
item types by title keywords, (3) are not accessories (swatches, covers, parts), and
(4) have a main image AND at least one alternate image. The alternate images are what
the retrieval eval queries with, so the index (main images) never contains the query.

Writes data/index/abo_products.jsonl (one Product per line, plus alt_image_paths).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from renopilot import config  # noqa: E402
from renopilot.catalog.taxonomy import EXCLUDE, classify  # noqa: E402

BUCKET = "https://amazon-berkeley-objects.s3.amazonaws.com/images/small"
FURNITURE_TYPES = {
    "SOFA", "CHAIR", "TABLE", "DESK", "BED", "LAMP", "LIGHT_FIXTURE", "RUG", "OTTOMAN",
    "STOOL_SEATING", "SHELF", "BOOKCASE", "CABINET", "DRESSER", "MIRROR", "BENCH",
    "HOME_FURNITURE_AND_DECOR", "HOME", "NIGHTSTAND", "FURNITURE",
}
def en(values: list[dict] | None) -> str:
    """First English value of a multilingual ABO field."""
    for v in values or []:
        if str(v.get("language_tag", "en")).startswith("en"):
            return str(v.get("value", "")).strip()
    return ""


def dims_of(d: dict) -> str:
    dim = (d.get("item_dimensions") or {})
    parts = []
    for k in ("width", "depth", "height"):
        v = dim.get(k, {}).get("normalized_value") or dim.get(k, {})
        if v.get("value"):
            parts.append(f"{k[0].upper()}{round(float(v['value']), 1)}{v.get('unit', '')[:2]}")
    return " ".join(parts)


def family_key(title: str, colour: str) -> str:
    """Colour variants of one model collapse to one family (kept once)."""
    t = title.lower()
    if colour:
        t = t.replace(colour.lower(), "")
    t = re.sub(r"[^a-z ]", " ", t)
    return " ".join(t.split()[:8])


def load_candidates() -> list[dict]:
    out, seen = [], set()
    for f in sorted((config.RAW / "abo").glob("listings_*.json.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                ptype = (d.get("product_type") or [{}])[0].get("value")
                if ptype not in FURNITURE_TYPES:
                    continue
                title = en(d.get("item_name"))
                if not title or EXCLUDE.search(title):
                    continue
                item_type = classify(title)
                if not item_type or not d.get("main_image_id") or not d.get("other_image_id"):
                    continue
                colour = en(d.get("color"))
                fam = family_key(title, colour)
                if fam in seen:
                    continue
                seen.add(fam)
                out.append({
                    "id": f"abo:{d['item_id']}", "source": "abo", "item_type": item_type,
                    "title": title[:200], "colour": colour, "material": en(d.get("material")),
                    "style": en(d.get("style")), "dims": dims_of(d),
                    "main_image_id": d["main_image_id"], "other_image_ids": d["other_image_id"][:3],
                    "abo_product_type": ptype,
                })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=120)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cands = load_candidates()
    print(f"{len(cands)} candidate families:", Counter(c["item_type"] for c in cands).most_common())

    rng = random.Random(621)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_type[c["item_type"]].append(c)
    keep = []
    for t, items in sorted(by_type.items()):
        rng.shuffle(items)
        keep += items[: args.per_type]

    paths = {}
    with gzip.open(config.RAW / "abo" / "images.csv.gz", "rt", encoding="utf-8") as fh:
        wanted = {c["main_image_id"] for c in keep} | {i for c in keep for i in c["other_image_ids"][:2]}
        for row in csv.DictReader(fh):
            if row["image_id"] in wanted:
                paths[row["image_id"]] = row["path"]

    img_dir = config.IMAGES / "abo"
    img_dir.mkdir(parents=True, exist_ok=True)

    def fetch(image_id: str) -> str | None:
        if image_id not in paths:
            return None
        dest = img_dir / f"{image_id}.jpg"
        if not dest.exists():
            try:
                r = requests.get(f"{BUCKET}/{paths[image_id]}", timeout=60)
                r.raise_for_status()
                dest.write_bytes(r.content)
            except requests.RequestException:
                return None
        return str(dest.relative_to(config.ROOT)).replace("\\", "/")

    jobs = sorted({c["main_image_id"] for c in keep} | {i for c in keep for i in c["other_image_ids"][:2]})
    with ThreadPoolExecutor(args.workers) as pool:
        local = dict(zip(jobs, pool.map(fetch, jobs)))

    today = date.today().isoformat()
    config.INDEX.mkdir(parents=True, exist_ok=True)
    n = 0
    with (config.INDEX / "abo_products.jsonl").open("w", encoding="utf-8") as out:
        for c in keep:
            main = local.get(c["main_image_id"])
            alts = [p for p in (local.get(i) for i in c["other_image_ids"][:2]) if p]
            if not main or not alts:
                continue
            out.write(json.dumps({
                "id": c["id"], "source": "abo", "item_type": c["item_type"], "title": c["title"],
                "colour": c["colour"], "material": c["material"], "dims": c["dims"],
                "price": None, "currency": None, "price_sgd": None,
                "url": f"https://www.amazon.com/dp/{c['id'].split(':')[1]}",
                "image_path": main, "alt_image_paths": alts, "retrieved_at": today,
                "style": c["style"],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} products to {config.INDEX / 'abo_products.jsonl'}")


if __name__ == "__main__":
    main()

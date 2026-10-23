"""Collect the Singapore price catalog (dated snapshot) from IKEA SG and Castlery.

    uv run python data/scripts/collect_sg_catalog.py [--per-type 8]

  IKEA SG   the public search API behind ikea.com/sg (robots.txt on that host: Allow /).
            One query per item type; SGD price, dims, colour, image and product URL.
  Castlery  category listing pages (robots.txt allows them); the JSON-LD ItemList on each
            page has name, URL, image and SGD price.
  Taobao    behind a login, so it is not scraped: add rows to data/taobao_manual.csv by hand
            (same columns, currency CNY) and they are merged here.

Polite by construction: one request at a time with a pause, a handful of pages per site.
Images are downloaded only to compute embeddings locally; the catalog links to the
retailer's image, it does not redistribute it.

Writes data/sg_catalog.csv (committed: the frozen snapshot) and caches images in data/images/sg/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from renopilot import config  # noqa: E402
from renopilot.catalog.taxonomy import classify  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RenoPilot-CS621-student-project"}
PAUSE_S = 1.5
CNY_TO_SGD = 0.185     # indicative snapshot rate for the budget filter only; Pricing owns real FX
COLUMNS = ["id", "source", "item_type", "title", "colour", "material", "dims", "price",
           "currency", "price_sgd", "url", "image_url", "image_path", "retrieved_at"]

IKEA_QUERIES = {
    "sofa": "sofa", "armchair": "armchair", "coffee_table": "coffee table",
    "side_table": "side table", "tv_console": "tv bench", "dining_table": "dining table",
    "dining_chair": "dining chair", "bar_stool": "bar stool", "bench": "bench",
    "ottoman": "pouffe", "bed_frame": "bed frame", "bedside_table": "bedside table",
    "dresser": "chest of drawers", "wardrobe": "wardrobe", "bookshelf": "bookcase",
    "desk": "desk", "office_chair": "office chair", "rug": "rug",
    "floor_lamp": "floor lamp", "table_lamp": "table lamp", "pendant_light": "pendant lamp",
    "mirror": "mirror",
}
# IKEA returns related types too (a sofa query also lists sofa covers); keep a result only
# if its typeName matches the item type.
IKEA_TYPE_OK = {
    "sofa": r"sofa(?! cover)|sofa-bed|corner sofa", "armchair": r"armchair|easy chair",
    "coffee_table": r"coffee table", "side_table": r"side table|nest of tables",
    "tv_console": r"tv bench|tv unit|tv storage", "dining_table": r"^table|dining table|extendable table",
    "dining_chair": r"^chair|dining chair", "bar_stool": r"bar stool", "bench": r"^bench",
    "ottoman": r"pouffe|footstool", "bed_frame": r"bed frame|day-bed|bed with",
    "bedside_table": r"bedside table", "dresser": r"chest of \d+ drawers|chest of drawers",
    "wardrobe": r"wardrobe", "bookshelf": r"bookcase|shelving unit",
    "desk": r"^desk|desk$|desk with", "office_chair": r"office chair|swivel chair",
    "rug": r"rug", "floor_lamp": r"floor lamp|floor uplighter", "table_lamp": r"table lamp",
    "pendant_light": r"pendant lamp", "mirror": r"^mirror",
}

CASTLERY_PAGES = {
    "sofa": ["sofas/all-sofas"], "armchair": ["chairs/armchairs", "chairs/all-armchairs"],
    "coffee_table": ["tables/coffee-tables", "tables/all-coffee-tables"],
    "side_table": ["tables/side-tables", "tables/side-end-tables"],
    "tv_console": ["storage/tv-units", "storage/tv-consoles", "storage/all-tv-units"],
    "dining_table": ["tables/dining-tables", "tables/all-dining-tables"],
    "dining_chair": ["chairs/dining-chairs", "chairs/all-dining-chairs"],
    "bar_stool": ["chairs/bar-stools", "chairs/bar-counter-stools"],
    "bench": ["chairs/benches", "chairs/dining-benches"],
    "ottoman": ["chairs/ottomans", "chairs/ottomans-footstools"],
    "bed_frame": ["beds/all-beds", "beds/bed-frames"],
    "bedside_table": ["storage/bedside-tables", "beds/bedside-tables"],
    "dresser": ["storage/dressers", "storage/chest-of-drawers", "storage/dressers-chests"],
    "bookshelf": ["storage/bookshelves", "storage/shelves-bookcases", "storage/bookcases"],
    "desk": ["tables/desks", "tables/study-desks", "home-office/desks"],
    "office_chair": ["chairs/office-chairs", "home-office/office-chairs"],
    "rug": ["decor/rugs", "accessories/rugs", "rugs/all-rugs"],
    "floor_lamp": ["lighting/floor-lamps", "decor/floor-lamps"],
    "table_lamp": ["lighting/table-lamps", "decor/table-lamps"],
    "pendant_light": ["lighting/pendant-lights", "lighting/ceiling-lights"],
    "mirror": ["decor/mirrors", "accessories/mirrors"],
}


def get(url: str, **kw) -> requests.Response | None:
    time.sleep(PAUSE_S)
    try:
        r = requests.get(url, headers=UA, timeout=45, **kw)
        return r if r.ok else None
    except requests.RequestException:
        return None


def ikea(per_type: int, today: str) -> list[dict]:
    rows = []
    for item_type, q in IKEA_QUERIES.items():
        r = get(f"https://sik.search.blue.cdtapps.com/sg/en/search-result-page"
                f"?q={quote(q)}&size=48&types=PRODUCT")
        if not r:
            print("  ikea: no response for", q)
            continue
        items = r.json().get("searchResultPage", {}).get("products", {}).get("main", {}).get("items", [])
        ok = re.compile(IKEA_TYPE_OK[item_type], re.I)
        kept = 0
        for it in items:
            p = it.get("product") or {}
            price = (p.get("salesPrice") or {}).get("numeral")
            if not price or not ok.search(p.get("typeName", "")) or not p.get("mainImageUrl"):
                continue
            colours = ", ".join(c.get("name", "") for c in p.get("colors", [])[:2])
            rows.append({
                "id": f"ikea_sg:{p['itemNo']}", "source": "ikea_sg", "item_type": item_type,
                "title": f"{p.get('name', '')} {p.get('typeName', '')}, {p.get('mainImageAlt', '')}".strip(", "),
                "colour": colours, "material": "", "dims": p.get("itemMeasureReferenceText", ""),
                "price": price, "currency": "SGD", "price_sgd": price, "url": p.get("pipUrl", ""),
                "image_url": p["mainImageUrl"], "retrieved_at": today,
            })
            kept += 1
            if kept >= per_type:
                break
        print(f"  ikea {item_type:<14} {kept}")
    return rows


def castlery(per_type: int, today: str) -> list[dict]:
    rows = []
    for item_type, paths in CASTLERY_PAGES.items():
        kept = 0
        for path in paths:
            r = get(f"https://www.castlery.com/sg/{path}")
            if not r:
                continue
            for block in re.findall(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', r.text, re.S):
                try:
                    d = json.loads(block)
                except json.JSONDecodeError:
                    continue
                if not isinstance(d, dict) or d.get("@type") != "ItemList":
                    continue
                for el in d.get("itemListElement", []):
                    p = el.get("item") or {}
                    offer = p.get("offers") or {}
                    if not p.get("name") or not offer.get("price") or not p.get("image"):
                        continue
                    slug = p["url"].rstrip("/").split("/")[-1]
                    rows.append({
                        "id": f"castlery:{slug}", "source": "castlery", "item_type": item_type,
                        "title": p["name"], "colour": "", "material": "", "dims": "",
                        "price": float(offer["price"]), "currency": offer.get("priceCurrency", "SGD"),
                        "price_sgd": float(offer["price"]), "url": p["url"], "image_url": p["image"],
                        "retrieved_at": today,
                    })
                    kept += 1
                    if kept >= per_type:
                        break
            if kept:
                break
        print(f"  castlery {item_type:<14} {kept}" + ("" if kept else "  (no page matched)"))
    return rows


def taobao_manual() -> list[dict]:
    f = config.DATA / "taobao_manual.csv"
    if not f.exists():
        with f.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, COLUMNS).writeheader()
        return []
    rows = []
    for r in csv.DictReader(f.open(encoding="utf-8")):
        if not r.get("title"):
            continue
        r["source"] = "taobao"
        if r.get("currency", "CNY") == "CNY" and r.get("price"):
            r["price_sgd"] = round(float(r["price"]) * CNY_TO_SGD, 2)
        rows.append(r)
    return rows


def retype(rows: list[dict]) -> int:
    """Re-type each row from its own title (categories leak: a lamp page lists both kinds).
    IKEA titles are "NAME type, image alt text": only the part before the comma is used."""
    n = 0
    for r in rows:
        head = r["title"].split(",")[0] if r["source"] == "ikea_sg" else r["title"]
        t = classify(head)
        if t and t != r["item_type"]:
            r["item_type"], n = t, n + 1
    return n


def cache_image(row: dict) -> str:
    url = row["image_url"]
    ext = ".png" if ".png" in url.lower() else ".jpg"
    dest = config.IMAGES / "sg" / (hashlib.sha1(url.encode()).hexdigest()[:16] + ext)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        r = get(url)
        if not r:
            return ""
        dest.write_bytes(r.content)
    return str(dest.relative_to(config.ROOT)).replace("\\", "/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--retype-only", action="store_true", help="re-type the existing CSV, no fetching")
    args = ap.parse_args()
    today = date.today().isoformat()
    out_csv = config.DATA / "sg_catalog.csv"
    if args.retype_only:
        rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
        print("retyped", retype(rows), "rows")
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return

    print("IKEA SG"); rows = ikea(args.per_type, today)
    print("Castlery"); rows += castlery(args.per_type, today)
    rows += taobao_manual()

    print("retyped", retype(rows), "rows from their titles")
    seen, out = set(), []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        r["image_path"] = cache_image(r) if r.get("image_url") else ""
        out.append(r)

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    by_src = {}
    for r in out:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"wrote {len(out)} rows to data/sg_catalog.csv  {by_src}")


if __name__ == "__main__":
    main()

"""Load ABO + SG catalog rows into SQLite and embed every product's main image.

    uv run python data/scripts/build_index.py [--embed ViT-B-16-SigLIP:webli]

Run once per embedding model (the retrieval eval compares models). Embeddings land in
data/index/<model-slug>/{image_emb.npy, ids.txt}; ids.txt fixes the row order.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from renopilot import config  # noqa: E402
from renopilot.catalog import store  # noqa: E402
from renopilot.catalog.embed import Embedder, slug  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default=config.EMBED_MODEL)
    ap.add_argument("--skip-load", action="store_true", help="only (re)embed what is in SQLite")
    args = ap.parse_args()

    if not args.skip_load:
        store.DB_PATH.unlink(missing_ok=True)          # a full load is a fresh snapshot: no stale rows
        rows = []
        abo = config.INDEX / "abo_products.jsonl"
        if abo.exists():
            rows += [json.loads(line) for line in abo.open(encoding="utf-8")]
        sg = config.DATA / "sg_catalog.csv"
        if sg.exists():
            rows += [r for r in csv.DictReader(sg.open(encoding="utf-8")) if r.get("image_path")]
        print("upserted", store.upsert(rows), "products")

    products, _ = store.load_all()
    products = [p for p in products if p.image_path and (config.ROOT / p.image_path).exists()]
    print(f"embedding {len(products)} images with {args.embed} ...")
    t0 = time.perf_counter()
    emb = Embedder(args.embed).images([p.image_path for p in products])
    out = config.INDEX / slug(args.embed)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "image_emb.npy", emb.astype(np.float32))
    (out / "ids.txt").write_text("\n".join(p.id for p in products), encoding="utf-8")
    print(f"done in {time.perf_counter() - t0:.0f}s -> {out}  shape={emb.shape}")


if __name__ == "__main__":
    main()

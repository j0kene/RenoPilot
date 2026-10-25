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
        # overseas rows (no images: priced and matched by text, never embedded)
        overseas = config.DATA / "taobao_manual.csv"
        if overseas.exists():
            for r in csv.DictReader(overseas.open(encoding="utf-8")):
                if r.get("title") and r.get("price"):
                    r["synthetic"] = str(r.get("synthetic", "")).strip().lower() in ("true", "1", "yes")
                    rows.append(r)
        print("upserted", store.upsert(rows), "products")

    products, _ = store.load_all()
    products = [p for p in products if p.image_path and (config.ROOT / p.image_path).exists()]
    out = config.INDEX / slug(args.embed)
    out.mkdir(parents=True, exist_ok=True)
    # reuse embeddings already computed for the same id: adding rows must not re-embed everything
    cached = {}
    if (out / "ids.txt").exists() and (out / "image_emb.npy").exists():
        old_ids = (out / "ids.txt").read_text(encoding="utf-8").split("\n")
        old_emb = np.load(out / "image_emb.npy")
        cached = {pid: old_emb[i] for i, pid in enumerate(old_ids) if i < len(old_emb)}
    todo = [p for p in products if p.id not in cached]
    print(f"{len(products)} products with images: {len(products) - len(todo)} cached, embedding {len(todo)} "
          f"with {args.embed} ...")
    t0 = time.perf_counter()
    fresh = dict(zip([p.id for p in todo], Embedder(args.embed).images([p.image_path for p in todo]))) if todo else {}
    emb = np.stack([cached[p.id] if p.id in cached else fresh[p.id] for p in products])
    np.save(out / "image_emb.npy", emb.astype(np.float32))
    (out / "ids.txt").write_text("\n".join(p.id for p in products), encoding="utf-8")
    print(f"done in {time.perf_counter() - t0:.0f}s -> {out}  shape={emb.shape}")


if __name__ == "__main__":
    main()

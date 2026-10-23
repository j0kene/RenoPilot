"""The product catalog: one SQLite table for ABO and the SG retailers, flagged by `source`.

A few thousand rows fit in memory, so search loads them once and filters in numpy; SQLite
is the durable, inspectable copy (`sqlite3 data/index/catalog.sqlite`).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from renopilot import config
from renopilot.schemas import Product

DB_PATH = config.INDEX / "catalog.sqlite"
_FIELDS = list(Product.model_fields)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS products (
    {", ".join(f"{f} {'REAL' if f in ('price', 'price_sgd') else 'TEXT'}" for f in _FIELDS)},
    alt_image_paths TEXT DEFAULT '[]',
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_type ON products(item_type);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def upsert(rows: list[dict], path: Path = DB_PATH) -> int:
    """rows: Product fields plus optional alt_image_paths (list). Validated on the way in."""
    con = connect(path)
    n = 0
    for r in rows:
        p = Product.model_validate({k: v for k, v in r.items() if k in _FIELDS and v not in ("", None)})
        values = [getattr(p, f) for f in _FIELDS] + [json.dumps(r.get("alt_image_paths") or [])]
        con.execute(f"INSERT OR REPLACE INTO products ({', '.join(_FIELDS)}, alt_image_paths) "
                    f"VALUES ({', '.join('?' * (len(_FIELDS) + 1))})", values)
        n += 1
    con.commit()
    con.close()
    return n


def load_all(path: Path = DB_PATH) -> tuple[list[Product], dict[str, list[str]]]:
    """All products (ordered by id) and each product's alternate image paths."""
    con = connect(path)
    rows = con.execute("SELECT * FROM products ORDER BY id").fetchall()
    con.close()
    products = [Product.model_validate({f: r[f] for f in _FIELDS if r[f] is not None}) for r in rows]
    alts = {r["id"]: json.loads(r["alt_image_paths"] or "[]") for r in rows}
    return products, alts

"""Hybrid product search: three rankings fused by reciprocal rank, after hard filters.

  text_img   the item description, embedded as text, against product IMAGES (cross-modal)
  board_img  the board's images against product images (max over board images)
  bm25       the description's words against title / colour / material

Hard constraints (item type, source, price cap, excluded ids) are applied in code BEFORE
ranking: "a constraint in the prompt is a request; in code it is a rule". Style fit is
the soft part, and that is what the rankings decide.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from renopilot import config, trace
from renopilot.catalog import store
from renopilot.catalog.embed import get_embedder, slug
from renopilot.schemas import Product
from renopilot.tools.budget import within

ALL_SIGNALS = ("text_img", "board_img", "bm25")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))


@dataclass
class Hit:
    product: Product
    score: float
    ranks: dict[str, int] = field(default_factory=dict)   # signal -> 1-based rank (for traces/ablation)


class CatalogIndex:
    def __init__(self, products: list[Product], image_emb: np.ndarray, embed_spec: str,
                 alt_images: dict[str, list[str]] | None = None):
        assert len(products) == len(image_emb), "embeddings out of sync with catalog: rerun build_index"
        self.products = products
        self.image_emb = image_emb
        self.embed_spec = embed_spec
        self.alt_images = alt_images or {}
        self.pos = {p.id: i for i, p in enumerate(products)}
        self.item_type = np.array([p.item_type for p in products])
        self.source = np.array([p.source for p in products])
        self.price = np.array([np.nan if p.price_sgd is None else p.price_sgd for p in products])
        self.bm25 = BM25Okapi([_tokens(f"{p.title} {p.colour} {p.material} {p.item_type}") or ["_"]
                               for p in products])

    @classmethod
    def load(cls, embed_spec: str = config.EMBED_MODEL) -> "CatalogIndex":
        products, alts = store.load_all()
        d = config.INDEX / slug(embed_spec)
        ids = (d / "ids.txt").read_text(encoding="utf-8").split("\n")
        emb = np.load(d / "image_emb.npy")
        order = {pid: i for i, pid in enumerate(ids)}
        products = [p for p in products if p.id in order]
        emb = emb[[order[p.id] for p in products]]
        return cls(products, emb, embed_spec, alts)

    # ------------------------------------------------------------------ filters
    def mask(self, item_type: str | None = None, max_price: float | None = None,
             sources: tuple[str, ...] | None = None, exclude_ids: set[str] | frozenset = frozenset()) -> np.ndarray:
        m = np.ones(len(self.products), dtype=bool)
        if item_type:
            m &= self.item_type == item_type
        if sources:
            m &= np.isin(self.source, list(sources))
        if max_price is not None:
            m &= np.array([within(None if np.isnan(x) else float(x), max_price) for x in self.price])
        for pid in exclude_ids:
            if pid in self.pos:
                m[self.pos[pid]] = False
        return m

    # ------------------------------------------------------------------ search
    @trace.tool("catalog_search", inputs=("query", "item_type", "max_price", "sources", "exclude_ids", "signals"),
                output=lambda hits: {"n": len(hits), "top": [{"id": h.product.id, "score": round(h.score, 5),
                                                          "ranks": h.ranks} for h in hits[:10]]})
    def search(self, query: str, *, item_type: str | None = None, board_emb: np.ndarray | None = None,
               max_price: float | None = None, sources: tuple[str, ...] | None = None,
               exclude_ids: set[str] | frozenset = frozenset(), k: int = config.WIDE_K,
               signals: tuple[str, ...] = ALL_SIGNALS, weights: dict[str, float] | None = None) -> list[Hit]:
        m = self.mask(item_type, max_price, sources, exclude_ids)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            return []
        weights = weights or {}
        scores: dict[str, np.ndarray] = {}
        if "text_img" in signals and query:
            q = get_embedder(self.embed_spec).texts([query])[0]
            scores["text_img"] = self.image_emb[idx] @ q
        if "board_img" in signals and board_emb is not None and len(board_emb):
            scores["board_img"] = (self.image_emb[idx] @ board_emb.T).max(axis=1)
        if "bm25" in signals and query:
            scores["bm25"] = self.bm25.get_scores(_tokens(query))[idx]

        fused = np.zeros(len(idx))
        ranks: dict[str, np.ndarray] = {}
        for name, s in scores.items():
            r = np.empty(len(idx), dtype=int)
            r[np.argsort(-s, kind="stable")] = np.arange(1, len(idx) + 1)
            ranks[name] = r
            fused += weights.get(name, 1.0) / (config.RRF_K + r)
        top = np.argsort(-fused, kind="stable")[:k]
        return [Hit(self.products[idx[j]], float(fused[j]), {n: int(r[j]) for n, r in ranks.items()})
                for j in top]

    def count(self, **filters) -> int:
        return int(self.mask(**filters).sum())

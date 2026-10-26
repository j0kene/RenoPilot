"""HDB renovation rule cards (data/compliance/hdb_rules.jsonl): item type -> verdict, conditions,
and the HDB text it rests on. Each card's `source` is an id in data/compliance/sources.json
(written by data/scripts/fetch_hdb_sources.py), which gives its URL."""
from __future__ import annotations

import json
from functools import lru_cache

from renopilot import config

IMPLIED_ROOM = {"bathroom_finishes": "bathroom", "kitchen_cabinets": "kitchen"}


@lru_cache(maxsize=1)
def cards() -> list[dict]:
    d = config.DATA / "compliance"
    sources = json.loads((d / "sources.json").read_text(encoding="utf-8"))
    rows = [json.loads(ln) for ln in (d / "hdb_rules.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [r | {"url": sources[r["source"]]["url"]} for r in rows]


def rules_for(item_type: str, room: str | None = None) -> list[dict]:
    """Cards for this item type, room-specific first; a room-limited card needs that room.
    Empty for loose furniture; a finish/built-in without a card falls under rule("HDB-NOT-LISTED")."""
    room = room or IMPLIED_ROOM.get(item_type)
    out = [c for c in cards() if item_type in c["item_types"] and (not c["rooms"] or room in c["rooms"])]
    return sorted(out, key=lambda c: not c["rooms"])


def rule(rule_id: str) -> dict:
    return next(c for c in cards() if c["id"] == rule_id)

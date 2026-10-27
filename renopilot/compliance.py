"""HDB renovation rule cards (data/compliance/hdb_rules.jsonl): item type -> verdict, conditions,
and the HDB text it rests on. Each card's `source` is an id in data/compliance/sources.json
(written by data/scripts/fetch_hdb_sources.py), which gives its URL."""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache

from renopilot import config

IMPLIED_ROOM = {"bathroom_finishes": "bathroom", "kitchen_cabinets": "kitchen"}
EASIEST = ("allow", "allow_with_permit", "escalate", "block")
PLAIN = {"allow": "allowed, no permit", "allow_with_permit": "needs an HDB permit",
         "escalate": "check with HDB first", "block": "not allowed"}


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


def routes(item_type: str, room: str | None = None, completion_date: date | None = None) -> list[dict]:
    """The ways to do this work, one card each, easiest first (allow, permit, escalate, block).
    A room card replaces the general card for the same method (bathroom overlay over floor overlay);
    a known completion date drops the card for the other block age (`block_age`)."""
    found = rules_for(item_type, room)
    room_methods = {c["method"] for c in found if c["rooms"]}
    found = [c for c in found if c["rooms"] or c["method"] not in room_methods]
    if completion_date:
        age = "under_3y" if (date.today() - completion_date).days < 3 * 365.25 else "3y_plus"
        found = [c for c in found if c.get("block_age") in (None, age)]
    found.sort(key=lambda c: (not c["rooms"], len(c["item_types"])))   # the most specific card first
    return sorted(found, key=lambda c: EASIEST.index(c["decision"]))


def rule(rule_id: str) -> dict:
    return next(c for c in cards() if c["id"] == rule_id)

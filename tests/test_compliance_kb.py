"""HDB rule cards: every card is cited and grounded, every finish/built-in is covered."""
import re
from datetime import date

import pytest

from renopilot import config
from renopilot.compliance import cards, routes, rules_for
from renopilot.vocab import LOOSE_FURNITURE, ROOMS, WORK_TYPE_OF

RAW = config.DATA / "raw" / "hdb"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip().lower()


def test_every_card_is_cited():
    for c in cards():
        assert c["url"].startswith("https://"), c["id"]
        assert c["decision"] in ("allow", "allow_with_permit", "block", "escalate"), c["id"]
        assert set(c["item_types"]) <= set(WORK_TYPE_OF), c["id"]


@pytest.mark.skipif(not RAW.exists(), reason="run data/scripts/fetch_hdb_sources.py to check quotes")
def test_every_quote_is_in_its_source():
    for c in cards():
        text = _norm((RAW / f"{c['source']}.txt").read_text(encoding="utf-8"))
        missing = [p for p in re.split(r" \| | / ", c["quote"]) if _norm(p) not in text]
        assert not missing, (c["id"], missing)


def test_every_finish_and_built_in_has_a_card():
    for t, work in WORK_TYPE_OF.items():
        if work != "loose_furniture":
            assert rules_for(t), t


def test_loose_furniture_has_no_rules():
    assert not any(rules_for(t) for t in LOOSE_FURNITURE)


def test_rooms():
    assert rules_for("floor_tiles", room="bathroom")[0]["id"] == "HDB-BATH-3YR"
    assert "HDB-BATH-3YR" not in [c["id"] for c in rules_for("floor_tiles")]
    assert "HDB-BATH-3YR" in [c["id"] for c in rules_for("bathroom_finishes")]        # room implied
    assert "HDB-KITCHEN-CABINET" in [c["id"] for c in rules_for("kitchen_cabinets")]


def test_routes_easiest_first():
    assert [c["id"] for c in routes("floor_tiles", room="living")] == \
        ["HDB-FLOOR-OVERLAY", "HDB-FLOOR-REPLACE", "HDB-FLOOR-RAISE"]
    assert [c["decision"] for c in routes("partition_wall_change")] == ["allow", "allow_with_permit", "block"]


def test_bathroom_routes_follow_the_block_age():
    ids = lambda **kw: [c["id"] for c in routes("floor_tiles", room="bathroom", **kw)]   # noqa: E731
    assert ids()[0] == "HDB-BATH-3YR-OVERLAY" and "HDB-FLOOR-OVERLAY" not in ids()   # room card replaces the general one
    assert {"HDB-BATH-3YR", "HDB-BATH-FLOOR-REPLACE"} <= set(ids())                  # unknown age: both listed
    new, old = ids(completion_date=date.today()), ids(completion_date=date(2015, 1, 1))
    assert "HDB-BATH-3YR" in new and "HDB-BATH-FLOOR-REPLACE" not in new
    assert "HDB-BATH-FLOOR-REPLACE" in old and "HDB-BATH-3YR" not in old


def test_every_finish_and_built_in_has_a_legal_route_in_every_room():
    """Otherwise the guard blocks the item outright: it is shown, but the owner gets no way to do it."""
    for t, work in WORK_TYPE_OF.items():
        if work != "loose_furniture":
            for room in ROOMS:
                assert any(c["decision"] != "block" for c in routes(t, room=room)), (t, room)

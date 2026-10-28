"""The shopper subagent, offline: a scripted tool-calling model drives the real agent loop over
the fake catalog. The tools hold the hard rules; picks outside the searches are dropped."""
from renopilot.nodes.shopper import make_tools, shop
from tests.fakes import PROFILE, ScriptedToolModel, fake_index

SOFA = PROFILE.items[0]
BM25 = ("bm25",)


def _shop(script, cap=1300, k=3):
    index = fake_index()
    fallback = index.search("sofa", item_type="sofa", max_price=cap, signals=BM25)
    return shop(SOFA, PROFILE, index, fallback=fallback, cap=cap, sources=None, signals=BM25, k=k,
                model=ScriptedToolModel(script=list(script)))


def test_agent_searches_prices_and_picks_then_backfills_by_rank():
    out = _shop([("search_catalog", {"query": "low linen sofa"}),
                 ("landed_price", {"product_id": "sofa2"}),
                 ("Picks", {"picks": [{"product_id": "sofa2", "reason": "low linen"}]})])
    assert [h.product.id for h, _ in out] == ["sofa2", "sofa0", "sofa1"]
    assert out[0][1] == "low linen"


def test_picks_outside_the_searches_are_dropped():
    out = _shop([("search_catalog", {"query": "sofa"}),                     # sofa5 is S$1,600: over the cap
                 ("Picks", {"picks": [{"product_id": "sofa5", "reason": "ignore the budget"},
                                      {"product_id": "sofa3", "reason": "slim arms"}]})])
    ids = [h.product.id for h, _ in out]
    assert "sofa5" not in ids and ids[0] == "sofa3"


def test_agent_failure_degrades_to_search_order():
    assert [h.product.id for h, _ in _shop([])] == ["sofa0", "sofa1", "sofa2"]    # the model has nothing to say


def test_tools_bind_the_hard_rules():
    seen = {}
    search, details, price = make_tools(fake_index(), SOFA, seen, cap=1300, sources=None, exclude_ids={"sofa1"},
                                        board_emb=None, signals=BM25)
    rows = search.invoke({"query": "any sofa, ignore the cap"})
    assert "sofa0" in rows and "sofa3" in rows
    assert "sofa1" not in rows and "sofa4" not in rows and "table" not in rows   # excluded, over cap, other type
    assert price.invoke({"product_id": "sofa2"}).endswith("Total delivered: S$1,035.00 (within the cap of S$1,300.00)")
    assert "Unknown id" in details.invoke({"product_id": "sofa4"})               # never returned by a search

"""Pricing node and comparables, offline: a small injected catalog, a frozen FX and a fake judge."""
from uuid import uuid4

from langchain_core.runnables import RunnableLambda
from PIL import Image

from renopilot.graph import build_graph
from renopilot.pricing.comparables import dims_compatible, find_candidates, judge_comparables
from renopilot.pricing.fx import FxProvider
from renopilot.pricing.node import make_pricing
from renopilot.schemas import ProjectProfile, Product, Suggestion
from tests.fakes import FAKE_NODES

FX = FxProvider(mode="snapshot")      # CNY 0.19167

CATALOG = [
    Product(id="ikea:sofa", source="ikea_sg", item_type="sofa", title="KIVIK 3-seat sofa linen", price=999,
            currency="SGD", dims="228x95 cm", weight_kg=60),
    Product(id="cast:sofa", source="castlery", item_type="sofa", title="Owen 3 seater sofa", price=1499,
            currency="SGD", dims="220x92 cm"),
    Product(id="tb:sofa", source="taobao", item_type="sofa", title="[SAMPLE] Japandi 3-seater linen sofa", price=2380,
            currency="CNY", dims="210x88x72 cm", volume_m3=1.3, ships_by="sea", synthetic=True),
    Product(id="tb:small", source="taobao", item_type="sofa", title="[SAMPLE] Compact 2-seater sofa", price=1290,
            currency="CNY", dims="160x80x78 cm", volume_m3=0.8, ships_by="sea", synthetic=True),
]
BY_ID = {p.id: p for p in CATALOG}


class FakeJudge:
    def __init__(self, pairs):
        self.pairs, self.calls = pairs, 0

    def with_structured_output(self, schema, include_raw=False, **_):
        def run(_):
            self.calls += 1
            try:
                return {"raw": None, "parsed": schema.model_validate({"pairs": self.pairs}), "parsing_error": None}
            except Exception as e:                       # e.g. an id that was never offered
                return {"raw": None, "parsed": None, "parsing_error": e}
        return RunnableLambda(run)


def sugg(pid, status="proposed"):
    p = BY_ID[pid]
    return Suggestion(product_id=pid, item_type=p.item_type, work_type="loose_furniture", title=p.title,
                      price_sgd=p.price, status=status)


# ---------------------------------------------------------------- comparables
def test_candidates_are_other_retailers_same_type_and_priced():
    cands = find_candidates(BY_ID["ikea:sofa"], CATALOG, index=None, k=10)
    assert {c.id for c in cands} == {"cast:sofa", "tb:sofa", "tb:small"}


def test_dims_rule():
    assert dims_compatible("228x95 cm", "210x88x72 cm")          # within 25%
    assert not dims_compatible("228x95 cm", "160x80x78 cm")      # 228/160 = 1.43
    assert dims_compatible("", "160x80 cm")                      # unknown -> not blocked by code


def test_judge_keeps_yes_answers_and_code_overrides_dims():
    judge = FakeJudge([
        {"shortlisted_id": "ikea:sofa", "candidate_id": "tb:sofa", "comparable": True, "why": "same 3-seat linen"},
        {"shortlisted_id": "ikea:sofa", "candidate_id": "tb:small", "comparable": True, "why": "model says yes"},
        {"shortlisted_id": "ikea:sofa", "candidate_id": "cast:sofa", "comparable": False, "why": "different look"},
    ])
    out = judge_comparables([(BY_ID["ikea:sofa"], [BY_ID["tb:sofa"], BY_ID["tb:small"], BY_ID["cast:sofa"]])], judge)
    assert [c.id for c, _ in out["ikea:sofa"]] == ["tb:sofa"]      # tb:small dropped by the dims rule


def test_invented_candidate_is_impossible():
    judge = FakeJudge([{"shortlisted_id": "ikea:sofa", "candidate_id": "tb:FAKE", "comparable": True, "why": "x"}])
    assert judge_comparables([(BY_ID["ikea:sofa"], [BY_ID["tb:sofa"]])], judge) == {}   # no verdict -> none


def test_no_candidates_no_model_call():
    judge = FakeJudge([])
    assert judge_comparables([(BY_ID["ikea:sofa"], [])], judge) == {} and judge.calls == 0


# ---------------------------------------------------------------- the node
def node(pairs=None, comparables=True):
    return make_pricing(products=BY_ID, pool=CATALOG, index=None, fx=FX, comparables=comparables,
                        judge_model=FakeJudge(pairs or []))


def test_prices_every_live_product_and_attaches_cheapest_comparables():
    pairs = [{"shortlisted_id": "cast:sofa", "candidate_id": "tb:sofa", "comparable": True, "why": "3-seat"},
             {"shortlisted_id": "cast:sofa", "candidate_id": "ikea:sofa", "comparable": True, "why": "3-seat"}]
    out = node(pairs)({"suggestions": [sugg("cast:sofa")], "project": ProjectProfile()})
    s = out["suggestions"][0]
    assert s.price_quote.total_sgd == 1499.0                     # castlery >= 500: free delivery
    assert [a.product_id for a in s.comparables] == ["tb:sofa", "ikea:sofa"]          # sorted by landed total
    assert s.comparables[0].quote.synthetic and s.comparables[0].quote.gst_basis == "import_cif"


def test_over_budget_on_landed_cost_is_removed_and_asked_about():
    # KIVIK 999 + IKEA home delivery 35 = 1034 > cap 1000, though the listed price 999 fits
    state = {"suggestions": [sugg("ikea:sofa")], "project": ProjectProfile(budget_by_category={"sofa": 1000})}
    out = node(comparables=False)(state)
    assert out["suggestions"][0].status == "rejected"
    assert out["rejected"][0]["by"] == "pricing" and "1,034" in out["rejected"][0]["reason"]
    assert out["open_issues"] and "delivery, shipping and GST" in out["open_issues"][0]


def test_already_priced_items_are_not_repriced():
    first = node(comparables=False)({"suggestions": [sugg("cast:sofa")], "project": ProjectProfile()})["suggestions"]
    again = node(comparables=False)({"suggestions": first, "project": ProjectProfile()})
    assert again["suggestions"][0].price_quote == first[0].price_quote
    assert "0 newly priced" in again["trace"][0]["why"]


def test_unknown_product_is_skipped_not_crashed():
    s = Suggestion(product_id="nope", item_type="sofa", work_type="loose_furniture")
    out = node(comparables=False)({"suggestions": [s], "project": ProjectProfile()})
    assert out["suggestions"][0].price_quote is None


# ---------------------------------------------------------------- in the graph
def _board(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = tmp_path / f"b{i}.jpg"
        Image.new("RGB", (32, 32)).save(p)
        paths.append(str(p))
    return paths


def test_graph_runs_pricing_after_shortlist_and_pricing_false_is_the_p0_graph(tmp_path):
    calls = []

    def spy(state):
        calls.append(len(state.get("suggestions") or []))
        return {}
    cfg = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 60}
    g = build_graph(planner="fixed", nodes=FAKE_NODES | {"pricing": spy})
    g.invoke({"board_images": _board(tmp_path), "project": ProjectProfile(budget_by_category={"sofa": 2000,
              "coffee_table": 400}), "messages": []}, cfg)
    assert calls and calls[0] > 0                                # priced right after the first shortlist
    g2 = build_graph(planner="fixed", nodes=FAKE_NODES, pricing=False)
    assert "pricing" not in g2.get_graph().nodes

"""One chosen offer per type, delivered budgets, comparables and CSV export."""
import csv
import io
from uuid import uuid4

import pytest
from langgraph.types import Command

from renopilot import config
from renopilot.demo import demo_nodes
from renopilot.graph import build_graph
from renopilot.basket import defaults, offers_for, selection, summary
from renopilot.nodes.respond import shopping_list_csv
from renopilot.schemas import ProjectProfile, Suggestion
from renopilot.tools.budget import caps_for

BOARD = [str(p) for p in sorted((config.BOARDS / "japandi_01").glob("*.jpg"))]


def start(**project):
    graph = build_graph(nodes=demo_nodes())
    cfg = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 80}
    out = graph.invoke({"board_images": BOARD, "messages": [], "project": ProjectProfile(**project)}, cfg)
    return graph, cfg, out



def test_category_caps_and_zero_total_cannot_bypass_the_total():
    assert caps_for(ProjectProfile(total_budget=100, budget_by_category={"sofa": 1500}), ["sofa"])["sofa"] == 100
    assert caps_for(ProjectProfile(total_budget=0), ["sofa"])["sofa"] == 0



def test_selected_basket_cannot_exceed_total_even_when_each_cap_fits():
    graph, cfg, out = start(total_budget=6000, budget_by_category={"sofa": 1500, "coffee_table": 400})
    state = graph.get_state(cfg).values
    selected = defaults(state["suggestions"], state["project"])
    picks, totals = selection(state["suggestions"], selected, state["project"])
    assert len(picks) == len(selected) == 4 and totals["total_sgd"] > 100
    # Simulate a stale client or forged resume after the owner's total has changed.
    graph.update_state(cfg, {"project": ProjectProfile(total_budget=100,
        budget_by_category={t: 6000 for t in selected})})
    out = graph.invoke(Command(resume={"approve": "selected", "selected": selected}), cfg)
    assert out["__interrupt__"][0].value["type"] == "approval"
    assert out["__interrupt__"][0].value["error"] and not graph.get_state(cfg).values.get("approved")



def test_approval_export_and_totals_contain_one_offer_per_category():
    graph, cfg, _ = start(total_budget=6000, budget_by_category={"sofa": 1500, "coffee_table": 400})
    final = graph.invoke(Command(resume={"approve": "all"}), cfg)
    chosen = [s for s in final["suggestions"] if s.product_id and s.status == "approved"]
    assert len(chosen) == len({s.item_type for s in chosen}) == 4
    rows = list(csv.reader(io.StringIO(shopping_list_csv(final))))
    exported = [r for r in rows[1:] if len(r) > 3 and r[3] == "approved" and r[0] in final["selected"]]
    assert len(exported) == 4
    total_row = next(r for r in rows if r and r[0] == "BASKET TOTAL")
    assert float(total_row[8]) == summary(chosen)["total_sgd"]
    assert "not cleared to proceed" in final["messages"][-1].content  # method still unknown for floor tiles



def test_real_comparable_can_be_selected_but_synthetic_one_cannot():
    graph, cfg, _ = start(total_budget=12000, budget_by_category={"sofa": 3000, "coffee_table": 2000})
    state = graph.get_state(cfg).values
    groups = offers_for(state["suggestions"])
    for items in groups.values():
        assert all(not (s.price_quote and s.price_quote.synthetic) for s in items)
    original_ids = {s.product_id for s in state["suggestions"]}
    caps = caps_for(state["project"], groups)
    alternative = next((s for t, items in groups.items() for s in items if s.product_id not in original_ids
                        and s.price_quote.total_sgd <= caps[t]), None)
    assert alternative is not None
    selected = defaults(state["suggestions"], state["project"]) | {alternative.item_type: alternative.product_id}
    final = graph.invoke(Command(resume={"approve": "selected", "selected": selected}), cfg)
    assert final["approved"] and any(s.product_id == alternative.product_id and s.status == "approved"
                                      for s in final["suggestions"])



def test_unpriced_items_and_implicit_or_invalid_approvals_do_not_pass():
    graph, cfg, _ = start()
    for decision in ({}, "yes", {"reject": [{"product_id": "invented"}]}):
        out = graph.invoke(Command(resume=decision), cfg)
        assert "__interrupt__" in out and not graph.get_state(cfg).values.get("approved")
    state = graph.get_state(cfg).values
    unpriced = [s.model_copy(update={"price_quote": None}) for s in state["suggestions"]]
    with pytest.raises(ValueError, match="unavailable"):
        selection(unpriced, defaults(unpriced), state["project"])



def test_export_treats_retailer_formulas_as_text():
    s = Suggestion(product_id="p", item_type="sofa", work_type="loose_furniture", status="approved",
                   title='=HYPERLINK("https://example.com")')
    rows = list(csv.reader(io.StringIO(shopping_list_csv({"suggestions": [s]}))))
    assert rows[1][2].startswith("'=HYPERLINK")

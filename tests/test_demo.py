"""The offline demo world (renopilot/demo.py) runs the real graph end to end with no model and
no catalog: the Streamlit demo mode and evals/demo_trace.py depend on it."""
from uuid import uuid4

from langgraph.types import Command

from renopilot import config
from renopilot.demo import demo_nodes
from renopilot.graph import build_graph
from renopilot.schemas import ProjectProfile

BOARD = [str(p) for p in sorted((config.BOARDS / "japandi_01").glob("*.jpg"))]


def _start(**budget):
    g, cfg = build_graph(nodes=demo_nodes()), {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 80}
    return g, cfg, g.invoke({"board_images": BOARD, "messages": [], "project": ProjectProfile(**budget)}, cfg)


def test_reject_with_a_reason_becomes_a_refined_search_then_approve():
    g, cfg, out = _start(total_budget=6000, budget_by_category={"sofa": 1500, "coffee_table": 400})
    gate = out["__interrupt__"][0].value
    sofas = [c for c in gate["suggestions"] if c["item_type"] == "sofa"]
    assert sofas and all(c["verdict"] == "allow" for c in sofas)
    live = [s for s in g.get_state(cfg).values["suggestions"] if s.item_type == "sofa" and s.status != "rejected"]
    assert all(s.price_quote.total_sgd <= 1500 for s in live)                 # the cap holds on LANDED cost
    assert any(c["comparables"] for c in gate["suggestions"])                 # the Pricing agent compared offers
    out = g.invoke(Command(resume={"reject": [{"product_id": sofas[0]["product_id"], "reason": "too bulky"}]}), cfg)
    again = [c["product_id"] for c in out["__interrupt__"][0].value["suggestions"] if c["item_type"] == "sofa"]
    assert sofas[0]["product_id"] not in again
    state = g.get_state(cfg).values
    assert state["refinements"] == {"sofa": "too bulky"}                       # the owner's words reached the search
    assert g.invoke(Command(resume={"approve": "all"}), cfg)["approved"] is True


def test_tight_cap_asks_the_owner_and_a_raised_cap_continues():
    g, cfg, out = _start(budget_by_category={"sofa": 1500, "coffee_table": 50})
    gate = out["__interrupt__"][0].value
    assert gate["type"] == "question" and "coffee table" in gate["question"]
    out = g.invoke(Command(resume={"answer": "ok", "budget_by_category": {"coffee_table": 300}}), cfg)
    gate = out["__interrupt__"][0].value
    assert gate["type"] == "approval"
    assert len([c for c in gate["suggestions"] if c["item_type"] == "coffee_table"]) >= 3

"""The owner's details: form fields and a free-text brief, merged at intake, reaching the HDB guard."""
from datetime import date, timedelta
from uuid import uuid4

from langchain_core.messages import HumanMessage

from renopilot import config
from renopilot.demo import demo_nodes, parse_brief
from renopilot.graph import build_graph
from renopilot.nodes.intake import make_intake, merge
from renopilot.schemas import ProjectProfile

BOARD = [str(p) for p in sorted((config.BOARDS / "japandi_01").glob("*.jpg"))]
BRIEF = ("4-room BTO, keys in June 2024, sofa under $1,500 and coffee table below S$300, total budget $6000. "
         "We also want to redo the bathroom floor.")


# ---------------------------------------------------------------- merge rules
def test_form_fields_win_and_the_brief_fills_gaps():
    form = ProjectProfile(total_budget=8000, budget_by_category={"sofa": 2000})
    out = merge(form, parse_brief(BRIEF), BRIEF)
    assert out.total_budget == 8000                                   # form wins
    assert out.budget_by_category == {"sofa": 2000, "coffee_table": 300}   # brief adds a missing category only
    assert out.completion_date is None and out.flat_type == "4-room"   # free-text date is not confirmed
    assert out.notes == BRIEF                                         # the brief is kept for later steps


def test_no_parse_result_keeps_the_form_and_the_brief():
    form = ProjectProfile(total_budget=5000)
    out = merge(form, None, "hello")
    assert out.total_budget == 5000 and out.notes == "hello"


def test_brief_is_read_for_category_caps_even_with_flat_and_total_and_never_crashes_intake():
    calls = []

    def reader(text):
        calls.append(text)
        raise RuntimeError("model down")

    full = ProjectProfile(flat_type="4-room", completion_date=date(2024, 6, 1), total_budget=6000)
    make_intake(parse=reader)({"board_images": BOARD, "project": full, "messages": [HumanMessage(BRIEF)]})
    assert calls == [BRIEF]                                            # brief may still supply category caps
    out = make_intake(parse=reader)({"board_images": BOARD, "project": ProjectProfile(), "messages": [HumanMessage(BRIEF)]})
    assert calls and out["project"].notes == BRIEF                    # a failed parse is logged, not fatal


def test_brief_is_read_once_not_on_every_turn():
    calls = []
    reader = lambda t: calls.append(t) or ProjectProfile()           # noqa: E731
    intake = make_intake(parse=reader)
    first = intake({"board_images": BOARD, "project": ProjectProfile(), "messages": [HumanMessage(BRIEF)]})
    intake({"board_images": BOARD, "project": first["project"], "messages": [HumanMessage("ok")]})
    assert len(calls) == 1


# ---------------------------------------------------------------- demo end to end: the date reaches the guard
def _bathroom_routes(completed):
    g = build_graph(nodes=demo_nodes())
    cfg = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 80}
    form = ProjectProfile(total_budget=6000, completion_date=completed)
    out = g.invoke({"board_images": BOARD, "project": form, "messages": [HumanMessage("Japandi; redo the bathroom floor")]}, cfg)
    gate = out["__interrupt__"][0].value
    card = next(c for c in gate["suggestions"] if c["item_type"] == "bathroom_finishes")
    return {r["rule_id"]: r["decision"] for r in card["routes"]}


def test_new_block_bathroom_replacement_is_blocked():
    routes = _bathroom_routes(date.today() - timedelta(days=365))
    assert routes.get("HDB-BATH-3YR") == "block"
    assert "HDB-BATH-FLOOR-REPLACE" not in routes                     # the 3y+ route is dropped
    assert routes.get("HDB-BATH-3YR-OVERLAY") == "escalate"           # overlay facts still need confirmation


def test_older_block_bathroom_replacement_needs_a_permit():
    routes = _bathroom_routes(date.today() - timedelta(days=5 * 365))
    assert routes.get("HDB-BATH-FLOOR-REPLACE") == "allow_with_permit"
    assert "HDB-BATH-3YR" not in routes


def test_unknown_date_shows_both_block_age_routes():
    routes = _bathroom_routes(None)
    assert {"HDB-BATH-3YR", "HDB-BATH-FLOOR-REPLACE"} <= set(routes)

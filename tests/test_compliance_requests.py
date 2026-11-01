"""Requested methods and per-item rooms survive the complete compliance flow."""
from datetime import date
from uuid import uuid4

import pytest
from langgraph.types import Command

from renopilot import config
from renopilot.demo import demo_nodes
from renopilot.graph import build_graph
from renopilot.demo import demo_index
from renopilot.nodes.guard import compliance_guard, verdict_for
from renopilot.nodes.planner import make_planner
from renopilot.nodes.style import build_shortlist
from renopilot.schemas import ItemSpec, ProjectProfile, Suggestion
from tests.fakes import PROFILE, ScriptedPlanner

BOARD = [str(p) for p in sorted((config.BOARDS / "japandi_01").glob("*.jpg"))]


def start(**project):
    graph = build_graph(nodes=demo_nodes())
    cfg = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 80}
    out = graph.invoke({"board_images": BOARD, "messages": [], "project": ProjectProfile(**project)}, cfg)
    return graph, cfg, out



def test_mixed_board_keeps_two_tile_requests_and_their_own_rooms():
    profile = PROFILE.model_copy(update={"room": "mixed", "items": [
        ItemSpec(item_type="floor_tiles", room="living", work_method="overlay", description="Terrazzo tiles"),
        ItemSpec(item_type="floor_tiles", room="bathroom", work_method="replace", description="Replace bathroom tiles")]})
    suggestions, _ = build_shortlist(profile, ProjectProfile(), [], index=demo_index(), signals=("bm25",), use_rerank=False)
    assert len(suggestions) == 2 and len({s.key for s in suggestions}) == 2
    checked = compliance_guard({"style_profile": profile, "suggestions": suggestions,
        "project": ProjectProfile(completion_date=date(2025, 1, 1), as_of=date(2026, 10, 28))})["suggestions"]
    by_room = {s.room: s for s in checked}
    assert by_room["living"].verdict.rule_id == "HDB-FLOOR-OVERLAY"
    assert by_room["bathroom"].verdict.rule_id == "HDB-BATH-3YR"
    assert by_room["bathroom"].status == "rejected"



def test_destructive_text_cannot_be_relabelled_as_an_erection_or_loose_furniture():
    s = Suggestion(item_type="partition_wall_change", work_type="loose_furniture", room="living",
                   work_method="erect", title="Hack and remove a structural RC wall")
    v = verdict_for(s, ProjectProfile())
    assert v.decision == "block" and v.rule_id == "HDB-WALL-STRUCTURAL"



@pytest.mark.parametrize("item_type", ["floor_tiles", "vinyl_flooring", "timber_flooring", "bathroom_finishes"])
def test_specific_raising_rule_overrides_general_flooring_permission(item_type):
    s = Suggestion(item_type=item_type, work_type="finish", room="bathroom" if item_type == "bathroom_finishes" else "living",
                   title="Raise floor level by 100mm with concrete", work_method="raise")
    assert verdict_for(s, ProjectProfile()).decision == "block"



def test_unknown_item_room_does_not_inherit_the_boards_majority_room():
    s = Suggestion(item_type="floor_tiles", work_type="finish", title="Replace floor tiles", work_method="replace")
    profile = PROFILE.model_copy(update={"room": "living", "items": [ItemSpec(
        item_type="floor_tiles", description="Replace floor tiles", work_method="replace")]})
    checked = compliance_guard({"style_profile": profile, "suggestions": [s], "project": ProjectProfile()})
    assert checked["suggestions"][0].verdict.decision == "escalate"



def test_blocked_work_gets_a_separate_owner_chosen_and_rechecked_alternative():
    nodes = demo_nodes()
    original = nodes["style_extract"]

    def extract(state):
        result = original(state)
        result["style_profile"] = result["style_profile"].model_copy(update={"items": [
            ItemSpec(item_type="partition_wall_change", room="living", work_method="demolish", structural=True,
                     description="Remove a structural RC wall")]})
        return result

    graph = build_graph(nodes=nodes | {"style_extract": extract})
    cfg = {"configurable": {"thread_id": uuid4().hex}}
    out = graph.invoke({"board_images": BOARD, "messages": [], "project": ProjectProfile()}, cfg)
    blocked = out["__interrupt__"][0].value["blocked"][0]
    out = graph.invoke(Command(resume={"alternatives": {blocked["key"]: "HDB-WALL-ERECT"}}), cfg)
    gate = out["__interrupt__"][0].value
    assert gate["blocked"][0]["key"] == blocked["key"]
    assert len(gate["suggestions"]) == 1 and gate["suggestions"][0]["rule_id"] == "HDB-WALL-ERECT"
    assert gate["suggestions"][0]["key"] != blocked["key"]
    assert sum(d["node"] == "compliance_guard" for d in graph.get_state(cfg).values["trace"]) == 2



def test_unknown_bathroom_age_cannot_be_confirmed_by_picking_the_older_age_route():
    s = Suggestion(item_type="floor_tiles", work_type="finish", room="bathroom", work_method="replace")
    assert verdict_for(s, ProjectProfile()).decision == "escalate"
    v = verdict_for(s, ProjectProfile())
    assert all(o.decision not in ("allow", "allow_with_permit") for o in v.options if o.rule_id and "REPLACE" in o.rule_id)



def test_unspecified_wall_work_cannot_assert_non_structural_status_by_selecting_a_route():
    s = Suggestion(item_type="partition_wall_change", work_type="built_in", room="living", title="Change a wall")
    v = verdict_for(s, ProjectProfile())
    demolition = next(o for o in v.options if o.rule_id == "HDB-WALL-DEMOLISH")
    assert demolition.decision == "escalate"



WALL = ItemSpec(item_type="partition_wall_change", room="mixed", work_method="demolish", structural=True,
                description="Hack and remove a structural RC wall")


def test_the_guard_checks_requested_work_no_suggestion_carries():
    """Regression: the live planner left the wall out of its search, so the guard never saw it."""
    profile = PROFILE.model_copy(update={"items": [WALL, ItemSpec(item_type="sofa", description="Linen sofa")]})
    sofa = Suggestion(product_id="s", item_type="sofa", work_type="loose_furniture")
    checked = compliance_guard({"style_profile": profile, "suggestions": [sofa], "project": ProjectProfile()})
    wall = next(s for s in checked["suggestions"] if s.item_type == "partition_wall_change")
    assert wall.verdict.rule_id == "HDB-WALL-STRUCTURAL" and wall.status == "rejected"
    assert len(checked["suggestions"]) == 2


def test_the_guard_does_not_duplicate_work_a_suggestion_already_carries():
    tiles = ItemSpec(item_type="floor_tiles", room="living", description="Terrazzo tiles")
    profile = PROFILE.model_copy(update={"items": [tiles]})
    routed = Suggestion(item_type="floor_tiles", work_type="finish", room="living", work_method="replace")
    checked = compliance_guard({"style_profile": profile, "suggestions": [routed], "project": ProjectProfile()})
    assert len(checked["suggestions"]) == 1              # the owner's route changed its key, not the request


def test_a_narrowing_planner_still_gets_furniture_and_a_verdict_on_the_work():
    """The trace of the failed live runs, replayed: search what the brief named, then try to stop."""
    nodes = demo_nodes()
    original = nodes["style_extract"]

    def extract(state):
        result = original(state)
        result["style_profile"] = result["style_profile"].model_copy(update={"items": [
            WALL, ItemSpec(item_type="sofa", description="low linen sofa with light oak legs"),
            ItemSpec(item_type="coffee_table", description="round light oak coffee table")]})
        return result

    script = [{"action": "extract_style", "why": ""}, {"action": "shortlist", "item_types": ["sofa"], "why": "only this"},
              {"action": "finalize", "why": "eager"}, {"action": "finalize", "why": "still eager"}]
    graph = build_graph(nodes=nodes | {"style_extract": extract, "planner": make_planner("llm", ScriptedPlanner(script))})
    cfg = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 80}
    out = graph.invoke({"board_images": BOARD, "messages": [], "project": ProjectProfile(total_budget=6000)}, cfg)
    gate = out["__interrupt__"][0].value
    assert {c["item_type"] for c in gate["suggestions"]} == {"sofa", "coffee_table"}
    assert [b["rule_id"] for b in gate["blocked"]] == ["HDB-WALL-STRUCTURAL"]
    final = graph.invoke(Command(resume={"approve": "all"}), cfg)
    assert final["approved"] and "Not allowed under HDB rules" in final["messages"][-1].content


def test_owner_date_correction_rechecks_the_work_without_a_stale_block():
    graph, cfg, out = start(notes="Replace bathroom floor tiles", completion_date=date(2025, 1, 1), as_of=date(2026, 10, 28))
    assert any(b["rule_id"] == "HDB-BATH-3YR" for b in out["__interrupt__"][0].value["blocked"])
    out = graph.invoke(Command(resume={"completion_date": "2020-01-01"}), cfg)
    gate = out["__interrupt__"][0].value
    assert not any(b["rule_id"] == "HDB-BATH-3YR" for b in gate["blocked"])
    assert any(s["rule_id"] == "HDB-BATH-FLOOR-REPLACE" for s in gate["suggestions"])

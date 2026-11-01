"""The whole graph, offline: fake style nodes, the fixed planner, real intake / guard /
approval / respond. Checks the routes and the autonomy boundaries, not model quality."""
from uuid import uuid4

from langgraph.types import Command
from PIL import Image

from renopilot.graph import build_graph
from renopilot.schemas import ProjectProfile
from tests.fakes import FAKE_NODES


def _board(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = tmp_path / f"b{i}.jpg"
        Image.new("RGB", (32, 32), (200, 180, 150)).save(p)
        paths.append(str(p))
    return paths


def _start(tmp_path, n_images=3, **project):
    g = build_graph(planner="fixed", nodes=FAKE_NODES)
    cfg = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 60}
    out = g.invoke({"board_images": _board(tmp_path, n_images), "project": ProjectProfile(**project),
                    "messages": []}, cfg)
    return g, cfg, out


def test_happy_path_parks_at_approval_then_responds(tmp_path):
    g, cfg, out = _start(tmp_path, budget_by_category={"sofa": 2000, "coffee_table": 400})
    gate = out["__interrupt__"][0].value
    assert gate["type"] == "approval"
    assert {c["item_type"] for c in gate["suggestions"]} == {"sofa", "coffee_table", "floor_tiles"}
    # every suggestion reached the gate with a guard verdict
    assert all(c["verdict"] for c in gate["suggestions"])
    tiles = next(c for c in gate["suggestions"] if c["item_type"] == "floor_tiles")
    # Unknown method is not silently replaced with the easiest legal one.
    assert tiles["verdict"] == "escalate"
    assert [r["decision"] for r in tiles["routes"]] == ["escalate", "allow_with_permit", "block"]
    # the owner will hack out the old tiles instead; a blocked route (raising the floor) is never taken
    final = g.invoke(Command(resume={"approve": "all", "routes": {"floor_tiles": "HDB-FLOOR-REPLACE"}}), cfg)
    assert final["approved"] is True
    assert "Selected basket total" in final["messages"][-1].content
    assert sum(s.status == "approved" for s in final["suggestions"] if s.product_id) == 2
    tiles = next(s for s in final["suggestions"] if s.item_type == "floor_tiles")
    assert tiles.verdict.rule_id == "HDB-FLOOR-REPLACE" and tiles.verdict.decision == "allow_with_permit"
    assert tiles.status == "proposed"  # selected route still has technical conditions to verify
    assert len(tiles.verdict.options) == 3


def test_a_blocked_route_is_never_taken(tmp_path):
    g, cfg, _ = _start(tmp_path, budget_by_category={"sofa": 2000, "coffee_table": 400})
    final = g.invoke(Command(resume={"approve": "all", "routes": {"floor_tiles": "HDB-FLOOR-RAISE"}}), cfg)
    tiles = next(s for s in final["suggestions"] if s.item_type == "floor_tiles")
    assert tiles.verdict.decision == "escalate"
    assert "__interrupt__" in final and not final.get("approved")


def test_reject_triggers_a_new_search_that_excludes_the_rejected_item(tmp_path):
    g, cfg, out = _start(tmp_path, budget_by_category={"sofa": 2000, "coffee_table": 400})
    first = out["__interrupt__"][0].value["suggestions"]
    sofa = next(c for c in first if c["item_type"] == "sofa")
    out = g.invoke(Command(resume={"reject": [{"product_id": sofa["product_id"], "reason": "too bulky"}]}), cfg)
    second = out["__interrupt__"][0].value["suggestions"]
    assert sofa["product_id"] not in {c["product_id"] for c in second}
    assert any(c["item_type"] == "sofa" for c in second)
    actions = [t["action"] for t in g.get_state(cfg).values["trace"] if t["node"] == "planner"]
    assert actions == ["extract_style", "shortlist", "finalize", "shortlist", "finalize"]


def test_too_few_in_budget_asks_the_owner_and_budget_change_reshortlists(tmp_path):
    g, cfg, out = _start(tmp_path, budget_by_category={"sofa": 900, "coffee_table": 400})
    gate = out["__interrupt__"][0].value
    assert gate["type"] == "question" and "sofa" in gate["question"]
    out = g.invoke(Command(resume={"answer": "ok, up to 2000", "budget_by_category": {"sofa": 2000}}), cfg)
    gate = out["__interrupt__"][0].value
    assert gate["type"] == "approval"
    assert sum(c["item_type"] == "sofa" for c in gate["suggestions"]) >= 3
    assert g.get_state(cfg).values["project"].budget_by_category["sofa"] == 2000
    searches = [t["item_types"] for t in g.get_state(cfg).values["trace"] if t.get("action") == "shortlist"]
    assert searches[1:] == [["sofa"]]                    # only the type whose cap moved is searched again


def test_a_changed_total_searches_every_type_whose_share_moved(tmp_path):
    g, cfg, out = _start(tmp_path, total_budget=1200)
    assert out["__interrupt__"][0].value["type"] == "question"
    out = g.invoke(Command(resume={"answer": "more", "total_budget": 6000}), cfg)
    searches = [t["item_types"] for t in g.get_state(cfg).values["trace"] if t.get("action") == "shortlist"]
    assert searches[1:] == [["sofa", "coffee_table"]]


def test_owner_can_decline_to_raise_budget(tmp_path):
    g, cfg, out = _start(tmp_path, budget_by_category={"sofa": 900, "coffee_table": 400})
    out = g.invoke(Command(resume={"answer": "keep my budget"}), cfg)
    assert out["__interrupt__"][0].value["type"] == "approval"        # no loop back to the question


def test_too_few_images_asks_before_any_model_call(tmp_path):
    g, cfg, out = _start(tmp_path, n_images=2)
    gate = out["__interrupt__"][0].value
    assert gate["type"] == "question" and "at least 3" in gate["question"]
    assert g.get_state(cfg).values.get("style_profile") is None


def test_nothing_is_approved_without_the_owner(tmp_path):
    g, cfg, out = _start(tmp_path, budget_by_category={"sofa": 2000, "coffee_table": 400})
    state = g.get_state(cfg).values
    assert not state.get("approved")
    assert all(s.status == "proposed" for s in state["suggestions"])


def test_answering_without_adding_images_does_not_bypass_the_board_rule(tmp_path):
    """Regression: the LLM planner read a 2-image board after the owner just said "ok"."""
    g, cfg, out = _start(tmp_path, n_images=2)
    out = g.invoke(Command(resume={"answer": "ok"}), cfg)
    assert out["__interrupt__"][0].value["type"] == "question"          # asked again
    assert g.get_state(cfg).values.get("style_profile") is None
    out = g.invoke(Command(resume={"answer": "added one", "board_images": _board(tmp_path, 3)}), cfg)
    assert out["__interrupt__"][0].value["type"] == "approval"


def test_a_route_picked_at_approval_survives_a_rejection_round(tmp_path):
    g, cfg, out = _start(tmp_path, budget_by_category={"sofa": 2000, "coffee_table": 400})
    sofa = next(c for c in out["__interrupt__"][0].value["suggestions"] if c["item_type"] == "sofa")
    out = g.invoke(Command(resume={"reject": [{"product_id": sofa["product_id"], "reason": "too bulky"}],
                                   "routes": {"floor_tiles": "HDB-FLOOR-REPLACE"}}), cfg)
    tiles = next(c for c in out["__interrupt__"][0].value["suggestions"] if c["item_type"] == "floor_tiles")
    assert tiles["rule_id"] == "HDB-FLOOR-REPLACE"           # the guard's re-check keeps the owner's route


def test_a_blocked_finish_is_shown_not_dropped(tmp_path, monkeypatch):
    from renopilot.compliance import rule
    monkeypatch.setattr("renopilot.nodes.guard.routes", lambda *a, **k: [rule("HDB-FLOOR-RAISE")])
    g, cfg, out = _start(tmp_path, budget_by_category={"sofa": 2000, "coffee_table": 400})
    gate = out["__interrupt__"][0].value
    assert "floor_tiles" not in {c["item_type"] for c in gate["suggestions"]}
    assert [b["rule_id"] for b in gate["blocked"]] == ["HDB-FLOOR-RAISE"]
    final = g.invoke(Command(resume={"approve": "all"}), cfg)
    assert "Not allowed under HDB rules" in final["messages"][-1].content

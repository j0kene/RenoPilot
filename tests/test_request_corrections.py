"""Focused checks for confirmed dates and approval-time work/route corrections."""
from datetime import date

import pytest

from renopilot import config, demo, run_cli
from renopilot.demo import demo_nodes, parse_brief
from renopilot.graph import build_graph
from renopilot.nodes import approval as approval_node
from renopilot.nodes.approval import _apply_work_updates, _card, _stale_types, apply_routes
from renopilot.nodes.guard import compliance_guard, request_from, verdict_for
from renopilot.nodes.intake import merge
from renopilot.nodes.style import make_shortlist
from renopilot.schemas import ItemSpec, ProjectProfile, StyleProfile, Suggestion, WorkEstimate


def test_cli_auto_approve_stops_on_a_failed_approval_instead_of_looping(monkeypatch, capsys):
    monkeypatch.setattr(demo, "PROFILE", StyleProfile(style_tags=["japandi"], palette=["white"], summary="Tiles.", items=[
        ItemSpec(item_type="floor_tiles", description="tiles", room="living", work_method="overlay")]))
    monkeypatch.setattr(run_cli, "build_graph", lambda planner: build_graph("fixed", nodes=demo_nodes()))
    monkeypatch.setattr("sys.argv", ["run_cli", "--board", str(config.BOARDS / "japandi_01"), "--total", "6000", "--auto-approve"])
    with pytest.raises(SystemExit):
        run_cli.main()
    assert "no checked items" in capsys.readouterr().out


def test_brief_dates_stay_in_notes_and_never_fill_the_confirmed_date():
    for text in ("keys in June 2024", "completed 2024-06", "completed 2024-06-17"):
        parsed = parse_brief(text)
        assert parsed.completion_date is None
        merged = merge(ProjectProfile(), parsed, text)
        assert merged.completion_date is None and merged.notes == text
    exact = date(2024, 6, 17)
    assert merge(ProjectProfile(completion_date=exact), parse_brief("keys in June 2024"), "keys in June 2024").completion_date == exact


def test_routes_helper_applies_only_an_offered_permitted_route():
    work = Suggestion(item_type="floor_tiles", work_type="finish", title="floor tiles", room="bathroom",
                      work_details={"overlay_thickness_mm": 10, "existing_floor_layers": 1})
    checked = compliance_guard({"project": ProjectProfile(), "suggestions": [work]})["suggestions"][0]
    out = apply_routes([checked], {checked.key: "HDB-BATH-3YR-OVERLAY"}, ProjectProfile())
    assert out[0].verdict.rule_id == "HDB-BATH-3YR-OVERLAY"
    assert out[0].verdict.decision in ("allow", "allow_with_permit")
    with pytest.raises(ValueError):
        apply_routes([checked], {checked.key: "HDB-NOT-LISTED"}, ProjectProfile())


def test_work_edit_updates_original_spec_and_reopens_previously_blocked_request():
    spec = ItemSpec(item_type="bathroom_finishes", description="replace bathroom floor", room="bathroom",
                    work_method="replace", request_id="bathroom-1")
    profile = StyleProfile(style_tags=["japandi"], palette=["white"], items=[spec], summary="Soft Japandi.")
    blocked = request_from(spec).model_copy(update={"status": "rejected", "verdict": verdict_for(request_from(spec), ProjectProfile())})
    state = {"style_profile": profile, "suggestions": [blocked]}
    updated_profile, updated, redo, invalidated = _apply_work_updates(state, {blocked.key: {"description": "overlay bathroom floor",
        "work_method": "overlay", "work_details": {"overlay_thickness_mm": 10, "existing_floor_layers": 1}}})
    assert updated_profile.items[0].description == "overlay bathroom floor"
    assert updated_profile.style_tags == profile.style_tags
    assert updated[0].status == "proposed" and updated[0].verdict is None
    assert not redo
    assert invalidated == ["bathroom-1"]
    checked = compliance_guard({"project": ProjectProfile(), "style_profile": updated_profile,
                                "suggestions": updated})["suggestions"][0]
    assert checked.status == "proposed" and checked.verdict.decision == "allow"
    with pytest.raises(ValueError):
        _apply_work_updates(state, {blocked.key: {"made_up": True}})


def test_furniture_edit_uses_profile_key_when_products_have_no_request_id():
    spec = ItemSpec(item_type="sofa", description="compact sofa", room="bedroom", room_label="Bedroom 1")
    profile = StyleProfile(style_tags=["japandi"], palette=["white"], items=[spec], summary="Soft Japandi.")
    rows = [Suggestion(product_id=pid, item_type="sofa", work_type="loose_furniture", title=pid,
                       room="bedroom") for pid in ("shop:a", "shop:b")]
    key = request_from(spec).key
    updated_profile, updated, redo, invalidated = _apply_work_updates({"style_profile": profile, "suggestions": rows},
        {key: {"quantity": 2, "room_label": "Bedroom 1", "max_width_cm": 180}})
    assert updated_profile.items[0].quantity == 2 and updated_profile.items[0].max_width_cm == 180
    assert len(updated) == 2 and all(s.status == "proposed" for s in updated)
    assert redo == ["sofa"]
    assert invalidated == []


def test_stale_types_detect_quantity_aware_per_request_budget_changes():
    profile = StyleProfile(style_tags=["japandi"], palette=["white"], summary="Soft Japandi.", items=[
        ItemSpec(request_id="sofa-a", item_type="sofa", description="sofa A"),
        ItemSpec(request_id="sofa-b", item_type="sofa", description="sofa B"),
    ])
    old = ProjectProfile(total_budget=3000, budget_by_category={"sofa": 2000})
    new = ProjectProfile(total_budget=4000, budget_by_category={"sofa": 2000})
    assert _stale_types(old, new, profile) == ["sofa"]


def test_work_scope_edit_invalidates_linked_estimate_rate_and_check_date(monkeypatch):
    spec = ItemSpec(request_id="bath-a", item_type="bathroom_finishes", description="bathroom tiles", room="bathroom")
    profile = StyleProfile(style_tags=["japandi"], palette=["white"], summary="Soft Japandi.", items=[spec])
    work = request_from(spec)
    quote = WorkEstimate(request_id="bath-a", description="Old tile job", quantity=8,
                         unit_rate_sgd=900, source="contractor quote", checked_at=date(2026, 1, 1))
    project = ProjectProfile(work_estimates=[quote])
    monkeypatch.setattr(approval_node, "interrupt", lambda _: {"work_updates": {"bath-a": {"quantity": 2}}})
    command = approval_node.approval({"style_profile": profile, "suggestions": [work], "project": project})
    updated = command.update["project"].work_estimates[0]
    assert updated.request_id == "bath-a" and updated.source == "contractor quote"
    assert updated.quantity == 8  # keep the prior line for reference; don't scale it to the new scope
    assert updated.unit_rate_sgd is None and updated.checked_at is None and updated.total_sgd is None


def test_approval_add_requests_assigns_ids_and_routes_by_work_type(monkeypatch):
    profile = StyleProfile(style_tags=["japandi"], palette=["white"], summary="Soft Japandi.")
    monkeypatch.setattr(approval_node, "interrupt", lambda _: {"add_requests": [
        {"item_type": "other_renovation", "description": "Install a feature treatment", "room": "living"},
        {"item_type": "sofa", "description": "Small sofa", "room": "bedroom"},
    ]})
    command = approval_node.approval({"style_profile": profile, "suggestions": [], "selected": {}})
    assert command.goto == "planner"
    updated_profile = command.update["style_profile"]
    assert len(updated_profile.items) == 2
    assert all(spec.request_id for spec in updated_profile.items)
    assert command.update["redo_types"] == ["sofa"]
    assert command.update["suggestions"][0].request_id == updated_profile.items[0].request_id
    assert command.update["selected"] == {} and not command.update["approved"]


def test_approval_marks_duplicate_product_ids_by_request_pair(monkeypatch):
    rows = [Suggestion(request_id=request, product_id="shared", item_type="sofa", work_type="loose_furniture",
                       title=request, verdict={"decision": "allow"}) for request in ("bed-a", "bed-b")]
    extra = rows[0].model_copy(update={"request_id": "bed-c", "title": "Comparable"})
    monkeypatch.setattr(approval_node, "interrupt", lambda _: {"approve": "selected", "selected": {}})
    monkeypatch.setattr(approval_node, "selection", lambda *a, **k: ([rows[0], rows[1], extra], {}))
    state = {"suggestions": rows, "selected": {"sofa": "shared"}, "project": ProjectProfile(),
             "pricing_enabled": False}
    command = approval_node.approval(state)
    result = command.update["suggestions"]
    assert [(s.request_id, s.status) for s in result] == [
        ("bed-a", "approved"), ("bed-b", "approved"), ("bed-c", "approved")]
    card = _card(rows[0])
    assert card["request_id"] == "bed-a" and card["selection_key"] == "bed-a"
    assert card["quantity"] == 1 and card["room_label"] == "" and card["fit_note"] == ""


def test_approval_does_not_mark_pending_work_cleared_in_a_mixed_basket(monkeypatch):
    sofa = Suggestion(request_id="bed-a", product_id="sofa:1", item_type="sofa", work_type="loose_furniture",
                      title="Sofa", verdict={"decision": "allow"})
    work = Suggestion(request_id="wall-1", item_type="partition_wall_change", work_type="built_in",
                      title="Partition", verdict={"decision": "allow", "conditions": ["Pending verification: material"]})
    monkeypatch.setattr(approval_node, "interrupt", lambda _: {"approve": "selected", "selected": {}})
    monkeypatch.setattr(approval_node, "selection", lambda *a, **k: ([sofa], {}))
    command = approval_node.approval({"suggestions": [sofa, work], "selected": {"sofa": "sofa:1"},
                                      "project": ProjectProfile(), "pricing_enabled": False})
    statuses = {s.request_id: s.status for s in command.update["suggestions"]}
    assert statuses == {"bed-a": "approved", "wall-1": "proposed"}


@pytest.mark.parametrize("request_id,expected", [("bed-a", ["rejected", "proposed"]), (None, ["rejected", "rejected"])])
def test_rejection_can_be_scoped_to_one_request_or_legacy_product(request_id, expected, monkeypatch):
    rows = [Suggestion(request_id=request, product_id="shared", item_type="sofa", work_type="loose_furniture",
                       title=request, verdict={"decision": "allow"}) for request in ("bed-a", "bed-b")]
    reject = {"product_id": "shared", "reason": "too large"}
    if request_id:
        reject["request_id"] = request_id
    monkeypatch.setattr(approval_node, "interrupt", lambda _: {"reject": [reject]})
    state = {"suggestions": rows, "selected": {"sofa": "shared"}, "project": ProjectProfile()}
    command = approval_node.approval(state)
    assert [s.status for s in command.update["suggestions"]] == expected


def test_rejected_product_exclusion_is_scoped_to_its_request():
    class Index:
        embed_spec = "fake"

        def __init__(self):
            self.calls = []

        def search(self, *args, **kwargs):
            self.calls.append((kwargs["item_type"], set(kwargs["exclude_ids"])))
            return []

    profile = StyleProfile(style_tags=["japandi"], palette=["white"], summary="Soft Japandi.", items=[
        ItemSpec(request_id="bed-a", item_type="sofa", description="sofa", room="bedroom"),
        ItemSpec(request_id="bed-b", item_type="sofa", description="sofa", room="bedroom"),
    ])
    index = Index()
    make_shortlist(index=index, signals=("bm25",))({
        "style_profile": profile, "project": ProjectProfile(), "board_images": [],
        "shortlist_types": ["sofa"], "suggestions": [],
        "rejected": [{"product_id": "catalog:shared", "request_id": "bed-a", "by": "user"}],
    })
    assert index.calls == [("sofa", {"catalog:shared"}), ("sofa", set())]

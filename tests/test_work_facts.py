"""Structured renovation facts that change an HDB route or keep it pending."""
from renopilot.nodes.guard import compliance_guard, request_from, verdict_for, with_route
from renopilot.schemas import ItemSpec, ProjectProfile, StyleProfile, Suggestion, WorkDetails


def test_overlay_measurement_blocks_over_13mm_and_unknown_layers_escalate():
    too_thick = Suggestion(item_type="floor_tiles", work_type="finish", room="living", work_method="overlay",
                           title="Tile overlay", work_details=WorkDetails(overlay_thickness_mm=30,
                                                                           existing_floor_layers=1))
    v = verdict_for(too_thick, ProjectProfile())
    assert v.decision == "block" and "30mm" in v.reason and v.citation

    unknown = too_thick.model_copy(update={"work_details": WorkDetails(overlay_thickness_mm=12)})
    v = verdict_for(unknown, ProjectProfile())
    assert v.decision == "escalate" and "one existing floor-finish layer" in v.reason and v.citation


def test_overlay_at_limit_with_one_layer_passes_with_unimplemented_conditions_pending():
    s = Suggestion(item_type="floor_tiles", work_type="finish", room="living", work_method="overlay",
                   title="Tile overlay", work_details=WorkDetails(overlay_thickness_mm=13,
                                                                   existing_floor_layers=1))
    v = verdict_for(s, ProjectProfile())
    assert v.decision == "allow"
    assert any(c.startswith("Pending verification:") for c in v.conditions)


def test_explicit_text_that_conflicts_with_structured_measurement_escalates():
    s = Suggestion(item_type="floor_tiles", work_type="finish", room="living", work_method="overlay",
                   title="30mm tile overlay", work_details=WorkDetails(overlay_thickness_mm=10,
                                                                         existing_floor_layers=1))
    v = verdict_for(s, ProjectProfile())
    assert v.decision == "escalate" and "conflicts" in v.reason and v.citation
    overlay = next(option for option in v.options if option.rule_id == "HDB-FLOOR-OVERLAY")
    assert overlay.decision == "escalate"


def test_unselected_method_cannot_offer_an_overlay_that_facts_block():
    s = Suggestion(item_type="floor_tiles", work_type="finish", room="living", title="Floor tile work",
                   work_details=WorkDetails(overlay_thickness_mm=30, existing_floor_layers=1))
    v = verdict_for(s, ProjectProfile())
    overlay = next(option for option in v.options if option.rule_id == "HDB-FLOOR-OVERLAY")
    assert v.decision == "escalate" and overlay.decision == "block"
    assert with_route(v, "HDB-FLOOR-OVERLAY").rule_id == v.rule_id


def test_false_ceiling_measurements_and_unknown_facts():
    too_low = Suggestion(item_type="false_ceiling", work_type="built_in", room="living",
                         title="1.8m false ceiling")
    assert verdict_for(too_low, ProjectProfile()).decision == "block"

    contradictory = too_low.model_copy(update={"work_details": WorkDetails(ceiling_clearance_mm=2500,
                                                                            non_combustible=True,
                                                                            gas_pipe_enclosed=False)})
    assert verdict_for(contradictory, ProjectProfile()).decision == "escalate"

    unrelated_dimension = too_low.model_copy(update={"title": "False ceiling with 30mm panel",
        "work_details": WorkDetails(ceiling_clearance_mm=2500, non_combustible=True, gas_pipe_enclosed=False)})
    assert verdict_for(unrelated_dimension, ProjectProfile()).decision == "allow"

    unknown = too_low.model_copy(update={"title": "False ceiling", "work_details": WorkDetails()})
    assert verdict_for(unknown, ProjectProfile()).decision == "escalate"

    compliant = too_low.model_copy(update={"title": "False ceiling", "work_details": WorkDetails(
        ceiling_clearance_mm=2400, non_combustible=True, gas_pipe_enclosed=False)})
    assert verdict_for(compliant, ProjectProfile()).decision == "allow"


def test_plywood_partition_is_blocked_and_unknown_material_escalates():
    plywood = Suggestion(item_type="partition_wall_change", work_type="built_in", room="living",
                         work_method="erect", title="New plywood partition")
    v = verdict_for(plywood, ProjectProfile())
    assert v.decision == "block" and v.rule_id == "HDB-WALL-ERECT" and v.citation

    unknown = plywood.model_copy(update={"title": "New partition"})
    v = verdict_for(unknown, ProjectProfile())
    assert v.decision == "escalate" and v.citation


def test_requested_work_carries_measured_facts_and_identity_into_the_guard():
    spec = ItemSpec(item_type="floor_tiles", description="Overlay floor tiles", room="living",
                    work_method="overlay", request_id="tile-1", room_label="Living room", quantity=2,
                    work_details=WorkDetails(overlay_thickness_mm=30, existing_floor_layers=1))
    request = request_from(spec)
    assert (request.request_id, request.room_label, request.quantity) == ("tile-1", "Living room", 2)
    profile = StyleProfile(style_tags=["japandi"], palette=["white"], room="living", items=[spec], summary="Simple")
    checked = compliance_guard({"style_profile": profile, "suggestions": [], "project": ProjectProfile()})
    assert checked["suggestions"][0].verdict.decision == "block"

"""Live extraction failures reproduced at the model boundary, without a fixed demo profile."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from renopilot import trace
from renopilot.nodes.guard import compliance_guard, verdict_for
from renopilot.nodes.style import extract_style_profile, normalize_extracted_requests
from renopilot.schemas import ItemSpec, ProjectProfile, StyleProfile, Suggestion
from renopilot.tools.budget import request_caps


def profile(*items):
    return StyleProfile(style_tags=["japandi"], palette=["beige"], summary="Warm wood", items=list(items))


def test_model_duplicates_do_not_create_extra_purchases_or_shrink_caps():
    raw = profile(ItemSpec(item_type="sofa", description="green sofa", room="living", request_id="model-a"),
                  ItemSpec(item_type="sofa", description="cream sofa", room="bedroom", request_id="model-b"),
                  ItemSpec(item_type="pendant_light", description="paper light", room="living"),
                  ItemSpec(item_type="pendant_light", description="round light", room="living"))
    class Model:
        def with_structured_output(self, *args, **kwargs):
            return SimpleNamespace(invoke=lambda _: {"raw": AIMessage(content=""), "parsed": raw, "parsing_error": None})
    with trace.tracing() as logged:
        result, _ = extract_style_profile([], model=Model())
    assert [i.item_type for i in result.items] == ["sofa", "pendant_light"]
    assert all(not i.request_id for i in result.items)
    expected = [raw.items[0].model_copy(update={"request_id": ""}), raw.items[2]]
    assert request_caps(ProjectProfile(total_budget=1000), result.items) == request_caps(ProjectProfile(total_budget=1000), expected)
    row = next(r for r in logged.rows if r["event"] == "result")
    assert row["requests"][0]["room"] == "living"
    assert row["requests"][0]["room_label"] == ""


@pytest.mark.parametrize("brief", ["", "A sofa for my living room", "Raise the floor by 100mm"])
def test_single_room_or_unrelated_brief_does_not_authorize_two_sofas(brief):
    raw = profile(*[ItemSpec(item_type="sofa", description="sofa", room=room) for room in ("living", None, "bedroom")])
    assert len(normalize_extracted_requests(raw, brief).items) == 1


def test_explicit_multiroom_requests_and_quantity_survive_but_repeated_photos_do_not():
    a = ItemSpec(item_type="sofa", description="compact sofa", room="bedroom", room_label="Bedroom 1", quantity=2)
    b = a.model_copy(update={"room_label": "Bedroom 2", "quantity": 1})
    work = [ItemSpec(item_type="floor_tiles", description="tiles", room=r) for r in ("living", "bathroom")]
    result = normalize_extracted_requests(profile(a, a, b, *work), "Two sofas for Bedroom 1 and one sofa for Bedroom 2")
    assert [(i.room_label, i.quantity) for i in result.items[:2]] == [("Bedroom 1", 2), ("Bedroom 2", 1)]
    assert [i.room for i in result.items[2:]] == ["living", "bathroom"]


def test_catchall_floor_raise_is_normalized_and_blocked():
    raw = profile(ItemSpec(item_type="other_renovation", description="Raise the floor level by 100mm with concrete"))
    normalized = normalize_extracted_requests(raw)
    assert normalized.items[0].item_type == "floor_tiles"
    rows = compliance_guard({"style_profile": normalized, "project": ProjectProfile()})["suggestions"]
    assert rows[0].verdict.rule_id == "HDB-FLOOR-RAISE" and rows[0].status == "rejected"


@pytest.mark.parametrize("room", [None, "living", "bathroom"])
def test_guard_blocks_existing_catchall_floor_raise_even_without_extraction(room):
    s = Suggestion(item_type="other_renovation", work_type="built_in", title="Raise floor by 100mm with concrete", room=room)
    v = verdict_for(s, ProjectProfile())
    assert (v.decision, v.rule_id) == ("block", "HDB-FLOOR-RAISE")


def test_unrelated_unsupported_work_stays_unsupported():
    spec = ItemSpec(item_type="other_renovation", description="Raise an electrical socket above the floor", room="living")
    # Raising a socket is not raising the floor.
    assert normalize_extracted_requests(profile(spec)).items[0].item_type == "other_renovation"

"""Offline AppTest regressions for the owner-facing work and request flows."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from renopilot import config, demo  # noqa: E402
from renopilot.schemas import ItemSpec, StyleProfile, WorkDetails  # noqa: E402


def profile(*items):
    return StyleProfile(style_tags=["japandi"], palette=["beige"], room="mixed", items=list(items), summary="Test board")


def app_for(monkeypatch, *items):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setattr(demo, "PROFILE", profile(*items))
    app = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=30).run()
    next(r for r in app.radio if r.label == "Mode").set_value("Demo (offline)").run()
    next(b for b in app.button if b.label == "Run").click().run()
    assert not app.exception
    return app


def test_work_only_route_can_be_selected_and_reviewed_without_claiming_clearance(monkeypatch):
    app = app_for(monkeypatch, ItemSpec(item_type="floor_tiles", description="Floor tiles", room="living",
        work_method="unspecified", work_details=WorkDetails(overlay_thickness_mm=10, existing_floor_layers=1)))
    route = next(r for r in app.radio if r.label == "How will this be done?")
    assert any(option.startswith("Overlay:") for option in route.options)
    route.set_value("HDB-FLOOR-OVERLAY").run()
    approve = next(b for b in app.button if b.label == "Approve selected basket")
    assert not approve.disabled
    approve.click().run()
    assert not app.exception
    state = app.session_state.graph.get_state(app.session_state.cfg).values
    work = next(s for s in state["suggestions"] if not s.product_id)
    assert state["approved"] and work.status == "proposed"
    assert work.verdict.rule_id == "HDB-FLOOR-OVERLAY"
    assert any(c.startswith("Pending verification:") for c in work.verdict.conditions)


def test_request_quantity_edit_saves_and_rechecks(monkeypatch):
    app = app_for(monkeypatch, ItemSpec(item_type="sofa", description="Low linen sofa", room="bedroom",
                                        request_id="sofa-bed2", room_label="Bedroom 2"))
    next(n for n in app.number_input if n.label == "Quantity").set_value(3)
    next(b for b in app.get("form_submit_button") if b.label == "Save request and recheck").click().run()
    assert not app.exception
    state = app.session_state.graph.get_state(app.session_state.cfg).values
    spec = next(i for i in state["style_profile"].items if i.request_id == "sofa-bed2")
    suggestion = next(s for s in state["suggestions"] if s.request_id == "sofa-bed2")
    assert spec.quantity == suggestion.quantity == 3


def test_rebuild_clears_old_picks_and_rejection_ticks(monkeypatch):
    app = app_for(monkeypatch, ItemSpec(item_type="sofa", description="Low linen sofa", room="living"))
    choice = next(s for s in app.selectbox if s.label.startswith("Choose one sofa"))
    default = choice.value
    state = app.session_state.graph.get_state(app.session_state.cfg).values
    other = next(s.product_id for s in state["suggestions"] if s.product_id and s.product_id != default)
    choice.set_value(other).run()
    next(c for c in app.checkbox if c.key.startswith("rej_") and c.key.endswith(other)).set_value(True).run()
    next(b for b in app.get("form_submit_button") if b.label == "Rebuild from these inputs").click().run()
    assert not app.exception
    assert next(s for s in app.selectbox if s.label.startswith("Choose one sofa")).value == default
    assert not any(c.value for c in app.checkbox if c.key.startswith("rej_"))


def test_two_bedroom_sofa_requests_have_independent_choices(monkeypatch):
    app = app_for(monkeypatch,
        ItemSpec(item_type="sofa", description="Compact sofa", room="bedroom", request_id="bedroom-1",
                 room_label="Bedroom 1"),
        ItemSpec(item_type="sofa", description="Compact sofa", room="bedroom", request_id="bedroom-2",
                 room_label="Bedroom 2"))
    choices = [s for s in app.selectbox if s.label.startswith("Choose one sofa")]
    assert len(choices) == 2
    suggestions = app.session_state.graph.get_state(app.session_state.cfg).values["suggestions"]
    second = next(s.product_id for s in suggestions if s.request_id == "bedroom-2" and
                  s.product_id != choices[0].value)
    choices[1].set_value(second).run()
    assert choices[0].value != choices[1].value
    assert not app.exception

"""The optional UI group exercises the same approval and validation gates as the CLI."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from renopilot import config  # noqa: E402


def demo(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    app = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=20).run()
    next(r for r in app.radio if r.label == "Mode").set_value("Demo (offline)").run()
    return app


def test_select_and_export_one_offer_per_type(monkeypatch):
    app = demo(monkeypatch)
    assert next(s for s in app.selectbox if s.label == "Board").options == ["japandi_01"]
    assert next(r for r in app.radio if r.label == "Board source").options == ["Example board"]
    next(b for b in app.button if b.label == "Run").click().run()
    assert not app.exception
    choose = next(s for s in app.selectbox if s.label == "Choose one sofa")
    alternatives = app.session_state.graph.get_state(app.session_state.cfg).values["suggestions"]
    other = next(s.product_id for s in alternatives if s.item_type == "sofa" and s.product_id != choose.value
                 and s.status != "rejected")
    choose.set_value(other).run()
    assert not app.exception
    next(b for b in app.button if b.label == "Approve selected basket").click().run()
    assert not app.exception and app.success[0].value == "Selected basket approved."
    state = app.session_state.graph.get_state(app.session_state.cfg).values
    assert state["selected"]["sofa"] == other
    picks = [s for s in state["suggestions"] if s.product_id and s.status == "approved"]
    assert len(picks) == len({s.item_type for s in picks}) == 4
    assert len(app.get("download_button")) == 1


def test_bad_budget_entry_preserves_the_question_and_can_be_corrected(monkeypatch):
    app = demo(monkeypatch)
    next(n for n in app.number_input if n.label == "Max for one coffee table").set_value(50).run()
    next(b for b in app.button if b.label == "Run").click().run()
    field = next(t for t in app.text_input if t.label.startswith("Change budgets"))
    field.input("coffee_table=abc")
    next(b for b in app.button if b.label == "Send").click().run()
    assert not app.exception and app.error
    next(t for t in app.text_input if t.label.startswith("Change budgets")).input("coffee_table=300")
    next(b for b in app.button if b.label == "Send").click().run()
    assert not app.exception and any(b.label == "Approve selected basket" for b in app.button)

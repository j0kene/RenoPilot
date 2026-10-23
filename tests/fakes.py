"""Stand-ins so the graph runs with no model and no catalog: fake style nodes and a fake
planner model that replays scripted decisions."""
from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from renopilot.schemas import ItemSpec, StyleProfile, Suggestion
from renopilot.tools.budget import caps_for

PROFILE = StyleProfile(
    style_tags=["japandi"], palette=["light_wood", "off_white_cream"], materials=["light_wood", "linen_cotton"],
    items=[ItemSpec(item_type="sofa", description="low linen sofa"),
           ItemSpec(item_type="coffee_table", description="round light oak coffee table"),
           ItemSpec(item_type="floor_tiles", description="matte grey terrazzo floor tiles", priority="nice")],
    summary="Warm, pared-back Japandi living room.")

# a tiny catalog: (id, type, price)
CATALOG = [(f"sofa{i}", "sofa", 600 + 200 * i) for i in range(8)] + \
          [(f"table{i}", "coffee_table", 80 + 40 * i) for i in range(8)]


def fake_style_extract(state):
    return {"style_profile": PROFILE, "trace": [{"node": "style_extract", "action": "profile", "why": "fake"}]}


def fake_shortlist(state):
    profile = state["style_profile"]
    types = state.get("shortlist_types") or [i.item_type for i in profile.items]
    caps = caps_for(state.get("project"), types)
    exclude = {r["product_id"] for r in state.get("rejected") or [] if r.get("product_id")}
    new, esc = [], []
    for t in types:
        if t == "floor_tiles":
            new.append(Suggestion(item_type=t, work_type="finish", title="terrazzo tiles"))
            continue
        ok = [(i, p) for i, tt, p in CATALOG if tt == t and i not in exclude and (caps[t] is None or p <= caps[t])]
        if len(ok) < 3:
            esc.append(f"Only {len(ok)} {t} option(s) within budget")
        new += [Suggestion(product_id=i, item_type=t, work_type="loose_furniture", title=i, price_sgd=p,
                           reason="fake match") for i, p in ok[:5]]
    kept = [s for s in state.get("suggestions") or [] if s.item_type not in types or s.status == "rejected"]
    return {"suggestions": kept + new, "escalations": esc, "open_issues": esc,
            "redo_types": [t for t in state.get("redo_types") or [] if t not in types], "shortlist_types": []}


FAKE_NODES = {"style_extract": fake_style_extract, "shortlist": fake_shortlist}


class ScriptedPlanner:
    """Replays Decision dicts in order; behaves like model.with_structured_output(...)."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = 0

    def with_structured_output(self, schema, include_raw=False, **_):
        def run(_msgs):
            self.calls += 1
            d = self.decisions.pop(0) if self.decisions else None
            parsed = schema.model_validate(d) if d else None
            return {"raw": None, "parsed": parsed, "parsing_error": None if parsed else ValueError("no reply")}
        return RunnableLambda(run)

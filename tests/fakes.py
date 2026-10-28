"""Stand-ins so the graph runs with no model and no catalog: fake style nodes, a fake
planner model that replays scripted decisions, and a scripted tool-calling chat model."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from renopilot.schemas import ItemSpec, StyleProfile, Suggestion
from renopilot.tools.budget import caps_for

PROFILE = StyleProfile(
    style_tags=["japandi"], palette=["light_wood", "off_white_cream"], materials=["light_wood", "linen_cotton"],
    items=[ItemSpec(item_type="sofa", description="low linen sofa"),
           ItemSpec(item_type="coffee_table", description="round light oak coffee table"),
           ItemSpec(item_type="floor_tiles", description="matte grey terrazzo floor tiles", priority="nice", room="living")],
    summary="Warm, pared-back Japandi living room.")

# a tiny catalog: (id, type, price)
CATALOG = [(f"sofa{i}", "sofa", 600 + 200 * i) for i in range(8)] + \
          [(f"table{i}", "coffee_table", 80 + 40 * i) for i in range(8)]


def fake_style_extract(state):
    return {"style_profile": PROFILE, "trace": [{"node": "style_extract", "action": "profile", "why": "fake"}]}


def fake_shortlist(state):
    profile = state["style_profile"]
    types = state.get("shortlist_types") or [i.item_type for i in profile.items]
    caps = caps_for(state.get("project"), [i.item_type for i in profile.items if i.item_type != "floor_tiles"])
    exclude = {r["product_id"] for r in state.get("rejected") or [] if r.get("product_id")}
    new, esc = [], []
    for t in types:
        if t == "floor_tiles":
            new.append(Suggestion(item_type=t, work_type="finish", title="terrazzo tiles", room="living"))
            continue
        ok = [(i, p) for i, tt, p in CATALOG if tt == t and i not in exclude and (caps[t] is None or p <= caps[t])]
        if len(ok) < 3:
            esc.append(f"Only {len(ok)} {t} option(s) within budget")
        new += [Suggestion(product_id=i, item_type=t, work_type="loose_furniture", title=i, price_sgd=p,
                           reason="fake match") for i, p in ok[:5]]
    kept = [s for s in state.get("suggestions") or [] if s.item_type not in types or s.status == "rejected"]
    return {"suggestions": kept + new, "escalations": esc, "open_issues": esc,
            "redo_types": [t for t in state.get("redo_types") or [] if t not in types], "shortlist_types": []}


def fake_pricing(state):
    return fake_pricing_node()(state)


FAKE_NODES = {"style_extract": fake_style_extract, "shortlist": fake_shortlist, "pricing": fake_pricing}


class ScriptedPlanner:
    """Replays Decision dicts in order as tool calls; behaves like model.bind_tools(...)."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = 0

    def bind_tools(self, tools, **_):
        def run(_msgs):
            self.calls += 1
            if not self.decisions:
                return AIMessage("")                     # no tool call
            d = dict(self.decisions.pop(0))
            return AIMessage("", tool_calls=[{"name": d.pop("action"), "args": d, "id": f"call_{self.calls}"}])
        return RunnableLambda(run)


class ScriptedToolModel(BaseChatModel):
    """A chat model that replays one tool call per turn, [(name, args), ...], for agents
    (create_agent binds tools and reads AIMessage.tool_calls). A real model, so callbacks fire."""

    script: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **_):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **_):
        return self._call(*self.script.pop(0), n=len(messages))

    @staticmethod
    def _call(name, args, n):
        msg = AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{n}"}],
                        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
        return ChatResult(generations=[ChatGeneration(message=msg)])


class FakeShopperModel(ScriptedToolModel):
    """Plays the shopper from the conversation so far (stateless, so one instance can serve
    every item type's thread): search with the item's description, price the first hit, pick it."""

    def _generate(self, messages, stop=None, run_manager=None, **_):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        rows = results[0].content.split("\n")[1:] if results else []
        first = rows[0].split(" | ")[0] if rows else None
        if not results:
            item = next(m.content for m in messages if m.type == "human").split("ITEM TO SHOP FOR: ")[1]
            call = ("search_catalog", {"query": item.split(": ", 1)[1].split("\n")[0]})
        elif len(results) == 1 and first:
            call = ("landed_price", {"product_id": first})
        else:
            call = ("Picks", {"picks": [{"product_id": first, "reason": "fake pick"}] if first else []})
        return self._call(*call, n=len(messages))


# ---------------------------------------------------------------- pricing in the fake world
def fake_priced_catalog():
    """The fake shortlist's products as IKEA SG listings: sofas need home delivery (S$35),
    tables go by parcel (S$10). So a S$1,000 sofa costs S$1,035 landed."""
    from renopilot.schemas import Product
    out = {}
    for pid, t, price in CATALOG:
        heavy = t == "sofa"
        out[pid] = Product(id=pid, source="ikea_sg", item_type=t, title=pid, price=price, currency="SGD",
                           weight_kg=60 if heavy else 8, dims="200 cm" if heavy else "80 cm")
    return out


class StaleFx:
    """An FX provider whose copy is older than 24 h (offline with an old cache)."""

    def get(self, currency):
        from renopilot.pricing.fx import FxRate
        return FxRate(currency, 1.0, "2026-10-20", "2026-10-20T00:00:00+00:00", "old cache", True)


def fake_index():
    """The fake catalog as a search index: BM25 only (no embeddings), so search with signals=("bm25",)."""
    import numpy as np

    from renopilot.catalog.search import CatalogIndex
    products = [p.model_copy(update={"price_sgd": p.price, "title": f"{p.item_type.replace('_', ' ')} {p.id}"})
                for p in fake_priced_catalog().values()]
    return CatalogIndex(products, np.zeros((len(products), 4)), "none")


def fake_pricing_node(stale: bool = False):
    from renopilot.pricing.fx import FxProvider
    from renopilot.pricing.node import make_pricing
    cat = fake_priced_catalog()
    return make_pricing(products=cat, pool=list(cat.values()), index=None, comparables=False,
                        fx=StaleFx() if stale else FxProvider(mode="snapshot"))

"""The offline demo world: no model and no catalog index needed. The graph, the gates, the shopper
agents' tools, landed-cost pricing, the HDB compliance guard and the tracing are the real code, over the
frozen SG catalog snapshot (real IKEA SG and Castlery products, prices and photos, data/sg_catalog.csv;
the overseas rows are the synthetic samples of data/taobao_manual.csv). Only the models are played
by code, and search is BM25 only (no embeddings).

    demo_nodes()     node overrides for build_graph(nodes=...): used by the Streamlit app's
                     demo mode and evals/demo_trace.py

The master is played by the fixed policy, but through the master's own tool calls (so its trace
looks like the LLM master's), and it turns the owner's rejection reason into a search
refinement as the LLM master would. Each shopper searches with the item's description, prices
its best hit, and picks hits that share the board's words. The comparables judge says yes to every
pair it is offered; the Pricing agent's size rule still rejects the ones that do not fit.
The owner's typed brief is read by a small regex parser (flat type, completion month, budgets)
instead of the intake model, and a brief that mentions the bathroom adds bathroom finishes to the
board's items, so the HDB 3-year rule can be seen reacting to the completion date.
"""
from __future__ import annotations

import csv
import re
from datetime import date

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from renopilot import config
from renopilot.catalog.search import CatalogIndex
from renopilot.nodes.intake import make_intake
from renopilot.nodes.planner import fixed_policy, make_planner
from renopilot.nodes.style import make_shortlist
from renopilot.pricing.fx import FxProvider
from renopilot.pricing.node import make_pricing
from renopilot.schemas import ItemSpec, ProjectProfile, Product, StyleProfile
from renopilot.vocab import LOOSE_FURNITURE

PROFILE = StyleProfile(
    style_tags=["japandi", "minimalist"], palette=["light_wood", "off_white_cream", "beige"],
    materials=["light_wood", "linen_cotton", "paper_fabric_shade"],
    items=[ItemSpec(item_type="sofa", description="low linen sofa with light oak legs", priority="must"),
           ItemSpec(item_type="coffee_table", description="round light oak coffee table", priority="must"),
           ItemSpec(item_type="floor_lamp", description="paper shade floor lamp", priority="nice"),
           ItemSpec(item_type="rug", description="natural jute flatwoven rug", priority="nice"),
           ItemSpec(item_type="floor_tiles", description="matte grey terrazzo floor tiles", priority="nice")],
    summary="Warm, pared-back Japandi living room: pale oak, linen and paper light.")

TYPES = tuple(i.item_type for i in PROFILE.items)
_STOP = {"palette", "materials", "and", "the", "with", "room", "living", "warm", "pared", "back"}


def catalog() -> list[Product]:
    """The snapshot's rows for the board's item types: SG retailers (with photos) and overseas samples."""
    fields = set(Product.model_fields)
    rows = list(csv.DictReader((config.DATA / "sg_catalog.csv").open(encoding="utf-8")))
    for r in csv.DictReader((config.DATA / "taobao_manual.csv").open(encoding="utf-8")):
        if r.get("title") and r.get("price"):
            rows.append(r | {"synthetic": str(r.get("synthetic", "")).lower() in ("true", "1", "yes")})
    return [Product.model_validate({k: v for k, v in r.items() if k in fields and v not in ("", None)})
            for r in rows if r["item_type"] in TYPES]


def demo_index() -> CatalogIndex:
    """The SG rows as a search index (overseas rows have no SGD list price: like the live index, they
    only ever appear as comparables). BM25 only, so search with signals=("bm25",)."""
    import numpy as np
    products = [p for p in catalog() if p.price_sgd is not None]
    return CatalogIndex(products, np.zeros((len(products), 4)), "none")


# --------------------------------------------------------------------------- the models, played by code
def _reply(name: str, args: dict, n: int) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(
        "", tool_calls=[{"name": name, "args": args, "id": f"call_{n}"}],
        usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}))])


class DemoMaster(BaseChatModel):
    """The fixed policy, speaking the master's tool calls. Built per step: it needs the state."""

    state: dict = {}

    @property
    def _llm_type(self) -> str:
        return "demo-master"

    def bind_tools(self, tools, **_):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **_):
        d = fixed_policy(self.state)
        args = {"why": d.why}
        if d.action == "shortlist":
            args["item_types"] = list(d.item_types)
            reasons = [r["reason"] for r in self.state.get("rejected") or []          # owner feedback -> refinement
                       if r.get("by") == "user" and r.get("item_type") in d.item_types and r.get("reason")]
            if reasons and self.state.get("redo_types"):
                args["refinement"] = " ".join(reasons[-1].split()[:12])
        if d.action == "ask_user":
            args["question"] = d.question
        return _reply(d.action, args, len(messages))


class DemoShopper(BaseChatModel):
    """Plays a shopper from the conversation so far (stateless, so one instance serves every
    item type's thread): search with the description, price the best hit, pick the hits in budget."""

    @property
    def _llm_type(self) -> str:
        return "demo-shopper"

    def bind_tools(self, tools, **_):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **_):
        prompt = next(m.content for m in messages if m.type == "human")
        item = prompt.split("ITEM TO SHOP FOR: ")[1].split("\n")[0].split(": ", 1)[1]
        board = set(re.findall(r"[a-z]+", prompt.split("STYLE PROFILE: ")[1].split("\n")[0].lower())) - _STOP
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            return _reply("search_catalog", {"query": item}, len(messages))
        rows = [r.split(" | ") for r in results[0].content.split("\n")[1:]]
        if len(results) == 1 and rows:
            return _reply("landed_price", {"product_id": rows[0][0]}, len(messages))
        over = {rows[0][0]} if rows and "OVER" in results[-1].content else set()
        picks = []
        for pid, title, colour, material, _price in (r[:5] for r in rows if len(r) >= 5):
            shared = sorted(board & set(re.findall(r"[a-z]+", f"{title} {colour} {material}".lower())))
            if pid not in over and shared:                  # only hits that share the board's words
                picks.append({"product_id": pid, "reason": f"{', '.join(shared[:3]).capitalize()}: in keeping with the board"})
        return _reply("Picks", {"picks": picks[:3]}, len(messages))


class DemoJudge:
    """The comparables judge, played by code: every offered pair is 'comparable'. The Pricing
    agent's dims rule (code) still rejects pairs whose sizes differ too much."""

    def with_structured_output(self, schema, include_raw=False, **_):
        def run(messages):
            text = messages[-1].content
            pairs = []
            for block in text.split("SHORTLISTED: ")[1:]:
                sid = block.split(" | ")[0]
                for line in block.split("CANDIDATES:\n")[1].splitlines():
                    if line.strip():
                        pairs.append({"shortlisted_id": sid, "candidate_id": line.strip().split(" | ")[0],
                                      "comparable": True, "why": "Same kind of piece at another retailer."})
            return {"raw": None, "parsed": schema.model_validate({"pairs": pairs}), "parsing_error": None}
        return RunnableLambda(run)


# --------------------------------------------------------------------------- the brief, read by code
MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                      "nov", "dec"), 1)}


def parse_brief(text: str) -> ProjectProfile:
    """Plays the intake model: flat type, block completion month, total and per-item budgets."""
    t = text.lower().replace(",", "")
    flat = re.search(r"\b([2-5])[- ]?room\b", t)
    flat_type = f"{flat.group(1)}-room" if flat else ("executive" if "executive" in t else None)
    done = None
    if m := re.search(r"\b(20\d\d)-(\d{1,2})(?:-(\d{1,2}))?\b", t):
        done = date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))
    elif m := re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(20\d\d)\b", t):
        done = date(int(m.group(2)), MONTHS[m.group(1)], 1)
    names = "|".join(sorted((i.replace("_", " ") for i in LOOSE_FURNITURE), key=len, reverse=True))
    caps = {}
    for word, amount in re.findall(rf"\b({names})s?\b\D{{0,12}}?(?:under|below|max|up to|within)\s*s?\$?\s*(\d+)", t):
        caps[word.replace(" ", "_")] = float(amount)
    total = re.search(r"(?:total|overall|whole)?\s*budget\D{0,15}?s?\$\s*(\d+)", t)
    return ProjectProfile(flat_type=flat_type, completion_date=done, budget_by_category=caps,
                          total_budget=float(total.group(1)) if total else None)


# --------------------------------------------------------------------------- nodes
def style_extract(state: dict) -> dict:
    project = state.get("project")
    brief = (project.notes if project else "").lower()
    profile = PROFILE
    if any(w in brief for w in ("bathroom", "toilet")):        # the live VLM is told to add items the brief asks for
        profile = PROFILE.model_copy(update={"items": PROFILE.items + [ItemSpec(
            item_type="bathroom_finishes", description="new bathroom floor and wall tiles, as in the owner's brief",
            priority="must")]})
    return {"style_profile": profile,
            "trace": [{"node": "style_extract", "action": "profile", "why": "demo: a fixed Japandi reading of the board"
                       + (" + the bathroom finishes the brief asked for" if profile is not PROFILE else "")}]}


def planner(state: dict):
    return make_planner("llm", model=DemoMaster(state=state))(state)


def demo_nodes() -> dict:
    products = catalog()
    return {"intake": make_intake(parse=parse_brief), "style_extract": style_extract, "planner": planner,
            "shortlist": make_shortlist(index=demo_index(), signals=("bm25",), shopper_model=DemoShopper()),
            "pricing": make_pricing(products={p.id: p for p in products}, pool=products, index=None,
                                    judge_model=DemoJudge(), fx=FxProvider(mode="snapshot"))}

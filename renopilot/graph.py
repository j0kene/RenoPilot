"""The RenoPilot graph (proposal Fig. 1).

    START -> intake -> planner -> {style_extract | ask_user} -> planner ...
                       planner -> shortlist -> pricing -> planner
                       planner -> compliance_guard -> approval -> respond -> END
                                                      approval -> planner   (rejections)

The planner and the approval gate route with Command(goto=...): the node that validates a
decision is also the node that routes it. Everything else is a fixed edge. In particular
compliance_guard -> approval is an edge, so nothing reaches the owner unchecked.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from renopilot.llm import is_transient
from renopilot.nodes.approval import approval, ask_user
from renopilot.nodes.guard import compliance_guard
from renopilot.nodes.intake import intake
from renopilot.nodes.planner import make_planner
from renopilot.nodes.respond import respond
from renopilot.nodes.style import shortlist_node, style_extract_node
from renopilot.pricing.node import pricing as pricing_node
from renopilot.state import State

RETRY = RetryPolicy(max_attempts=3, retry_on=is_transient)   # retry the node, not the run

# Checkpoints hold our pydantic models; allow exactly those to be deserialised.
SERDE = JsonPlusSerializer(allowed_msgpack_modules=[
    ("renopilot.schemas", name) for name in
    ("ProjectProfile", "StyleProfile", "ItemSpec", "Product", "Suggestion", "Verdict",
                 "PriceQuote", "PriceLine", "PricedAlternative")])


def make_checkpointer():
    return InMemorySaver(serde=SERDE)


def build_graph(planner: str = "llm", checkpointer=None, nodes: dict | None = None, pricing: bool = True):
    """planner: "llm" | "fixed". pricing: False drops the pricing node (the P0 graph, for the
    ablation). `nodes` swaps implementations (tests use fakes; the compliance workstream plugs in here)."""
    impl = {"intake": intake, "planner": make_planner(planner), "style_extract": style_extract_node,
            "shortlist": shortlist_node, "pricing": pricing_node, "ask_user": ask_user,
            "compliance_guard": compliance_guard, "approval": approval, "respond": respond} | (nodes or {})
    b = StateGraph(State)
    b.add_node("intake", impl["intake"], retry_policy=RETRY)
    b.add_node("planner", impl["planner"], retry_policy=RETRY,
               destinations=("style_extract", "shortlist", "ask_user", "compliance_guard", "respond"))
    b.add_node("style_extract", impl["style_extract"], retry_policy=RETRY)
    b.add_node("shortlist", impl["shortlist"], retry_policy=RETRY)
    if pricing:
        b.add_node("pricing", impl["pricing"], retry_policy=RETRY)
    b.add_node("ask_user", impl["ask_user"])
    b.add_node("compliance_guard", impl["compliance_guard"])
    b.add_node("approval", impl["approval"], destinations=("respond", "planner"))
    b.add_node("respond", impl["respond"])

    b.add_edge(START, "intake")
    b.add_edge("intake", "planner")
    for n in ("style_extract", "ask_user"):
        b.add_edge(n, "planner")
    if pricing:                                     # every new shortlist is priced before the planner sees it
        b.add_edge("shortlist", "pricing")
        b.add_edge("pricing", "planner")
    else:
        b.add_edge("shortlist", "planner")
    b.add_edge("compliance_guard", "approval")
    b.add_edge("respond", END)
    return b.compile(checkpointer=checkpointer or make_checkpointer())


if __name__ == "__main__":
    print(build_graph().get_graph().draw_mermaid())

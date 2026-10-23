"""The reply to the owner, written from state by a template. No model here: every figure
and claim in the reply comes from a tool or the owner, so there is nothing to make up."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from renopilot import trace
from renopilot.tools.budget import fmt_sgd


def render(state: dict) -> str:
    p = state.get("style_profile")
    sugg = [s for s in state.get("suggestions") or [] if s.status != "rejected"]
    lines = []
    if p:
        lines.append(f"Your board reads as {', '.join(t.replace('_', ' ') for t in p.style_tags)}: {p.summary}")
    shown = [s for s in sugg if s.status == "approved"] or sugg
    by_type: dict[str, list] = {}
    for s in shown:
        by_type.setdefault(s.item_type, []).append(s)
    for t, items in by_type.items():
        lines.append(f"\n{t.replace('_', ' ').title()}")
        for s in items:
            flag = "" if not s.verdict or s.verdict.decision == "allow" else f"  [{s.verdict.decision}: {s.verdict.reason}]"
            if s.product_id:
                lines.append(f"  - {s.title[:80]} ({fmt_sgd(s.price_sgd)}) - {s.reason}{flag}\n    {s.url}")
            else:
                lines.append(f"  - {s.title}{flag}")
    # The shortlist offers alternatives: the owner buys ONE per item type, so never sum them all.
    priced = {t: [s for s in items if s.product_id and s.price_sgd is not None] for t, items in by_type.items()}
    priced = {t: v for t, v in priced.items() if v}
    if priced:
        first = sum(v[0].price_sgd for v in priced.values())
        cheapest = sum(min(s.price_sgd for s in v) for v in priced.values())
        line = (f"\nBuying one item per type: the first option for each comes to {fmt_sgd(first)}, "
                f"the cheapest options to {fmt_sgd(cheapest)} (listed prices, before delivery, shipping and GST).")
        project = state.get("project")
        if project is not None and project.total_budget:
            line += f" Your furniture budget is {fmt_sgd(project.total_budget)}."
        lines.append(line)
    for e in state.get("escalations") or []:
        lines.append(f"\nNote: {e}")
    if not state.get("approved") and sugg:
        lines.append("\nNothing has been approved yet. Nothing is bought or added to a cart; you decide.")
    return "\n".join(lines) or "I could not build a shortlist from this board."


def respond(state: dict) -> dict:
    text = render(state)
    trace.log("respond", "reply", chars=len(text), approved=bool(state.get("approved")))
    return {"messages": [AIMessage(text)],
            "trace": [{"node": "respond", "action": "replied", "why": f"{len(text)} chars"}]}

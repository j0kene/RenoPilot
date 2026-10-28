"""The reply to the owner, written from state by a template. No model here: every figure
and claim in the reply comes from a tool or the owner, so there is nothing to make up."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from renopilot import trace
from renopilot.compliance import PLAIN
from renopilot.nodes.guard import blocked
from renopilot.pricing.landed_cost import basket_total
from renopilot.tools.budget import fmt_sgd

SAMPLE = " [SAMPLE DATA, not a real offer]"


def _source(s) -> str:
    """Retailer of a suggestion, from its product id prefix (ikea_sg:..., castlery:..., taobao:...)."""
    return (s.product_id or "other_sg").split(":", 1)[0]


def _cost(s) -> float | None:
    return s.price_quote.total_sgd if s.price_quote else s.price_sgd


def _totals(priced: dict[str, list]) -> tuple[float, float, str]:
    """(first option per type, cheapest option per type, basis). The shortlist offers
    alternatives: the owner buys ONE per item type, so alternatives are never summed."""
    firsts = [v[0] for v in priced.values()]
    cheapest = [min(v, key=_cost) for v in priced.values()]
    if all(s.price_quote for s in firsts + cheapest):      # landed: each retailer's delivery charged once
        total = lambda picks: basket_total([(_source(s), s.price_quote) for s in picks])["total_sgd"]  # noqa: E731
        return total(firsts), total(cheapest), "landed in SGD, delivery charged once per retailer, GST included"
    return (sum(_cost(s) for s in firsts), sum(_cost(s) for s in cheapest),
            "listed prices, before delivery, shipping and GST")


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
            if not s.product_id:                       # a finish/built-in: the HDB route and its rule
                v = s.verdict
                how = f"\n    HDB: {PLAIN[v.decision]}. {v.route} ({v.rule_id}, {v.citation})" if v and v.rule_id else flag
                lines.append(f"  - {s.title}{how}" + "".join(f"\n      - {c}" for c in v.conditions if v))
                continue
            q = s.price_quote
            price = f"{fmt_sgd(q.total_sgd)} landed{' (est.)' if q.estimated else ''}" if q else fmt_sgd(s.price_sgd)
            lines.append(f"  - {s.title[:80]} ({price}) - {s.reason}{flag}\n    {s.url}")
            for a in s.comparables:
                if q and a.quote.total_sgd < q.total_sgd:
                    lines.append(f"      cheaper comparable: {a.title[:60]} ({a.source}), "
                                 f"{fmt_sgd(a.quote.total_sgd)} landed{SAMPLE if a.quote.synthetic else ''}")
    priced = {t: [s for s in items if s.product_id and _cost(s) is not None] for t, items in by_type.items()}
    priced = {t: v for t, v in priced.items() if v}
    if priced:
        first, cheapest, basis = _totals(priced)
        line = (f"\nBuying one item per type: the first option for each comes to {fmt_sgd(first)}, "
                f"the cheapest options to {fmt_sgd(cheapest)} ({basis}).")
        project = state.get("project")
        if project is not None and project.total_budget:
            line += f" Your furniture budget is {fmt_sgd(project.total_budget)}."
        lines.append(line)
    pending = [s for s in sugg if not s.product_id and s.status != "approved" and s.verdict]
    if pending:
        lines.append("\nWorks still requiring clarification (not cleared to proceed):")
        for s in pending:
            lines.append(f"  - {s.title} ({s.room or 'room unknown'}): {s.verdict.reason}")
    for b in blocked(state.get("suggestions")):
        lines.append(f"\nNot allowed under HDB rules: {b['title'] or b['item_type'].replace('_', ' ')}. {b['reason']}"
                     + (f" Instead: {b['alternative']}" if b["alternative"] else "") + f" ({b['rule_id']}, {b['citation']})")
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

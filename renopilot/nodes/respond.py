"""The reply to the owner, written from state by a template. No model here: every figure
and claim in the reply comes from a tool or the owner, so there is nothing to make up."""
from __future__ import annotations

import csv
import io

from langchain_core.messages import AIMessage

from renopilot import trace
from renopilot.basket import summary, work_totals
from renopilot.compliance import PLAIN
from renopilot.nodes.guard import blocked
from renopilot.tools.budget import fmt_sgd

def _cost(s) -> float | None:
    return s.price_quote.total_sgd if s.price_quote else s.price_sgd


def _totals(priced: dict[str, list]) -> tuple[float, float, str]:
    """Quantity-aware totals for the first and lowest offer per independent request."""
    firsts = [v[0] for v in priced.values()]
    cheapest = [min(v, key=_cost) for v in priced.values()]
    a, b = summary(firsts), summary(cheapest)
    return a["total_sgd"], b["total_sgd"], a["basis"]


def _room(s) -> str:
    return s.room_label or (s.room.replace("_", " ") if s.room else "")


def _request_label(s) -> str:
    return f"{s.item_type.replace('_', ' ').title()}" + (f" — {_room(s)}" if _room(s) else "")


def _works_summary(project, suggestions, furniture_total: float) -> list[str]:
    totals = work_totals(project, suggestions)
    active_works = [s for s in suggestions if not s.product_id and s.status != "rejected"]
    has_work = bool(active_works or totals["rows"] or project and project.works_budget is not None)
    if not has_work:
        return []

    known = any(r.total_sgd is not None for r in totals["rows"])
    lines = []
    if project and project.works_budget is not None:
        remaining = (f"; remaining {fmt_sgd(totals['remaining_sgd'])}" if totals["complete"] else
                     "; remaining not determined while work costs are incomplete")
        lines.append(f"Works budget: {fmt_sgd(project.works_budget)}{remaining}.")
    if totals["complete"]:
        works_label = f"Works subtotal: {fmt_sgd(totals['subtotal_sgd'])}."
        combined = furniture_total + totals["subtotal_sgd"]
        project_label = f"Project estimate: {fmt_sgd(combined)} (furniture and works)."
    else:
        works_label = (f"Works subtotal: {fmt_sgd(totals['subtotal_sgd'])} known, incomplete."
                       if known else "Works subtotal: incomplete; no verified total.")
        project_label = (f"Project estimate: {fmt_sgd(furniture_total + totals['subtotal_sgd'])} known costs only; incomplete."
                         if known else "Project estimate: incomplete; work costs are unknown.")
    lines.extend([works_label, project_label])

    for row in totals["rows"]:
        amount = fmt_sgd(row.total_sgd) if row.total_sgd is not None else "amount unknown"
        provenance = ", ".join(x for x in (row.source or "source unknown",
                                            row.checked_at.isoformat() if row.checked_at else "check date unknown") if x)
        lines.append(f"  Work estimate: {row.description} ({row.quantity:g} {row.unit}) — {amount}; {provenance}.")
    if totals["missing_requests"]:
        lines.append("  Work estimates missing for: " + ", ".join(totals["missing_requests"]) + ".")
    return lines


def render(state: dict) -> str:
    p = state.get("style_profile")
    sugg = [s for s in state.get("suggestions") or [] if s.status != "rejected"]
    lines = []
    if p:
        lines.append(f"Your board reads as {', '.join(t.replace('_', ' ') for t in p.style_tags)}: {p.summary}")
    shown = ([s for s in sugg if s.product_id and s.status == "approved"] if state.get("approved")
             else [s for s in sugg if s.product_id])
    by_type: dict[str, list] = {}
    for s in shown:
        by_type.setdefault(s.selection_key, []).append(s)
    for _, items in by_type.items():
        s0 = items[0]
        lines.append(f"\n{_request_label(s0)}")
        for s in items:
            count = f" × {s.quantity}" if s.quantity > 1 else ""
            flag = "" if not s.verdict or s.verdict.decision == "allow" else f"  [{s.verdict.decision}: {s.verdict.reason}]"
            q = s.price_quote
            price = f"{fmt_sgd(q.item_sgd)} each{count}" if q else (f"{fmt_sgd(s.price_sgd)} each{count}" if s.price_sgd is not None else "price unknown")
            if s.fit_note:
                lines.append(f"  - Fit: {s.fit_note}")
            lines.append(f"  - {s.title[:80]} ({price}) - {s.reason}{flag}\n    {s.url}")
            for a in s.comparables:
                if q and not a.quote.synthetic and a.quote.total_sgd < q.total_sgd:   # illustrations stay on the card
                    lines.append(f"      cheaper comparable: {a.title[:60]} ({a.source}), {fmt_sgd(a.quote.item_sgd)} each")
    priced = {t: [s for s in items if s.product_id and _cost(s) is not None] for t, items in by_type.items()}
    priced = {t: v for t, v in priced.items() if v}
    furniture_total = first = 0.0
    if priced:
        project = state.get("project")
        if state.get("approved"):
            totals = summary([s for items in priced.values() for s in items])
            furniture_total = totals["total_sgd"]
            line = f"\nSelected basket total: {fmt_sgd(furniture_total)} ({totals['basis']})."
            if project is not None and project.total_budget is not None:
                line += (f" Your furniture budget is {fmt_sgd(project.total_budget)}; "
                         f"remaining: {fmt_sgd(project.total_budget - furniture_total)}.")
        else:
            first, cheapest, basis = _totals(priced)
            furniture_total = first
            line = (f"\nBuying the requested quantities: the first offers total {fmt_sgd(first)}, "
                    f"the lowest offers total {fmt_sgd(cheapest)} ({basis}).")
        lines.append(line)
    lines.extend(_works_summary(state.get("project"), sugg, furniture_total))
    pending = [s for s in sugg if not s.product_id]
    if pending:
        lines.append("\nWork requests and conditions:")
        for s in pending:
            room = f" — {_room(s)}" if _room(s) else ""
            count = f" × {s.quantity}" if s.quantity > 1 else ""
            if s.verdict:
                v = s.verdict
                how = f" HDB: {PLAIN[v.decision]}. {v.route} ({v.rule_id}, {v.citation})." if v.rule_id else ""
                lines.append(f"  - {s.title}{room}{count}:{how} {v.reason}")
                lines.extend(f"      - {condition}" for condition in v.conditions)
                if s.status != "approved":
                    lines.append("      Pending: this work is not cleared to proceed.")
            else:
                lines.append(f"  - {s.title}{room}{count}: Not checked.")
    for b in blocked(state.get("suggestions")):
        lines.append(f"\nNot allowed under HDB rules: {b['title'] or b['item_type'].replace('_', ' ')}. {b['reason']}"
                     + (f" Instead: {b['alternative']}" if b["alternative"] else "") + f" ({b['rule_id']}, {b['citation']})")
    for e in state.get("escalations") or []:
        lines.append(f"\nNote: {e}")
    if not state.get("approved") and sugg:
        lines.append("\nNothing has been approved yet. Nothing is bought or added to a cart; you decide.")
    return "\n".join(lines) or "I could not build a shortlist from this board."


def shopping_list_csv(state: dict) -> str:
    """Export chosen offers and unresolved works, with the shared-delivery total made explicit."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)

    def write(row):
        # Retailer text remains text when the CSV is opened in a spreadsheet.
        writer.writerow(["'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v for v in row])

    write(["Category", "Room", "Item", "Status", "Item SGD", "Shipping SGD", "GST SGD",
                  "Delivery if bought alone SGD", "Total if bought alone SGD", "Estimated", "Link / HDB source", "Notes",
                  "Quantity", "Request ID", "Capture date", "Fit note"])
    suggestions = list(state.get("suggestions") or [])
    picks = []
    for s in suggestions:
        if s.product_id and s.status != "approved":
            continue
        q = s.price_quote
        if s.product_id:
            picks.append(s)
        write([s.item_type, _room(s), s.title, s.status,
                      q.item_sgd if q else "", q.shipping_sgd if q else "", q.gst_sgd if q else "",
                      q.local_delivery_sgd if q else "", q.total_sgd if q else "", q.estimated if q else "",
                      s.url if s.product_id else s.verdict.citation if s.verdict else "",
                      s.reason if s.product_id else s.verdict.reason if s.verdict else "Not checked",
                      s.quantity, s.request_id or s.selection_key, s.retrieved_at, s.fit_note])
    totals = summary(picks)
    write([])
    for source, fee in totals["delivery_by_source"].items():
        write(["DELIVERY", "", source, "", "", "", "", fee])
    project = state.get("project")
    works = work_totals(project, suggestions)
    estimates = list(works["rows"])
    active_works = [s for s in suggestions if not s.product_id and s.status != "rejected"]
    for s in active_works:
        if not any(r.request_id == s.key for r in estimates):
            write(["WORK ESTIMATE", _room(s), s.title, "missing", "", "", "", "", "", "",
                   "", "No estimate supplied", s.quantity, s.key, "", ""])
    for row in estimates:
        linked = next((s for s in active_works if row.request_id and s.key == row.request_id), None)
        notes = "; ".join(x for x in ("Source not supplied" if not row.source else "",
                                        "Check date not supplied" if not row.checked_at else "",
                                        "Rate not supplied" if row.unit_rate_sgd is None else "") if x)
        write(["WORK ESTIMATE", _room(linked) if linked else "", row.description, "estimate",
               row.unit_rate_sgd if row.unit_rate_sgd is not None else "", "", "", "",
               row.total_sgd if row.total_sgd is not None else "", "incomplete" if notes else "",
               row.source, notes, row.quantity, row.request_id, row.checked_at.isoformat() if row.checked_at else "", ""])

    furniture_total = totals["total_sgd"]
    known_work = any(r.total_sgd is not None for r in estimates)
    write(["BASKET TOTAL", "", "Furniture; delivery counted once per retailer", "", "", "", "", "", furniture_total])
    write(["WORKS SUBTOTAL", "", "Incomplete" if not works["complete"] else "Owner work estimates", "", "", "", "", "",
           works["subtotal_sgd"] if known_work else "", "incomplete" if not works["complete"] else ""])
    if project and project.works_budget is not None:
        write(["WORKS BUDGET", "", "", "", "", "", "", "", project.works_budget])
        write(["WORKS REMAINING", "", "Incomplete" if not works["complete"] else "", "", "", "", "", "",
               works["remaining_sgd"] if works["complete"] else ""])
    combined = (furniture_total + works["subtotal_sgd"]
                if works["complete"] or known_work else "")
    write(["PROJECT ESTIMATE", "", "" if works["complete"] else "INCOMPLETE; known costs only", "", "", "", "", "", combined])
    if project and project.total_budget is not None:
        write(["BUDGET", "", "", "", "", "", "", "", project.total_budget])
        write(["REMAINING", "", "", "", "", "", "", "", round(project.total_budget - furniture_total, 2)])
    return buf.getvalue()


def respond(state: dict) -> dict:
    text = render(state)
    trace.log("respond", "reply", chars=len(text), approved=bool(state.get("approved")))
    return {"messages": [AIMessage(text)],
            "trace": [{"node": "respond", "action": "replied", "why": f"{len(text)} chars"}]}

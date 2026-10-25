"""The two human gates. Approvals are runtime interrupts, not instructions in a prompt.

Course rule: nothing above interrupt() may have a side effect, because on resume the node
runs again from its first line. Both nodes only READ before the interrupt.

Resume values (what the UI / CLI sends back with Command(resume=...)):
  ask_user  {"answer": str, "budget_by_category": {item_type: SGD}?, "total_budget": SGD?,
             "board_images": [paths]?}
  approval  {"approve": "all"}  or  {"reject": [{"product_id": str, "reason": str}, ...]}
Only these resume values can change the budget: no model output can.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command, interrupt

from renopilot import trace
from renopilot.tools.budget import fmt_sgd


def ask_user(state: dict) -> dict:
    # Boundary messages are code, not model: an open issue is shown verbatim, and the
    # planner's own wording is only added as a follow-up.
    issues = state.get("open_issues") or []
    planner_q = (state.get("pending_question") or "").strip()
    if issues:
        question = issues[0] + (f" {planner_q}" if planner_q and planner_q not in issues[0] else "")
    else:
        question = planner_q or "Anything to change?"
    reply = interrupt({"type": "question", "question": question, "open_issues": issues})
    # ---- after resume: safe to write
    reply = reply if isinstance(reply, dict) else {"answer": str(reply)}
    update: dict = {"open_issues": [], "pending_question": None,
                    "messages": [AIMessage(question), HumanMessage(reply.get("answer", ""))],
                    "trace": [{"node": "ask_user", "action": "answered", "why": reply.get("answer", "")[:80]}]}
    if reply.get("board_images"):                      # the owner added or replaced images
        update["board_images"] = list(reply["board_images"])
    project = state.get("project")
    if project is not None and (reply.get("budget_by_category") or reply.get("total_budget") is not None):
        cats = {**project.budget_by_category, **(reply.get("budget_by_category") or {})}
        total = reply.get("total_budget", project.total_budget)
        update["project"] = project.model_copy(update={"budget_by_category": cats, "total_budget": total})
        changed = list(reply.get("budget_by_category") or {})
        profile = state.get("style_profile")
        if reply.get("total_budget") is not None and profile:
            changed = [i.item_type for i in profile.items]
        # a changed budget invalidates those types' shortlists
        update["redo_types"] = list(dict.fromkeys((state.get("redo_types") or []) + changed))
    trace.log("ask_user", "answered", question=question, reply=reply)
    return update


def _card(s) -> dict:
    q = s.price_quote
    return {"product_id": s.product_id, "item_type": s.item_type, "title": s.title[:90],
            "price": fmt_sgd(s.price_sgd), "reason": s.reason, "url": s.url, "image_path": s.image_path,
            "verdict": s.verdict.decision if s.verdict else None,
            "verdict_reason": s.verdict.reason if s.verdict else None,
            # pricing (None until the pricing node has run)
            "landed": fmt_sgd(q.total_sgd) if q else None,
            "landed_lines": [f"{ln.label}: {fmt_sgd(ln.amount_sgd)}" + (f" ({ln.note})" if ln.note else "")
                             for ln in q.lines] if q else [],
            "estimated": bool(q and q.estimated), "assumptions": q.assumptions if q else [],
            "comparables": [{"title": a.title[:80], "source": a.source, "landed": fmt_sgd(a.quote.total_sgd),
                             "cheaper": bool(q and a.quote.total_sgd < q.total_sgd), "sample": a.quote.synthetic,
                             "estimated": a.quote.estimated, "why": a.why_comparable, "url": a.url}
                            for a in s.comparables]}


def approval(state: dict) -> Command:
    live = [s for s in state.get("suggestions") or [] if s.status != "rejected"]
    decision = interrupt({"type": "approval", "suggestions": [_card(s) for s in live],
                          "escalations": state.get("escalations") or []})
    # ---- after resume
    decision = decision if isinstance(decision, dict) else {"approve": "all"}
    rejects = {r["product_id"]: r.get("reason", "") for r in decision.get("reject", []) if r.get("product_id")}
    by_id = {s.product_id: s for s in live if s.product_id}
    unknown = [pid for pid in rejects if pid not in by_id]
    rejected_rows, out = [], []
    for s in state.get("suggestions") or []:
        if s.product_id in rejects and s.product_id in by_id:
            out.append(s.model_copy(update={"status": "rejected"}))
            rejected_rows.append({"product_id": s.product_id, "item_type": s.item_type,
                                  "reason": rejects[s.product_id], "by": "user"})
        elif s.status != "rejected":
            out.append(s.model_copy(update={"status": "approved"}) if not rejects else s)
        else:
            out.append(s)
    trace.log("approval", "decision", n_rejected=len(rejected_rows), unknown_ids=unknown)
    if not rejected_rows:
        out = [s.model_copy(update={"status": "approved"}) if s.status != "rejected" else s for s in out]
        return Command(goto="respond", update={"suggestions": out, "approved": True,
                                               "trace": [{"node": "approval", "action": "approved", "why": "owner"}]})
    redo = list(dict.fromkeys(r["item_type"] for r in rejected_rows))
    return Command(goto="planner", update={
        "suggestions": out, "rejected": rejected_rows, "approved": False, "redo_types": redo,
        "messages": [HumanMessage("Rejected: " + "; ".join(f"{r['item_type']} ({r['reason'] or 'no reason'})"
                                                          for r in rejected_rows))],
        "trace": [{"node": "approval", "action": "rejected", "why": ", ".join(redo)}]})

"""The two human gates. Approvals are runtime interrupts, not instructions in a prompt.

Course rule: nothing above interrupt() may have a side effect, because on resume the node
runs again from its first line. Both nodes only READ before the interrupt.

Resume values (what the UI / CLI sends back with Command(resume=...)):
  ask_user  {"answer": str, "budget_by_category": {item_type: SGD}?, "total_budget": SGD?,
             "board_images": [paths]?}
  approval  {"approve": "selected", "selected": {item_type: product_id}} or reject rows.
            "approve": "all" is a CLI shortcut to one default offer per category.
            "routes": {suggestion.key: rule_id} chooses a method for unspecified work;
            "alternatives": {blocked_suggestion.key: rule_id} proposes a separate checked alternative.
            Budget/date changes return to the planner and invalidate affected shortlists.
Only these resume values can change the budget: no model output can.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command, interrupt

from renopilot import trace
from renopilot.basket import defaults, selection
from renopilot.compliance import rule
from renopilot.nodes.guard import blocked, requested_method, verdict_for, with_route
from renopilot.schemas import ProjectProfile
from renopilot.tools.budget import caps_for, fmt_sgd
from renopilot.vocab import WORK_TYPE_OF


def _stale_types(old: ProjectProfile, new: ProjectProfile, profile) -> list[str]:
    """What a project change invalidates: a new search where a cap moved, and a new
    check of finishes and built-ins where the completion date moved."""
    types = list(dict.fromkeys(i.item_type for i in profile.items))
    furniture = [t for t in types if WORK_TYPE_OF[t] == "loose_furniture"]
    was, now = caps_for(old, furniture), caps_for(new, furniture)
    return [t for t in types if (was[t] != now[t] if t in furniture else old.completion_date != new.completion_date)]


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
    update: dict = {"open_issues": [], "pending_question": None, "steps": 0,
                    "messages": [AIMessage(question), HumanMessage(reply.get("answer", ""))],
                    "trace": [{"node": "ask_user", "action": "answered", "why": reply.get("answer", "")[:80]}]}
    if reply.get("board_images"):                      # the owner added or replaced images
        update["board_images"] = list(reply["board_images"])
    project = state.get("project")
    if project is not None and any(k in reply for k in ("budget_by_category", "total_budget", "completion_date")):
        cats = {**project.budget_by_category, **(reply.get("budget_by_category") or {})}
        total = reply.get("total_budget", project.total_budget)
        try:
            update["project"] = ProjectProfile.model_validate(project.model_dump() | {
                "budget_by_category": cats, "total_budget": total,
                "completion_date": reply.get("completion_date", project.completion_date)})
        except ValueError:
            update["open_issues"] = ["Enter valid, non-negative SGD budgets and a valid completion date."]
            return update
        profile = state.get("style_profile")
        if profile:
            update["redo_types"] = list(dict.fromkeys((state.get("redo_types") or []) +
                                                      _stale_types(project, update["project"], profile)))
    trace.log("ask_user", "answered", question=question, reply=reply)
    return update


def _card(s) -> dict:
    q = s.price_quote
    return {"key": s.key, "product_id": s.product_id, "item_type": s.item_type, "room": s.room,
            "work_method": requested_method(s), "title": s.title[:90],
            "price": fmt_sgd(s.price_sgd), "reason": s.reason, "url": s.url, "image_path": s.image_path,
            "verdict": s.verdict.decision if s.verdict else None,
            "verdict_reason": s.verdict.reason if s.verdict else None,
            "rule_id": s.verdict.rule_id if s.verdict else None,
            "routes": [o.model_dump(exclude={"options"}) for o in s.verdict.options] if s.verdict else [],
            # pricing (None until the pricing node has run)
            "landed": fmt_sgd(q.total_sgd) if q else None,
            "landed_lines": [f"{ln.label}: {fmt_sgd(ln.amount_sgd)}" + (f" ({ln.note})" if ln.note else "")
                             for ln in q.lines] if q else [],
            "estimated": bool(q and q.estimated), "assumptions": q.assumptions if q else [],
            "comparables": [{"title": a.title[:80], "source": a.source, "landed": fmt_sgd(a.quote.total_sgd),
                             "cheaper": bool(q and not a.quote.synthetic and a.quote.total_sgd < q.total_sgd), "sample": a.quote.synthetic,
                             "estimated": a.quote.estimated, "why": a.why_comparable, "url": a.url}
                            for a in s.comparables]}


def approval(state: dict) -> Command:
    live = [s for s in state.get("suggestions") or [] if s.status != "rejected"]
    decision = interrupt({"type": "approval", "suggestions": [_card(s) for s in live],
                          "blocked": blocked(state.get("suggestions")), "escalations": state.get("escalations") or [],
                          "selected": state.get("selected") or defaults(live, state.get("project")),
                          "error": state.get("approval_error")})
    # ---- after resume
    if not isinstance(decision, dict):
        return Command(goto="approval", update={"approval_error": "Choose offers and explicitly approve the basket."})
    if any(k in decision for k in ("budget_by_category", "total_budget", "completion_date")):
        project = state.get("project") or ProjectProfile()
        try:
            updated = ProjectProfile.model_validate(project.model_dump() | {
                "budget_by_category": project.budget_by_category | decision.get("budget_by_category", {}),
                "total_budget": decision.get("total_budget", project.total_budget),
                "completion_date": decision.get("completion_date", project.completion_date)})
        except (ValueError, TypeError):
            return Command(goto="approval", update={"approval_error": "Enter valid, finite, non-negative SGD budgets."})
        profile = state.get("style_profile")
        return Command(goto="planner", update={"project": updated, "selected": {}, "approval_error": None,
            "approved": False, "steps": 0, "open_issues": [],
            "redo_types": list(dict.fromkeys(i.item_type for i in profile.items)) if profile else []})
    if decision.get("alternatives"):
        alternatives = []
        for s in state.get("suggestions") or []:
            rid = decision["alternatives"].get(s.key)
            if not rid or s.status != "rejected" or not s.verdict or s.verdict.decision != "block":
                continue
            v = next((o for o in s.verdict.options if o.rule_id == rid and
                      o.decision in ("allow", "allow_with_permit")), None)
            if v:
                method = rule(rid)["method"]
                alternatives.append(s.model_copy(update={"title": f"Alternative: {v.route}",
                    "work_method": method if method != "any" else "unspecified", "structural": None,
                    "status": "proposed", "verdict": None, "reason": f"Owner chose an alternative to: {s.title}"}))
        if not alternatives:
            return Command(goto="approval", update={"approval_error": "Choose a permitted alternative offered by the guard."})
        keys = {s.key for s in alternatives}
        return Command(goto="compliance_guard", update={"suggestions": [s for s in state["suggestions"]
            if s.key not in keys or s.status == "rejected"] + alternatives, "approved": False,
            "approval_error": None, "trace": [{"node": "approval", "action": "alternative", "why": ", ".join(keys)}]})
    chosen = decision.get("routes") or {}
    suggestions = []
    for s in state.get("suggestions") or []:
        # Legacy type-only route keys are accepted only when the work is unambiguous.
        type_count = sum(x.item_type == s.item_type and not x.product_id and x.status != "rejected" for x in live)
        rid = chosen.get(s.key) or (chosen.get(s.item_type) if type_count == 1 else None)
        if rid and s.status != "rejected":
            v = with_route(s.verdict, rid)
            method = rule(rid)["method"] if v and v.rule_id == rid else None
            if method is None or (requested_method(s) != "unspecified" and method not in (requested_method(s), "any")):
                return Command(goto="approval", update={"approval_error": "A different method must be proposed as a separate alternative."})
            s = s.model_copy(update={"verdict": v, "work_method": method if method != "any" else "unspecified"})
            checked = verdict_for(s, state.get("project"), s.room)
            if checked.decision not in ("allow", "allow_with_permit"):
                return Command(goto="approval", update={"approval_error": checked.reason})
            s = s.model_copy(update={"verdict": checked})
        suggestions.append(s)
    rejects = {r["product_id"]: r.get("reason", "") for r in decision.get("reject", []) if r.get("product_id")}
    by_id = {s.product_id: s for s in live if s.product_id}
    unknown = [pid for pid in rejects if pid not in by_id]
    if unknown:
        return Command(goto="approval", update={"approval_error": "A rejected offer is no longer on this shortlist."})
    rejected_rows, out = [], []
    for s in suggestions:
        if s.product_id in rejects and s.product_id in by_id:
            out.append(s.model_copy(update={"status": "rejected"}))
            rejected_rows.append({"product_id": s.product_id, "item_type": s.item_type,
                                  "reason": rejects[s.product_id], "by": "user"})
        else:
            out.append(s)
    trace.log("approval", "decision", n_rejected=len(rejected_rows), unknown_ids=unknown, routes=chosen)
    if not rejected_rows:
        if not decision.get("approve"):
            return Command(goto="approval", update={"approval_error": "Select an offer to reject, or explicitly approve your selection."})
        selected = decision.get("selected", defaults(out, state.get("project")))
        try:
            picks, _ = selection(out, selected, state.get("project"), require_quotes=state.get("pricing_enabled", True))
        except (ValueError, TypeError) as e:
            return Command(goto="approval", update={"approval_error": str(e), "selected": selected, "approved": False})
        selected_ids = {s.product_id for s in picks}
        out = [s.model_copy(update={"status": "approved" if (s.product_id in selected_ids if s.product_id else
               bool(s.verdict and s.verdict.decision in ("allow", "allow_with_permit"))) else "proposed"})
               if s.status != "rejected" else s for s in out]
        present = {s.product_id for s in out}
        out += [s.model_copy(update={"status": "approved"}) for s in picks if s.product_id not in present]
        return Command(goto="respond", update={"suggestions": out, "approved": True, "selected": selected,
                                               "approval_error": None,
                                               "trace": [{"node": "approval", "action": "approved", "why": "owner"}]})
    redo = list(dict.fromkeys(r["item_type"] for r in rejected_rows))
    return Command(goto="planner", update={
        "suggestions": out, "rejected": rejected_rows, "approved": False, "redo_types": redo,
        "selected": {}, "approval_error": None, "steps": 0,
        "messages": [HumanMessage("Rejected: " + "; ".join(f"{r['item_type']} ({r['reason'] or 'no reason'})"
                                                          for r in rejected_rows))],
        "trace": [{"node": "approval", "action": "rejected", "why": ", ".join(redo)}]})

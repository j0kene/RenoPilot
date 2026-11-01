"""The two human gates. Approvals are runtime interrupts, not instructions in a prompt.

Course rule: nothing above interrupt() may have a side effect, because on resume the node
runs again from its first line. Both nodes only READ before the interrupt.

Resume values (what the UI / CLI sends back with Command(resume=...)):
  ask_user  {"answer": str, "budget_by_category": {item_type: SGD}?, "total_budget": SGD?,
             "completion_date": "YYYY-MM-DD"?, "board_images": [paths]?, "notes": str?}
  approval  {"approve": "selected", "selected": {item_type: product_id}} or reject rows.
            "approve": "all" is a CLI shortcut to one default offer per category.
            "routes": {suggestion.key: rule_id} chooses a method for unspecified work;
            "alternatives": {blocked_suggestion.key: rule_id} proposes a separate checked alternative.
            Budget/date changes return to the planner and invalidate affected shortlists.
Only these resume values can change the budget: no model output can.
"""
from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command, Overwrite, interrupt

from renopilot import trace
from renopilot.basket import defaults, selection
from renopilot.compliance import rule
from renopilot.nodes.guard import blocked, request_from, requested_method, verdict_for, with_route
from renopilot.schemas import ItemSpec, ProjectProfile, StyleProfile
from renopilot.tools.budget import fmt_sgd, request_caps
from renopilot.vocab import WORK_TYPE_OF


def _stale_types(old: ProjectProfile, new: ProjectProfile, profile) -> list[str]:
    """What a project change invalidates: a new search where a cap moved, and a new
    check of finishes and built-ins where the completion date moved."""
    types = list(dict.fromkeys(i.item_type for i in profile.items))
    furniture = [i for i in profile.items if WORK_TYPE_OF[i.item_type] == "loose_furniture"]
    was, now = request_caps(old, furniture), request_caps(new, furniture)
    changed = {i.item_type for i in furniture
               if was.get(i.request_id or i.item_type) != now.get(i.request_id or i.item_type)}
    if old.completion_date != new.completion_date:
        changed.update(t for t in types if WORK_TYPE_OF[t] != "loose_furniture")
    return [t for t in types if t in changed]


def _invalidate_derived() -> dict:
    """Clear all work derived from an old board/brief, including append-only reducer fields."""
    return {"style_profile": None, "suggestions": [], "searched_types": [], "shortlist_types": [],
            "redo_types": [], "refinements": {}, "refine_requests": [], "selected": {}, "approved": False,
            "rejected": Overwrite([]), "escalations": Overwrite([]), "approval_error": None}


def _reallocated_types(old_profile, new_profile, project) -> set[str]:
    old = request_caps(project, [i for i in old_profile.items if WORK_TYPE_OF[i.item_type] == "loose_furniture"])
    furniture = [i for i in new_profile.items if WORK_TYPE_OF[i.item_type] == "loose_furniture"]
    new = request_caps(project, furniture)
    return {i.item_type for i in furniture if old.get(i.request_id or i.item_type) != new.get(i.request_id or i.item_type)}


def _apply_work_updates(state: dict, updates: dict) -> tuple[object, list, list[str], list[str]]:
    """Validate owner edits against ItemSpec, update the original profile, and reopen requests for guard."""
    if not isinstance(updates, dict) or not updates:
        raise ValueError("Edit at least one renovation request.")
    profile = state.get("style_profile")
    if profile is None:
        raise ValueError("There is no current style profile to edit.")
    specs = list(profile.items)
    suggestions = list(state.get("suggestions") or [])
    fields = set(ItemSpec.model_fields) - {"request_id"}
    def request_key(spec):
        return getattr(spec, "request_id", "") or request_from(spec).key

    redo = set()
    invalidate_estimates = set()
    for key, patch in updates.items():
        if not isinstance(patch, dict) or set(patch) - fields:
            raise ValueError("Request edits must use valid work detail fields.")
        source_matches = [spec for spec in specs if request_key(spec) == key]
        source = source_matches[0] if len(source_matches) == 1 else None
        target = next((s for s in suggestions if s.key == key or getattr(s, "request_id", "") == key), None)
        if target is None and source is not None:
            candidates = [s for s in suggestions if getattr(s, "request_id", "") == key]
            if not candidates:
                candidates = [s for s in suggestions if s.item_type == source.item_type and s.room == source.room]
            target = candidates[0] if candidates and len({request_key(spec) for spec in specs
                if spec.item_type == source.item_type and spec.room == source.room}) == 1 else None
        if target is None and source is None:
            raise ValueError("A request you edited is no longer on this shortlist.")
        target_id = getattr(target, "request_id", "") or key
        matches = ([specs.index(source)] if source is not None else [n for n, spec in enumerate(specs) if
                   (bool(getattr(spec, "request_id", "")) and getattr(spec, "request_id", "") == target_id) or
                   request_key(spec) == key or request_from(spec).key == key])
        if len(matches) != 1:
            raise ValueError("The edited request no longer maps to one original work item.")
        ix = matches[0]
        changes = dict(patch)
        if isinstance(changes.get("work_details"), dict):
            changes["work_details"] = specs[ix].work_details.model_dump() | changes["work_details"]
        old_spec = specs[ix]
        spec = ItemSpec.model_validate(old_spec.model_dump() | changes)
        if spec.item_type != old_spec.item_type:
            raise ValueError("Add a separate request to change the item or work type.")
        if old_spec.model_dump() != spec.model_dump() and (
                WORK_TYPE_OF[old_spec.item_type] != "loose_furniture" or
                WORK_TYPE_OF[spec.item_type] != "loose_furniture"):
            invalidate_estimates.add(request_key(old_spec))
        specs[ix] = spec
        request_id = getattr(spec, "request_id", "")
        old_key = target.key if target else key
        if WORK_TYPE_OF[spec.item_type] == "loose_furniture":
            redo.add(spec.item_type)
        for n, suggestion in enumerate(suggestions):
            same = ((bool(request_id) and getattr(suggestion, "request_id", "") == request_id) or
                    suggestion.key in (old_key, key) or
                    (source is not None and not source.request_id and not suggestion.request_id and
                     WORK_TYPE_OF[source.item_type] == "loose_furniture" and
                     suggestion.item_type == source.item_type and suggestion.room == source.room))
            if same:
                if suggestion.product_id and WORK_TYPE_OF[spec.item_type] == "loose_furniture":
                    suggestions[n] = suggestion.model_copy(update={"status": "proposed"})
                    continue
                suggestions[n] = suggestion.model_copy(update={
                    "request_id": spec.request_id, "item_type": spec.item_type, "title": spec.description, "room": spec.room,
                    "work_method": spec.work_method, "structural": spec.structural,
                    "work_details": spec.work_details, "room_label": spec.room_label,
                    "quantity": spec.quantity, "fit_note": "",
                    "status": "proposed", "verdict": None})
    updated = profile.model_copy(update={"items": specs})
    redo.update(_reallocated_types(profile, updated, state.get("project")))
    return updated, suggestions, sorted(redo), sorted(invalidate_estimates)


def apply_routes(suggestions: list, routes: dict, project: ProjectProfile | None) -> list:
    """Apply owner route choices and return only choices that pass the guard again.

    Invalid, blocked, method-changing, or explicitly requested-method overrides raise ValueError.
    """
    if not isinstance(routes, dict):
        raise ValueError("Routes must be a mapping of suggestion keys to permitted rule ids.")
    live = [s for s in suggestions if s.status != "rejected"]
    out = []
    for s in suggestions:
        type_count = sum(x.item_type == s.item_type and not x.product_id for x in live)
        rid = routes.get(s.key) or (routes.get(s.item_type) if type_count == 1 else None)
        if rid and s.status != "rejected":
            v = with_route(s.verdict, rid)
            method = rule(rid)["method"] if v and v.rule_id == rid else None
            if method is None or (requested_method(s) != "unspecified" and method not in (requested_method(s), "any")):
                raise ValueError("A different method must be proposed as a separate alternative.")
            s = s.model_copy(update={"verdict": v, "work_method": method if method != "any" else "unspecified"})
            checked = verdict_for(s, project, s.room)
            if checked.decision not in ("allow", "allow_with_permit"):
                raise ValueError(checked.reason)
            s = s.model_copy(update={"verdict": checked})
        out.append(s)
    return out


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
    correction = "board_images" in reply or "notes" in reply or "brief" in reply
    if "board_images" in reply:
        update["board_images"] = list(reply["board_images"] or [])
    project = state.get("project")
    project_keys = ("budget_by_category", "total_budget", "completion_date", "works_budget", "work_estimates")
    if correction or any(k in reply for k in project_keys):
        project = project or ProjectProfile()
        changes = {"notes": reply.get("notes", reply.get("brief", project.notes))}
        if any(k in reply for k in project_keys):
            cats = {**project.budget_by_category, **(reply.get("budget_by_category") or {})}
            changes.update({"budget_by_category": cats, "total_budget": reply.get("total_budget", project.total_budget),
                            "completion_date": reply.get("completion_date", project.completion_date)})
            for key in ("works_budget", "work_estimates"):
                if key in reply:
                    changes[key] = reply[key]
        if correction:
            changes["work_estimates"] = []
        try:
            update["project"] = ProjectProfile.model_validate(project.model_dump() | changes)
        except ValueError:
            update["open_issues"] = ["Enter valid, non-negative SGD budgets and a valid completion date."]
            return update
        profile = state.get("style_profile")
        if correction:
            update.update(_invalidate_derived())
        elif profile:
            update["redo_types"] = list(dict.fromkeys((state.get("redo_types") or []) +
                                                      _stale_types(project, update["project"], profile)))
    trace.log("ask_user", "answered", question=question, reply=reply)
    return update


def _card(s) -> dict:
    q = s.price_quote
    return {"key": s.key, "selection_key": s.selection_key, "request_id": s.request_id,
            "product_id": s.product_id, "item_type": s.item_type, "room": s.room,
            "room_label": s.room_label, "quantity": s.quantity, "fit_note": s.fit_note,
            "work_method": requested_method(s), "title": s.title[:90],
            "retrieved_at": s.retrieved_at, "dims": s.dims, "material": s.material,
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
    if "board_images" in decision or "notes" in decision or "brief" in decision:
        project = state.get("project") or ProjectProfile()
        try:
            updated = ProjectProfile.model_validate(project.model_dump() | {
                "notes": decision.get("notes", decision.get("brief", project.notes)), "work_estimates": []})
        except ValueError:
            return Command(goto="approval", update={"approval_error": "Enter a valid brief."})
        update = _invalidate_derived() | {"project": updated, "steps": 0, "open_issues": [],
            "trace": [{"node": "approval", "action": "replacement", "why": "board or brief changed"}]}
        if "board_images" in decision:
            update["board_images"] = list(decision["board_images"] or [])
        return Command(goto="planner", update=update)
    if "work_updates" in decision:
        try:
            profile, edited, redo, invalidate_estimates = _apply_work_updates(state, decision["work_updates"])
        except (ValueError, TypeError) as e:
            return Command(goto="approval", update={"approval_error": str(e)})
        edited_keys = set(decision["work_updates"])
        edited_types = {i.item_type for i in state["style_profile"].items
                        if (i.request_id or request_from(i).key) in edited_keys}
        preserved_rejections = [r for r in state.get("rejected") or [] if
            (r.get("request_id") not in edited_keys if r.get("request_id") else r.get("item_type") not in edited_types)]
        update = {"style_profile": profile, "suggestions": edited, "rejected": Overwrite(preserved_rejections),
            "selected": {}, "approved": False, "approval_error": None, "steps": 0,
            "redo_types": list(dict.fromkeys((state.get("redo_types") or []) + redo)),
            "trace": [{"node": "approval", "action": "work_updated", "why": ", ".join(decision["work_updates"])}]}
        project = state.get("project")
        if project and invalidate_estimates:
            update["project"] = project.model_copy(update={"work_estimates": [
                estimate.model_copy(update={"unit_rate_sgd": None, "checked_at": None})
                if estimate.request_id in invalidate_estimates else estimate
                for estimate in project.work_estimates]})
        return Command(goto="planner" if redo else "compliance_guard", update=update)
    if "add_requests" in decision:
        profile = state.get("style_profile")
        if profile is None or not isinstance(decision["add_requests"], list) or not decision["add_requests"]:
            return Command(goto="approval", update={"approval_error": "Add at least one valid work or furniture request."})
        try:
            added = [ItemSpec.model_validate(value).model_copy(update={"request_id": uuid4().hex})
                     for value in decision["add_requests"]]
            updated_profile = StyleProfile.model_validate(profile.model_dump() | {"items": profile.items + added})
        except (ValueError, TypeError):
            return Command(goto="approval", update={"approval_error": "One or more added requests are invalid."})
        furniture = sorted({s.item_type for s in added if WORK_TYPE_OF[s.item_type] == "loose_furniture"} |
                           _reallocated_types(profile, updated_profile, state.get("project")))
        work = [request_from(s) for s in added if WORK_TYPE_OF[s.item_type] != "loose_furniture"]
        update = {"style_profile": updated_profile, "suggestions": list(state.get("suggestions") or []) + work,
            "selected": {}, "approved": False, "approval_error": None, "steps": 0,
            "redo_types": list(dict.fromkeys((state.get("redo_types") or []) + furniture)),
            "trace": [{"node": "approval", "action": "requests_added", "why": str(len(added))}]}
        return Command(goto="planner" if furniture else "compliance_guard", update=update)
    project_keys = ("budget_by_category", "total_budget", "completion_date", "works_budget", "work_estimates")
    if any(k in decision for k in project_keys):
        project = state.get("project") or ProjectProfile()
        try:
            updated = ProjectProfile.model_validate(project.model_dump() | {
                "budget_by_category": project.budget_by_category | decision.get("budget_by_category", {}),
                "total_budget": decision.get("total_budget", project.total_budget),
                "completion_date": decision.get("completion_date", project.completion_date),
                "works_budget": decision.get("works_budget", project.works_budget),
                "work_estimates": decision.get("work_estimates", project.work_estimates)})
        except (ValueError, TypeError):
            return Command(goto="approval", update={"approval_error": "Enter valid, finite, non-negative SGD budgets and work estimates."})
        if not any(k in decision for k in ("budget_by_category", "total_budget", "completion_date")):
            return Command(goto="approval", update={"project": updated, "approved": False,
                "approval_error": None, "trace": [{"node": "approval", "action": "work_costs_updated", "why": "owner"}]})
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
                    "request_id": f"{s.request_id or s.key}:alternative:{rid}",
                    "work_method": method if method != "any" else "unspecified", "structural": None,
                    "status": "proposed", "verdict": None, "reason": f"Owner chose an alternative to: {s.title}"}))
        if not alternatives:
            return Command(goto="approval", update={"approval_error": "Choose a permitted alternative offered by the guard."})
        keys = {s.key for s in alternatives}
        return Command(goto="compliance_guard", update={"suggestions": [s for s in state["suggestions"]
            if s.key not in keys or s.status == "rejected"] + alternatives, "approved": False,
            "approval_error": None, "trace": [{"node": "approval", "action": "alternative", "why": ", ".join(keys)}]})
    chosen = decision.get("routes") or {}
    try:
        suggestions = apply_routes(state.get("suggestions") or [], chosen, state.get("project"))
    except (ValueError, TypeError, KeyError) as e:
        return Command(goto="approval", update={"approval_error": str(e)})
    rejects = {(r.get("request_id") or None, r["product_id"]): r.get("reason", "")
               for r in decision.get("reject", []) if r.get("product_id")}
    by_id = {s.product_id for s in live if s.product_id}
    by_request = {(s.request_id, s.product_id) for s in live if s.product_id and s.request_id}
    unknown = [(rid, pid) for rid, pid in rejects if
               (pid not in by_id if rid is None else (rid, pid) not in by_request)]
    if unknown:
        return Command(goto="approval", update={"approval_error": "A rejected offer is no longer on this shortlist."})
    rejected_rows, out = [], []
    for s in suggestions:
        reason = rejects.get((s.request_id or None, s.product_id))
        if reason is None:
            reason = rejects.get((None, s.product_id))
        if reason is not None and s.product_id in by_id:
            out.append(s.model_copy(update={"status": "rejected"}))
            rejected_rows.append({"product_id": s.product_id, "request_id": s.request_id,
                                  "item_type": s.item_type, "reason": reason, "by": "user"})
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
        selected_ids = {(s.selection_key, s.product_id) for s in picks}
        out = [s.model_copy(update={"status": "approved" if ((s.selection_key, s.product_id) in selected_ids if s.product_id else
               bool(s.verdict and s.verdict.decision in ("allow", "allow_with_permit") and
                    not any(c.startswith("Pending verification:") for c in s.verdict.conditions))) else "proposed"})
               if s.status != "rejected" else s for s in out]
        present = {(s.selection_key, s.product_id) for s in out}
        out += [s.model_copy(update={"status": "approved"}) for s in picks
                if (s.selection_key, s.product_id) not in present]
        return Command(goto="respond", update={"suggestions": out, "approved": True, "selected": selected,
                                               "approval_error": None,
                                               "trace": [{"node": "approval", "action": "approved", "why": "owner"}]})
    redo = list(dict.fromkeys(r["item_type"] for r in rejected_rows))
    return Command(goto="planner", update={
        "suggestions": out, "rejected": rejected_rows, "approved": False, "redo_types": redo,
        "refine_requests": [r["request_id"] for r in rejected_rows if r["request_id"]],
        "selected": {}, "approval_error": None, "steps": 0,
        "messages": [HumanMessage("Rejected: " + "; ".join(f"{r['item_type']} ({r['reason'] or 'no reason'})"
                                                          for r in rejected_rows))],
        "trace": [{"node": "approval", "action": "rejected", "why": ", ".join(redo)}]})

"""Compliance guard: every suggestion gets a Verdict from the HDB rule cards (renopilot/compliance.py,
data/compliance/hdb_rules.jsonl), each citing the HDB page it rests on.

Most finishes can be done more than one way, and the ways differ under HDB's rules: overlaying
floor tiles needs no permit, hacking them out does, and replacing bathroom tiles in the block's
first 3 years is not allowed at all. The guard does not guess which way the owner means. It
lists every route with its own verdict (Verdict.options, easiest first) and puts the easiest
legal one in effect; the owner can switch to another listed route at the approval gate, and a
re-check (after a rejection round) keeps the route in effect if it is still legal. Facts
the guard knows narrow the list: the board's room, and the completion date if the owner gave
one (otherwise the two block-age routes are both listed, and picking one states the fact).

Contract:
  - reads  state["suggestions"], state["project"] (completion date), state["style_profile"] (room)
  - writes a Verdict on EVERY suggestion; blocked ones are marked rejected (by "guard") and are
    still shown to the owner with the reason and the legal alternative (`blocked`)
  - runs on every path to the owner: it is a graph edge, not a planner choice, so no agent
    can skip it, and nothing downstream may override a verdict
"""
from __future__ import annotations

from renopilot import trace
from renopilot.compliance import routes, rule
from renopilot.schemas import ProjectProfile, Suggestion, Verdict


def _verdict(card: dict) -> Verdict:
    how = card["when"] if card["method"] == "any" else f"{card['method'].capitalize()}: {card['when']}"
    return Verdict(decision=card["decision"], reason=card["summary"], rule_id=card["id"], route=how,
                   conditions=card["conditions"], alternative=card["alternative"], citation=card["url"])


def with_route(v: Verdict | None, rule_id: str | None) -> Verdict | None:
    """v with the route `rule_id` in effect, if it is one of v's own routes and not blocked; else v."""
    pick = next((o for o in (v.options if v else []) if o.rule_id == rule_id and o.decision != "block"), None)
    return pick.model_copy(update={"options": v.options}) if pick else v


@trace.tool("hdb_rule_check", inputs=("s", "project", "room"))
def verdict_for(s: Suggestion, project: ProjectProfile | None, room: str | None = None) -> Verdict:
    if s.work_type == "loose_furniture":
        return Verdict(decision="allow", reason="Loose furniture: no HDB approval needed.")
    cards = routes(s.item_type, room, project.completion_date if project else None) or [rule("HDB-NOT-LISTED")]
    options = [_verdict(c) for c in cards]
    return options[0].model_copy(update={"options": options})     # the easiest route; all blocked -> block


def blocked(suggestions) -> list[dict]:
    """What the guard blocked, for the owner: never dropped silently (approval gate and reply)."""
    return [{"item_type": s.item_type, "title": s.title, "reason": s.verdict.reason, "rule_id": s.verdict.rule_id,
             "alternative": s.verdict.alternative, "citation": s.verdict.citation}
            for s in suggestions or [] if s.status == "rejected" and s.verdict and s.verdict.decision == "block"]


def compliance_guard(state: dict) -> dict:
    project, profile = state.get("project"), state.get("style_profile")
    room = profile.room if profile else None
    out, blocked = [], []
    for s in state.get("suggestions") or []:
        if s.status == "rejected":
            out.append(s)
            continue
        # a route already in effect (the owner's pick at an earlier approval) stays, if still legal
        v = with_route(verdict_for(s, project, room), s.verdict.rule_id if s.verdict else None)
        s = s.model_copy(update={"verdict": v, "status": "rejected" if v.decision == "block" else s.status})
        if v.decision == "block":
            blocked.append({"product_id": s.product_id, "item_type": s.item_type, "reason": v.reason, "by": "guard"})
        out.append(s)
    counts = {}
    for s in out:
        if s.verdict:
            counts[s.verdict.decision] = counts.get(s.verdict.decision, 0) + 1
    trace.log("compliance_guard", "result", **counts)
    return {"suggestions": out, "rejected": blocked,
            "trace": [{"node": "compliance_guard", "action": "checked", "why": str(counts)}]}

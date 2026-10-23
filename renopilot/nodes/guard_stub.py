"""Compliance guard: STUB with the final interface. The Compliance & Safety workstream
replaces `verdict_for` with the real rule-table lookup; the node and the graph stay as is.

Contract:
  - reads  state["suggestions"], state["project"] (flat type, completion date)
  - writes a Verdict on EVERY suggestion; blocked ones are marked rejected (by "guard")
  - runs on every path to the owner: it is a graph edge, not a planner choice, so no agent
    can skip it, and nothing downstream may override a verdict
"""
from __future__ import annotations

from renopilot import trace
from renopilot.schemas import ProjectProfile, Suggestion, Verdict


def verdict_for(s: Suggestion, project: ProjectProfile | None) -> Verdict:
    if s.work_type == "loose_furniture":
        return Verdict(decision="allow", reason="Loose furniture: no HDB approval needed.")
    # Not built yet: say so instead of guessing (proposal: "escalate when the rule base does not cover the case").
    return Verdict(decision="escalate", rule_id=None,
                   reason=f"{s.item_type.replace('_', ' ')} is a {s.work_type.replace('_', '-')} work; "
                          "the HDB rule check is not implemented yet. Check HDB's guidelines before proceeding.",
                   citation="https://www.hdb.gov.sg/residential/living-in-an-hdb-flat/renovation")


def compliance_guard(state: dict) -> dict:
    project = state.get("project")
    out, blocked = [], []
    for s in state.get("suggestions") or []:
        if s.status == "rejected":
            out.append(s)
            continue
        v = verdict_for(s, project)
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

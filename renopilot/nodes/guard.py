"""Compliance guard: every suggestion gets a Verdict from the HDB rule cards (renopilot/compliance.py,
data/compliance/hdb_rules.jsonl), each citing the HDB page it rests on.

Each item carries its own room and requested work method. Unknown methods/rooms/critical
facts escalate; a prohibited request stays blocked. Other routes are options, never silent
substitutions. A separate owner-chosen alternative goes through this guard again. An absent
completion date cannot be asserted by selecting an older-block route. Dates can be frozen
with ProjectProfile.as_of for evaluation.

Contract:
  - reads suggestions, project completion/as-of dates, and individual items in the style profile
  - writes a Verdict on EVERY suggestion; blocked ones are marked rejected (by "guard") and are
    still shown to the owner with the reason and the legal alternative (`blocked`)
  - runs on every path to the owner: it is a graph edge, not a planner choice, so no agent
    can skip it, and nothing downstream may override a verdict
"""
from __future__ import annotations

import re

from renopilot import trace
from renopilot.compliance import IMPLIED_ROOM, routes, rule
from renopilot.schemas import ProjectProfile, Suggestion, Verdict
from renopilot.vocab import WORK_TYPE_OF


def _verdict(card: dict) -> Verdict:
    how = card["when"] if card["method"] == "any" else f"{card['method'].capitalize()}: {card['when']}"
    return Verdict(decision=card["decision"], reason=card["summary"], rule_id=card["id"], route=how,
                   conditions=card["conditions"], alternative=card["alternative"], citation=card["url"])


def with_route(v: Verdict | None, rule_id: str | None) -> Verdict | None:
    """v with the route `rule_id` in effect, if it is one of v's own routes and not blocked; else v."""
    pick = next((o for o in (v.options if v else []) if o.rule_id == rule_id
                 and o.decision in ("allow", "allow_with_permit")), None)
    return pick.model_copy(update={"options": v.options}) if pick else v


def requested_method(s) -> str:
    """Conservative fallback for old profiles that do not have structured work details."""
    text = (getattr(s, "description", None) or s.title).lower()
    text = re.sub(r"\b(?:no|without)\s+(?:hacking|removing|removal|demolition)\b", "", text)
    if re.search(r"\b(raise|raising|raised|platform)\b", text):
        return "raise"
    if re.search(r"\b(hack\w*|remov\w*|demolish\w*)\b", text):
        return "demolish" if s.item_type == "partition_wall_change" else "replace"
    if s.work_method != "unspecified":
        return s.work_method
    for method, pattern in (("replace", r"\breplac\w*\b"), ("overlay", r"\boverlay\w*\b|\blay(?:ing)? over\b"),
                            ("erect", r"\berect\w*\b|\bnew partition\b")):
        if re.search(pattern, text):
            return method
    return "unspecified"


@trace.tool("hdb_rule_check", inputs=("s", "project", "room"))
def verdict_for(s: Suggestion, project: ProjectProfile | None, room: str | None = None) -> Verdict:
    if WORK_TYPE_OF[s.item_type] == "loose_furniture":
        return Verdict(decision="allow", reason="Loose furniture: no HDB approval needed.")
    room = s.room or (IMPLIED_ROOM.get(s.item_type) if room in (None, "mixed") else room)
    completed = project.completion_date if project else None
    cards = routes(s.item_type, room, completed, as_of=project.as_of if project else None) or [rule("HDB-NOT-LISTED")]
    options = [_verdict(c) for c in cards]
    method = requested_method(s)
    if room == "household_shelter":
        if cards[0]["id"] == "HDB-HS":
            return options[0].model_copy(update={"options": options})
        return Verdict(decision="escalate", reason="This household-shelter work needs a specific HDB check.",
                       rule_id="HDB-NOT-LISTED", citation=rule("HDB-NOT-LISTED")["url"])
    if not completed:
        options = [o.model_copy(update={"decision": "escalate", "reason":
                    "Confirm the block completion date before replacing bathroom finishes. " + o.reason})
                   if c.get("block_age") else o for c, o in zip(cards, options)]
    text = re.sub(r"\b(?:non[- ]structural|not structural)\b", "", s.title.lower())
    structural = s.structural is True or bool(re.search(
        r"\b(structural\s+(?:wall|member|column|beam|slab)|(?:wall|member)\s+is\s+structural|"
        r"load[- ]bearing|reinforced[- ]concrete|rc)\b", text))
    if s.item_type == "partition_wall_change" and (structural or s.structural is not False):
        options = [o.model_copy(update={"decision": "escalate", "reason":
                   "Confirm a non-structural wall with HDB before using this demolition route."})
                   if o.rule_id == "HDB-WALL-DEMOLISH" else o for o in options]
    if s.item_type == "partition_wall_change" and method == "demolish":
        rid = "HDB-WALL-STRUCTURAL" if structural else "HDB-WALL-DEMOLISH"
        chosen = next((o for o in options if o.rule_id == rid), _verdict(rule(rid)))
        if not structural and s.structural is not False and chosen.decision != "escalate":
            chosen = chosen.model_copy(update={"decision": "escalate", "reason":
                       "Confirm that the wall is non-structural with HDB before planning demolition. " + chosen.reason})
        # An unknown wall cannot be asserted non-structural by selecting a permit option.
        options = [chosen if o.rule_id == rid else o for o in options]
    elif method != "unspecified":
        matching = [o for c, o in zip(cards, options) if c["method"] == method]
        if not matching:
            matching = [o for c, o in zip(cards, options) if c["method"] == "any"]
        chosen = matching[0] if matching else Verdict(decision="escalate", reason=
                    f"No verified rule covers the requested method ({method}); check with HDB.",
                    rule_id="HDB-NOT-LISTED", citation=rule("HDB-NOT-LISTED")["url"])
    elif all(o.decision == "block" for o in options):
        chosen = options[0]
    elif any(c["method"] == "any" for c in cards):
        chosen = next(o for c, o in zip(cards, options) if c["method"] == "any")
    else:
        chosen = Verdict(decision="escalate", reason="Choose the renovation method; the requested work has not been cleared.",
                         rule_id="HDB-NOT-LISTED", citation=rule("HDB-NOT-LISTED")["url"])
    if room in (None, "mixed") and chosen.decision != "block":
        return Verdict(decision="escalate", reason="Confirm the room for this work before checking its HDB route.",
                       rule_id="HDB-NOT-LISTED", citation=rule("HDB-NOT-LISTED")["url"])
    return chosen.model_copy(update={"options": options})


def blocked(suggestions) -> list[dict]:
    """What the guard blocked, for the owner: never dropped silently (approval gate and reply)."""
    return [{"key": s.key, "item_type": s.item_type, "room": s.room, "title": s.title,
             "reason": s.verdict.reason, "rule_id": s.verdict.rule_id,
             "alternative": s.verdict.alternative, "citation": s.verdict.citation}
            for s in suggestions or [] if s.status == "rejected" and s.verdict and s.verdict.decision == "block"]


def compliance_guard(state: dict) -> dict:
    project, profile = state.get("project"), state.get("style_profile")
    out, blocked = [], []
    for s in state.get("suggestions") or []:
        if s.status == "rejected":
            out.append(s)
            continue
        # a route already in effect (the owner's pick at an earlier approval) stays, if still legal
        matches = [i for i in profile.items if i.item_type == s.item_type and
                   (not s.room or i.room == s.room)] if profile else []
        item = matches[0] if len(matches) == 1 else None
        if item:
            s = s.model_copy(update={"room": s.room or item.room,
                                    "work_method": s.work_method if s.work_method != "unspecified" else item.work_method,
                                    "structural": s.structural if s.structural is not None else item.structural})
        v = verdict_for(s, project, s.room)
        if requested_method(s) == "unspecified" and v.decision != "block":
            v = with_route(v, s.verdict.rule_id if s.verdict else None)
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

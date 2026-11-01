"""Compliance guard: every suggestion gets a Verdict from the HDB rule cards (renopilot/compliance.py,
data/compliance/hdb_rules.jsonl), each citing the HDB page it rests on.

Each item carries its own room and requested work method. Unknown methods/rooms/critical
facts escalate; a prohibited request stays blocked. Other routes are options, never silent
substitutions. A separate owner-chosen alternative goes through this guard again. An absent
completion date cannot be asserted by selecting an older-block route. Dates can be frozen
with ProjectProfile.as_of for evaluation.

Contract:
  - reads suggestions, project completion/as-of dates, and individual items in the style profile
  - adds any finish or built-in in the profile that no suggestion carries, so requested work always
    gets a verdict whichever path led here
  - writes a Verdict on EVERY suggestion; blocked ones are marked rejected (by "guard") and are
    still shown to the owner with the reason and the legal alternative (`blocked`)
  - runs on every path to the owner: it is a graph edge, not a planner choice, so no agent
    can skip it, and nothing downstream may override a verdict
"""
from __future__ import annotations

import re

from renopilot import trace
from renopilot.compliance import IMPLIED_ROOM, routes, rule
from renopilot.schemas import ItemSpec, ProjectProfile, StyleProfile, Suggestion, Verdict
from renopilot.vocab import WORK_TYPE_OF


def _verdict(card: dict) -> Verdict:
    how = card["when"] if card["method"] == "any" else f"{card['method'].capitalize()}: {card['when']}"
    return Verdict(decision=card["decision"], reason=card["summary"], rule_id=card["id"], route=how,
                   conditions=card["conditions"], alternative=card["alternative"], citation=card["url"])


def _check_work_facts(v: Verdict, s: Suggestion, room: str | None = None) -> Verdict:
    """Apply the few measured limits we can verify from structured facts.

    Missing facts stay unresolved; prose or a generic "conditions met" flag cannot clear them.
    """
    facts = getattr(s, "work_details", None)
    update = {}
    text = s.title.lower()
    conflict = object()

    def fact(name: str):
        """Use only narrowly contextual text. If it disagrees with structured input, escalate."""
        structured = getattr(facts, name, None)
        values = []
        if name == "overlay_thickness_mm":
            for m in re.finditer(r"(?:overlay|adhesive).{0,20}\b(\d+(?:\.\d+)?)\s*mm\b|"
                                 r"\b(\d+(?:\.\d+)?)\s*mm\b.{0,20}(?:overlay|adhesive)", text):
                values.append(float(m.group(1) or m.group(2)))
        elif name == "ceiling_clearance_mm":
            pattern = r"(?:false ceiling|ceiling clearance|clearance height|clearance)(?:\s+(?:height|of|at|is))?\s+"
            pattern += r"(\d+(?:\.\d+)?)\s*(mm|m|metres?|meters?)\b|"
            pattern += r"\b(\d+(?:\.\d+)?)\s*(mm|m|metres?|meters?)\s+(?:false ceiling|ceiling clearance|clearance)\b"
            for m in re.finditer(pattern, text):
                amount, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
                values.append(float(amount) * (1000 if unit != "mm" else 1))
        elif name == "existing_floor_layers":
            values = [int(m.group(1)) for m in re.finditer(
                r"\b(\d+)\s+(?:existing\s+(?:floor[- ]finish\s+)?layers?|"
                r"(?:existing\s+)?floor[- ]finish\s+layers?|layers?\s+of\s+existing\s+floor)\b", text)]
        if name == "non_combustible":
            states = []
            if re.search(r"\bnon[- ]combustible\b", text):
                states.append(True)
            if re.search(r"\b(?:combustible|plywood|plastic)\b", re.sub(r"\bnon[- ]combustible\b", "", text)):
                states.append(False)
            values = list(dict.fromkeys(states))
        elif name == "gas_pipe_enclosed":
            states = []
            if re.search(r"\b(?:gas pipes? (?:will be )?(?:not|never) enclosed|"
                         r"(?:no|without) gas pipes?\b.{0,30}\benclosed)\b", text):
                states.append(False)
            if re.search(r"\bgas pipes? (?:will be )?(?:enclosed|inside)\b", text):
                states.append(True)
            values = list(dict.fromkeys(states))
        elif name == "partition_material":
            materials = ("hollow_block", "glass_block", "gypsum", "plywood", "plastic", "asbestos")
            for material in materials:
                word = material.replace("_", " ")
                if re.search(rf"\b(?:{word})\b.{{0,20}}\bpartition\b|\bpartition\b.{{0,20}}\b(?:{word})\b", text):
                    if not re.search(rf"\b(?:not|without|no)\s+(?:made of\s+)?{word}\b", text):
                        values.append(material)
            values = list(dict.fromkeys(values))
        if len(values) > 1:
            return conflict
        explicit = values[0] if values else None
        if structured is not None and explicit is not None and structured != explicit:
            return conflict
        return structured if structured is not None else explicit

    rid = v.rule_id or ""
    checked: set[str] = set()
    relevant = {
        "HDB-BATH-3YR-OVERLAY": ("overlay_thickness_mm", "existing_floor_layers"),
        "HDB-FLOOR-OVERLAY": ("overlay_thickness_mm", "existing_floor_layers"),
        "HDB-WALLFIN-OVERLAY": ("overlay_thickness_mm",),
        "HDB-CEILING": (("ceiling_clearance_mm",) if (s.room or room) not in ("bathroom", "household_shelter") else ())
                        + ("non_combustible", "gas_pipe_enclosed"),
        "HDB-WALL-ERECT": ("partition_material",),
    }.get(rid, ())
    if any(fact(name) is conflict for name in relevant):
        return v.model_copy(update={"decision": "escalate",
            "reason": "The request description conflicts with its structured work facts. Confirm the measurement or material before checking this HDB route.",
            "conditions": [f"Pending verification: {c}" for c in v.conditions]})
    if rid in {"HDB-BATH-3YR-OVERLAY", "HDB-FLOOR-OVERLAY", "HDB-WALLFIN-OVERLAY"}:
        thickness = fact("overlay_thickness_mm")
        layers = fact("existing_floor_layers") if rid != "HDB-WALLFIN-OVERLAY" else None
        if thickness is None:
            update.update(decision="escalate", reason="Pending: confirm the new finish including adhesive thickness; HDB permits at most 13mm. " + v.reason)
        elif thickness > 13:
            update.update(decision="block", reason=f"The proposed overlay is {thickness:g}mm; HDB allows at most 13mm including adhesive.")
        else:
            checked.add("13mm")
        if rid != "HDB-WALLFIN-OVERLAY" and layers is None:
            update.setdefault("reason", "Pending: confirm there is exactly one existing floor-finish layer (including cement screed). " + v.reason)
            update["decision"] = "escalate" if update.get("decision") != "block" else "block"
        elif rid != "HDB-WALLFIN-OVERLAY" and layers != 1:
            update.update(decision="block", reason=f"The proposed overlay has {layers} existing floor-finish layers; HDB allows exactly one.")
        elif rid != "HDB-WALLFIN-OVERLAY":
            checked.add("only 1 existing layer")
    elif rid == "HDB-CEILING":
        non_combustible = fact("non_combustible")
        gas_pipe_enclosed = fact("gas_pipe_enclosed")
        if (s.room or room) not in ("bathroom", "household_shelter"):
            clearance = fact("ceiling_clearance_mm")
            if clearance is None:
                update.update(decision="escalate", reason="Pending: confirm false-ceiling clearance above finished floor level; HDB requires at least 2400mm. " + v.reason)
            elif clearance < 2400:
                update.update(decision="block", reason=f"The proposed false-ceiling clearance is {clearance:g}mm; HDB requires at least 2400mm.")
            else:
                checked.add("at least 2.4m")
        elif (s.room or room) in ("bathroom", "household_shelter"):
            checked.add("not applicable to toilets and bathrooms")
        if non_combustible is None:
            update.setdefault("reason", "Pending: confirm false-ceiling material is non-combustible. " + v.reason)
            update["decision"] = "escalate" if update.get("decision") != "block" else "block"
        elif not non_combustible:
            update.update(decision="block", reason="The proposed false-ceiling material is combustible; HDB requires non-combustible materials.")
        else:
            checked.add("Non-combustible materials only")
        if gas_pipe_enclosed is None:
            update.setdefault("reason", "Pending: confirm no gas pipe will be enclosed in the false ceiling. " + v.reason)
            update["decision"] = "escalate" if update.get("decision") != "block" else "block"
        elif gas_pipe_enclosed:
            update.update(decision="block", reason="The proposed false ceiling encloses a gas pipe, which HDB prohibits.")
        else:
            checked.add("No gas pipe enclosed")
    elif rid == "HDB-WALL-ERECT":
        material = fact("partition_material")
        if material is None:
            update.update(decision="escalate", reason="Pending: identify the partition material; HDB prohibits combustible or toxic-emission materials such as plywood.")
        elif material in ("plywood", "plastic", "asbestos"):
            update.update(decision="block", reason=f"A {material} partition is prohibited by HDB's partition-material rule.")
        elif material == "other":
            update.update(decision="escalate", reason="Pending: confirm that the partition material is permitted by HDB; this material is not covered by the checked options.")
        else:
            checked.add("No partitions of combustible or toxic-emission materials")
    if v.conditions:
        conditions = []
        for condition in v.conditions:
            matched = next((item for item in checked if item.lower() in condition.lower()), None)
            conditions.append(("Checked from stated work facts: " if matched else "Pending verification: ") + condition)
        update["conditions"] = conditions
        if any(c.startswith("Pending verification:") for c in conditions) and v.decision in ("allow", "allow_with_permit"):
            update["reason"] = (update.get("reason", v.reason) + " Subject to verification of the remaining HDB conditions.")
    return v.model_copy(update=update) if update else v


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
                            ("erect", r"\berect\w*\b|\bnew partition\b|\bplywood partition\b")):
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
    options = [_check_work_facts(o, s, room) for o in options]
    chosen = _check_work_facts(chosen, s, room)
    return chosen.model_copy(update={"options": options})


def blocked(suggestions) -> list[dict]:
    """What the guard blocked, for the owner: never dropped silently (approval gate and reply)."""
    return [{"key": s.key, "item_type": s.item_type, "room": s.room, "title": s.title,
             "reason": s.verdict.reason, "rule_id": s.verdict.rule_id,
             "alternative": s.verdict.alternative, "citation": s.verdict.citation}
            for s in suggestions or [] if s.status == "rejected" and s.verdict and s.verdict.decision == "block"]


def request_from(spec: ItemSpec) -> Suggestion:
    """A finish or built-in the owner asked for, as a product-less suggestion for the guard."""
    return Suggestion(item_type=spec.item_type, work_type=WORK_TYPE_OF[spec.item_type], title=spec.description,
                      room=spec.room, work_method=spec.work_method, structural=spec.structural,
                      work_details=getattr(spec, "work_details", None),
                      request_id=getattr(spec, "request_id", ""), room_label=getattr(spec, "room_label", ""),
                      quantity=getattr(spec, "quantity", 1),
                      reason="Requested by the board")


def unchecked_requests(profile: StyleProfile | None, suggestions: list[Suggestion]) -> list[Suggestion]:
    """Requested work no suggestion carries yet. A route the owner picked changes a key's method,
    so a request with no method of its own matches on item type and room."""
    out: list[Suggestion] = []
    for spec in profile.items if profile else []:
        if WORK_TYPE_OF[spec.item_type] == "loose_furniture":
            continue
        new = request_from(spec)
        if not any((s.request_id == spec.request_id if spec.request_id else
                    s.key == new.key or (not s.request_id and spec.work_method == "unspecified" and
                                        (s.item_type, s.room) == (spec.item_type, spec.room)))
                   for s in [*suggestions, *out] if not s.product_id):
            out.append(new)
    return out


def compliance_guard(state: dict) -> dict:
    project, profile = state.get("project"), state.get("style_profile")
    out, blocked = [], []
    suggestions = list(state.get("suggestions") or [])
    added = unchecked_requests(profile, suggestions)
    if added:
        trace.log("compliance_guard", "added_requests", keys=[s.key for s in added])
    for s in suggestions + added:
        if s.status == "rejected":
            out.append(s)
            continue
        # a route already in effect (the owner's pick at an earlier approval) stays, if still legal
        matches = [i for i in profile.items if (i.request_id == s.request_id if s.request_id else
                   i.item_type == s.item_type and (not s.room or i.room == s.room))] if profile else []
        item = matches[0] if len(matches) == 1 else None
        if item:
            details = getattr(s, "work_details", None)
            if not details or not any(value is not None for value in details.model_dump().values()):
                details = item.work_details
            s = s.model_copy(update={"room": s.room or item.room,
                                    "work_method": s.work_method if s.work_method != "unspecified" else item.work_method,
                                    "structural": s.structural if s.structural is not None else item.structural,
                                    "work_details": details})
        v = verdict_for(s, project, s.room)
        if requested_method(s) == "unspecified" and v.decision != "block":
            v = with_route(v, s.verdict.rule_id if s.verdict else None)
        s = s.model_copy(update={"verdict": v, "status": "rejected" if v.decision == "block" else s.status})
        if v.decision == "block":
            blocked.append({"product_id": s.product_id, "request_id": s.request_id,
                            "item_type": s.item_type, "reason": v.reason, "by": "guard"})
        out.append(s)
    counts = {}
    for s in out:
        if s.verdict:
            counts[s.verdict.decision] = counts.get(s.verdict.decision, 0) + 1
    trace.log("compliance_guard", "result", **counts)
    return {"suggestions": out, "rejected": blocked,
            "trace": [{"node": "compliance_guard", "action": "checked", "why": str(counts)}]}

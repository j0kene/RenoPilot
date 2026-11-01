"""The planner: the MASTER agent. Each step it calls one tool, and each tool hands the work to a
subagent node (the Command it returns routes there). The model decides; code enforces.

    extract_style   Style agent reads the board          needs a usable board, no profile yet
    shortlist       Style agent's shoppers find products  needs a profile; types must be on the board
                    (the Pricing agent prices every new shortlist: a fixed edge)
    ask_user        stop and ask the owner                needs a question
    finalize        hand over to the Compliance gate      needs at least one suggestion

`check_action` holds the rules. A refused call goes back to the model once, as that tool's
result with the reason (course: a checkable rejection reason and a round cap); after that, or
if the reply has no valid tool call, the fixed policy decides. The step cap ends the turn.

`fixed_policy` is also the whole planner in the "fixed pipeline" baseline, which is how the
eval shows whether the LLM planner earns its cost.

Where the LLM planner can add value over the fixed policy: turning owner feedback
("too bulky, darker wood") into search refinements, and choosing what to ask.
It can NOT change the budget: only the owner's answer at an interrupt can.
"""
from __future__ import annotations

import time
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError, create_model

from renopilot import config, trace
from renopilot.llm import get_model, usage_of
from renopilot.nodes.intake import check_board
from renopilot.schemas import ItemType

MENU = ("extract_style", "shortlist", "ask_user", "finalize")
GOTO = {"extract_style": "style_extract", "shortlist": "shortlist", "ask_user": "ask_user",
        "finalize": "compliance_guard"}
MAX_REVISIONS = 1
PROMPT_VERSION = "planner-tools-v2"     # v0 (structured output) results: evals/results/graph_llm_sonnet*.jsonl


class Decision(BaseModel):
    action: Literal[MENU]                                    # type: ignore[valid-type]
    item_types: list[ItemType] = Field(default_factory=list,
                                       description="For shortlist: which item types to (re)search")
    refinement: Optional[str] = Field(None, description="For shortlist: extra search words drawn from "
                                                        "the owner's feedback, at most 12 words")
    question: Optional[str] = Field(None, description="For ask_user: one short question for the owner")
    why: str = Field(description="One sentence")


# The master's tools: one per menu action, args as in Decision. Calling one hands over to a subagent.
_WHY = (str, Field(description="One sentence: why this step now"))
TOOLS = [
    create_model("extract_style", __doc__="Style agent: read the mood board into a style profile "
                 "(only if there is no profile yet).", why=_WHY),
    create_model("shortlist", __doc__="Style agent: find products for the listed item types; the Pricing agent "
                 "then prices them. Finishes and built-ins pass through it to the compliance check. Use it first "
                 "for every item on the board, and again for item types the owner rejected, with a refinement "
                 "drawn from their feedback.",
                 item_types=(list[ItemType], Field(description="Which item types to (re)search")),
                 refinement=(Optional[str], Field(None, description="Extra search words drawn from the owner's "
                                                                    "feedback, at most 12 words")), why=_WHY),
    create_model("ask_user", __doc__="Ask the owner ONE short question, when an open issue needs their decision "
                 "(e.g. too few products within budget, or an unclear request).",
                 question=(str, Field(description="One short question for the owner")), why=_WHY),
    create_model("finalize", __doc__="Hand the current shortlist to the compliance check and the owner's "
                 "approval.", why=_WHY),
]

PLANNER_SYSTEM = """You coordinate RenoPilot, which turns a homeowner's mood board into a shortlist of products.
You are the master agent: call exactly ONE tool for the next step; each tool hands the work to a subagent.
You cannot change the budget yourself: only the owner can, when you ask them.
The owner's message adds to the board's items; it never replaces them. Every item in the style profile,
including finishes, built-ins and renovation requests, goes through shortlist before finalize.
Be economical: do not repeat a step that has already succeeded."""


# --------------------------------------------------------------------------- rules (code)
def unsearched(state: dict) -> list[str]:
    """Item types in the profile not searched yet. A search that found nothing still counts."""
    profile = state.get("style_profile")
    done = set(state.get("searched_types") or []) | {s.item_type for s in state.get("suggestions") or []}
    return list(dict.fromkeys(i.item_type for i in profile.items if i.item_type not in done)) if profile else []


def check_action(state: dict, d: Decision) -> str | None:
    """Why this decision is not allowed now, or None."""
    profile = state.get("style_profile")
    if d.action == "extract_style":
        if profile is not None:
            return "a style profile already exists"
        problem = check_board(state.get("board_images") or [])   # re-checked here, not only at intake
        if problem:
            return f"the board is not usable: {problem}"
        if state.get("open_issues"):
            return f"the board has an open issue: {state['open_issues'][0]}"
    if d.action == "shortlist":
        if profile is None:
            return "no style profile yet: extract_style first"
        if not d.item_types:
            return "shortlist needs item_types"
        board = {i.item_type for i in profile.items}
        off = [t for t in d.item_types if t not in board]
        if off:
            return f"item types not on the board: {off}"
        # No repeat searches without a reason (MAST "step repetition"): a type may be searched
        # if it never was, or if the owner gave feedback / changed its budget since.
        allowed = set(unsearched(state)) | set(state.get("redo_types") or [])
        repeat = sorted(t for t in d.item_types if t not in allowed)
        if repeat:
            return (f"{repeat} already searched and nothing has changed since; "
                    f"you may search only {sorted(allowed) or 'nothing'}")
    if d.action == "ask_user" and not (d.question or "").strip():
        return "ask_user needs a question"
    if d.action == "finalize":
        if not any(s.status != "rejected" for s in state.get("suggestions") or []):
            return "nothing to finalize: shortlist first"
        if state.get("open_issues"):
            return f"open issue to resolve first: {state['open_issues'][0]}"
        if state.get("redo_types"):
            return f"item types still need a new search: {state['redo_types']}"
        if unsearched(state):                         # the live planner narrowed to what the brief named
            return f"items in the style profile not searched yet: {unsearched(state)}"
    return None


def fixed_policy(state: dict) -> Decision:
    """The deterministic order. Baseline planner, and the fallback for the LLM planner."""
    profile = state.get("style_profile")
    if state.get("open_issues"):
        return Decision(action="ask_user", question=state["open_issues"][0], why="open issue")
    if profile is None:
        problem = check_board(state.get("board_images") or [])
        if problem:
            return Decision(action="ask_user", question=problem, why="board still not usable")
        return Decision(action="extract_style", why="no profile yet")
    redo = list(state.get("redo_types") or [])
    if redo or unsearched(state):
        return Decision(action="shortlist", item_types=list(dict.fromkeys(redo + unsearched(state))),
                        why="owner feedback" if redo else "first search")
    if not state.get("suggestions"):
        return Decision(action="ask_user", why="every search came back empty",
                        question="Nothing on the board could be found within budget. Raise the budget or change the board?")
    return Decision(action="finalize", why="shortlist ready")


# --------------------------------------------------------------------------- model
def _briefing(state: dict) -> str:
    """Context-minimal: the planner never sees images or full product rows."""
    p = state.get("style_profile")
    sugg = state.get("suggestions") or []
    by_type: dict[str, int] = {}
    finishes = sorted({s.item_type for s in sugg if s.work_type != "loose_furniture"})
    for s in sugg:
        if s.status != "rejected" and s.work_type == "loose_furniture":
            by_type[s.item_type] = by_type.get(s.item_type, 0) + 1
    last_user = next((m.content for m in reversed(state.get("messages") or [])
                      if isinstance(m, HumanMessage) and isinstance(m.content, str)), "")
    pending = set(state.get("redo_types") or [])
    user_rej = [r for r in state.get("rejected") or [] if r.get("by") == "user"]
    todo = [f"{r['item_type']}: {r.get('reason') or 'no reason'}" for r in user_rej if r.get("item_type") in pending]
    done = [f"{r['item_type']}: {r.get('reason') or 'no reason'}" for r in user_rej if r.get("item_type") not in pending]
    return "\n".join([
        f"Board images: {len(state.get('board_images') or [])}",
        f"Style profile: {', '.join(p.style_tags) + ' | items: ' + ', '.join(i.item_type for i in p.items) if p else 'none'}",
        f"Live product suggestions per item type: {by_type or 'none'} (prices are landed SGD totals; items over "
        "budget after delivery/shipping/GST were already removed)",
        f"Items not searched yet (all must be, before finalize): {unsearched(state) or 'none'}",
        f"Finishes / built-ins passed to the compliance check as requests: {finishes or 'none'}",
        f"Owner feedback TO ACT ON (item type: reason): {todo or 'none'}",
        f"Earlier feedback, already acted on: {done[-4:] or 'none'}",
        f"Item types needing a new search: {state.get('redo_types') or 'none'}",
        f"Open issues: {state.get('open_issues') or 'none'}",
        f"Owner's last message: {last_user[:300] or 'none'}",
        f"Steps used this turn: {state.get('steps', 0)} of {config.MAX_PLANNER_STEPS}",
    ])


def llm_decide(state: dict, model=None) -> Decision:
    model = (model or get_model("planner")).bind_tools(TOOLS, tool_choice="any")
    msgs = [SystemMessage(PLANNER_SYSTEM), HumanMessage(_briefing(state))]
    for attempt in range(MAX_REVISIONS + 1):
        t0 = time.perf_counter()
        reply = model.invoke(msgs)
        calls, d, error = reply.tool_calls, None, None if reply.tool_calls else "no tool call"
        if calls:
            try:
                d = Decision.model_validate({**calls[0]["args"], "action": calls[0]["name"]})
            except ValidationError as e:
                error = str(e)
        trace.log("planner", "model_call", attempt=attempt, latency_s=round(time.perf_counter() - t0, 2),
                  prompt_version=PROMPT_VERSION, **usage_of(reply), tool_calls=[c["name"] for c in calls],
                  error=error)
        if d is None:
            break
        reason = check_action(state, d)
        if reason is None:
            return d
        trace.log("planner", "refused", action=d.action, reason=reason)
        msgs += [reply] + [ToolMessage(f"Not allowed: {reason}. Choose again." if i == 0 else
                                       "Ignored: call one tool at a time.", tool_call_id=c["id"], name=c["name"])
                           for i, c in enumerate(calls)]
    d = fixed_policy(state)
    trace.log("planner", "fallback", action=d.action)
    return d.model_copy(update={"why": f"fallback to fixed policy: {d.why}"})


# --------------------------------------------------------------------------- node
def make_planner(mode: str = "llm", model=None):
    """mode: "llm" (model decides, code enforces) or "fixed" (baseline). `model` overrides
    the planner model (tests pass a fake)."""

    def planner(state: dict) -> Command:
        steps = state.get("steps", 0)
        if steps >= config.MAX_PLANNER_STEPS:
            msg = f"Stopped after {steps} planning steps without finishing; please review what is here."
            trace.log("planner", "step_cap", steps=steps)
            # nothing reaches the owner unchecked: with suggestions, the cap exits through the guard
            goto = "compliance_guard" if state.get("suggestions") or state.get("style_profile") else "respond"
            return Command(goto=goto, update={"escalations": [msg],
                                              "trace": [{"node": "planner", "action": "step_cap", "why": msg}]})
        d = fixed_policy(state) if mode == "fixed" else llm_decide(state, model)
        reason = check_action(state, d)
        if reason:                                   # the fixed policy is valid by construction; belt and braces
            d = fixed_policy(state)
        trace.log("planner", "decision", mode=mode, step=steps + 1, **d.model_dump())
        update = {"steps": steps + 1,
                  "trace": [{"node": "planner", "action": d.action, "why": d.why,
                             "item_types": list(d.item_types), "refinement": d.refinement}]}
        if d.action == "shortlist":
            update["shortlist_types"] = list(d.item_types)
            update["searched_types"] = sorted(set(state.get("searched_types") or []) | set(d.item_types))
            redo = set(state.get("redo_types") or [])
            if d.refinement and redo:                # refinements come from owner feedback only
                update["refinements"] = {**(state.get("refinements") or {}),
                                         **{t: d.refinement for t in d.item_types if t in redo}}
        if d.action == "ask_user":
            update["pending_question"] = d.question
        return Command(goto=GOTO[d.action], update=update)

    planner.__name__ = f"planner_{mode}"
    return planner

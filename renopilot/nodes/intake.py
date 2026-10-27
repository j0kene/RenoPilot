"""Intake: check the inputs in code, and turn the owner's brief into a ProjectProfile.

Image-count limits are rules, not suggestions, so they are checked here and never asked
of a model. The owner can give their details two ways, and both are merged:
  - form fields (flat type, block completion date, budgets): always win
  - a free-text brief ("4-room, keys in June 2024, sofa under $1,500, redo the bathroom floor"):
    read by a small model, and used only to fill fields the form left empty
The brief itself is kept in ProjectProfile.notes, so later steps can read what the owner asked for.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from renopilot import config, trace
from renopilot.llm import get_model, usage_of
from renopilot.schemas import ProjectProfile

INTAKE_SYSTEM = """Extract the homeowner's project details from their message into the schema.
Budgets are in Singapore dollars. budget_by_category maps an item type (e.g. sofa, coffee_table,
dining_table, bed_frame, rug) to the most they will pay for ONE such item.
completion_date is the date their HDB block was completed (keys collected); if only a month and year
are given, use the 1st of that month.
Leave a field empty if the message does not state it. Never guess numbers or dates."""

Parser = Callable[[str], "ProjectProfile | None"]


def check_board(paths: list[str]) -> str | None:
    """The reason the board is unusable, or None."""
    missing = [p for p in paths if not (Path(p) if Path(p).is_absolute() else config.ROOT / p).exists()]
    if missing:
        return f"{len(missing)} board image(s) could not be found."
    if len(paths) < config.MIN_BOARD_IMAGES:
        return (f"The mood board has {len(paths)} image(s); please add at least "
                f"{config.MIN_BOARD_IMAGES} so the style is clear.")
    if len(paths) > config.MAX_BOARD_IMAGES:
        return f"Please keep the board to {config.MAX_BOARD_IMAGES} images or fewer ({len(paths)} given)."
    return None


def parse_profile(text: str, model=None) -> ProjectProfile | None:
    model = model or get_model("intake")
    t0 = time.perf_counter()
    out = model.with_structured_output(ProjectProfile, include_raw=True).invoke(
        [SystemMessage(INTAKE_SYSTEM), HumanMessage(text)])
    trace.log("intake", "model_call", latency_s=round(time.perf_counter() - t0, 2), **usage_of(out["raw"]))
    return out["parsed"]


def merge(form: ProjectProfile | None, parsed: ProjectProfile | None, brief: str) -> ProjectProfile:
    """Form fields win; the parsed brief fills only what the form left empty (per budget category too)."""
    form = form or ProjectProfile()
    if parsed is None:
        return form.model_copy(update={"notes": form.notes or brief})
    return form.model_copy(update={
        "flat_type": form.flat_type or parsed.flat_type,
        "completion_date": form.completion_date or parsed.completion_date,
        "total_budget": form.total_budget if form.total_budget is not None else parsed.total_budget,
        "budget_by_category": {**parsed.budget_by_category, **form.budget_by_category},
        "notes": form.notes or brief,
    })


def _needs_parse(form: ProjectProfile | None) -> bool:
    """Only call the model if the brief could fill something the form left empty."""
    return form is None or not (form.flat_type and form.completion_date and form.total_budget is not None)


def make_intake(parse: Parser | None = None):
    """parse: the brief reader (default: the intake model; demo mode passes a code parser)."""
    reader = parse or parse_profile

    def intake(state: dict) -> dict:
        """Start of every turn: reset the step counter, validate the board, fill the profile."""
        update: dict = {"steps": 0, "open_issues": []}
        problem = check_board(state.get("board_images") or [])
        if problem:
            update["open_issues"] = [problem]

        form = state.get("project")
        brief = next((m.content for m in reversed(state.get("messages") or [])
                      if isinstance(m, HumanMessage) and isinstance(m.content, str)), "").strip()
        parsed = None
        if brief and not (form and form.notes) and _needs_parse(form):   # first turn only: notes mark it read
            try:
                parsed = reader(brief)
            except Exception as e:                       # noqa: BLE001 - a brief we cannot read is not fatal
                trace.log("intake", "parse_failed", error=str(e))
        update["project"] = merge(form, parsed, brief) if brief else (form or ProjectProfile())
        p = update["project"]
        update["trace"] = [{"node": "intake", "action": "validated",
                            "why": problem or f"{len(state.get('board_images') or [])} images ok"
                                   + (f"; completed {p.completion_date}" if p.completion_date else "")
                                   + (f"; {p.flat_type}" if p.flat_type else "")}]
        trace.log("intake", "result", problem=problem, project=p.model_dump(mode="json"), parsed_brief=parsed is not None)
        return update

    return intake


intake = make_intake()

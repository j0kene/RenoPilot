"""Intake: check the inputs in code, and turn a free-text brief into a ProjectProfile.

Image-count limits are rules, not suggestions, so they are checked here and never asked
of a model. A budget typed in chat is parsed by a small model, but only when the UI did
not already supply a structured profile.
"""
from __future__ import annotations

import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from renopilot import config, trace
from renopilot.llm import get_model, usage_of
from renopilot.schemas import ProjectProfile

INTAKE_SYSTEM = """Extract the homeowner's project details from their message into the schema.
Budgets are in Singapore dollars. budget_by_category maps an item type (e.g. sofa, coffee_table,
dining_table, bed_frame, rug) to the most they will pay for ONE such item.
Leave a field empty if the message does not state it. Never guess numbers."""


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


def intake(state: dict) -> dict:
    """Start of every turn: reset the step counter, validate the board, fill the profile."""
    update: dict = {"steps": 0, "open_issues": []}
    problem = check_board(state.get("board_images") or [])
    if problem:
        update["open_issues"] = [problem]

    project = state.get("project")
    last_user = next((m.content for m in reversed(state.get("messages") or [])
                      if isinstance(m, HumanMessage) and isinstance(m.content, str)), "")
    if project is None and last_user:
        project = parse_profile(last_user) or ProjectProfile()
    update["project"] = project or ProjectProfile()
    update["trace"] = [{"node": "intake", "action": "validated",
                        "why": problem or f"{len(state.get('board_images') or [])} images ok"}]
    trace.log("intake", "result", problem=problem, project=update["project"].model_dump(mode="json"))
    return update

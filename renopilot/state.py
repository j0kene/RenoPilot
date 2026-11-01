"""The shared project state. Every node reads it and returns only the keys it changed
(a node returns a delta; LangGraph merges it). Every step is checkpointed per thread."""
from __future__ import annotations

import operator
from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from renopilot.schemas import ProjectProfile, Suggestion, StyleProfile


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]           # chat with the owner
    project: ProjectProfile
    board_images: list[str]                           # file paths
    style_profile: Optional[StyleProfile]
    suggestions: list[Suggestion]                     # current shortlist (all item types)
    rejected: Annotated[list[dict], operator.add]     # {"product_id"|"item_type", "reason", "by": "user"|"guard"}
    escalations: Annotated[list[str], operator.add]   # boundary hits, shown to the owner
    open_issues: list[str]                            # unresolved boundary hits; cleared once the owner answers
    pending_question: Optional[str]                   # set by the planner's ask_user, answered at the interrupt
    shortlist_types: list[str]                        # item types the planner asked to (re)shortlist
    searched_types: list[str]                         # item types the planner has already sent to shortlist
    redo_types: list[str]                             # item types needing a new search (rejections, budget changes)
    refinements: dict[str, str]                       # item_type -> extra search words from owner feedback
    steps: int                                        # planner decisions this turn (cap: MAX_PLANNER_STEPS)
    approved: bool                                    # owner has approved the shortlist
    selected: dict[str, str]                          # item type -> one approved product/offer id
    approval_error: Optional[str]                     # invalid selection stays at the approval gate
    pricing_enabled: bool                            # False only for the explicit pricing ablation
    trace: Annotated[list[dict], operator.add]        # compact decision log: {"node", "action", "why"}

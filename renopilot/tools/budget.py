"""Budget arithmetic, in code. The model never adds up money.

A budget arrives either per category ("sofa up to S$1,500") or as one total. A total is
split across the item types on the board by fixed weights, so the same board and the
same total always give the same caps.
"""
from __future__ import annotations

from typing import Iterable

from renopilot.schemas import ProjectProfile, Suggestion

# Relative share of a total furniture budget. Rough Singapore BTO norms; only ratios matter.
WEIGHTS = {
    "sofa": 25, "bed_frame": 16, "dining_table": 12, "wardrobe": 12, "tv_console": 8,
    "armchair": 8, "desk": 7, "dresser": 7, "dining_chair": 4, "coffee_table": 6, "bookshelf": 5,
    "office_chair": 5, "rug": 5, "bedside_table": 3, "side_table": 3, "bench": 4, "bar_stool": 3,
    "ottoman": 3, "floor_lamp": 3, "table_lamp": 2, "pendant_light": 3, "mirror": 2,
}
DEFAULT_WEIGHT = 4


def allocate(total: float, item_types: Iterable[str]) -> dict[str, float]:
    """Split `total` SGD across item types by WEIGHTS, rounded down to whole dollars."""
    types = list(dict.fromkeys(item_types))
    if not types or total <= 0:
        return {}
    w = {t: WEIGHTS.get(t, DEFAULT_WEIGHT) for t in types}
    s = sum(w.values())
    return {t: float(int(total * w[t] / s)) for t in types}


def caps_for(project: ProjectProfile | None, item_types: Iterable[str]) -> dict[str, float | None]:
    """Per-item SGD cap for each type. Explicit category budgets win; the rest share what
    is left of the total; no budget at all means no cap (None)."""
    types = list(dict.fromkeys(item_types))
    if project is None:
        return {t: None for t in types}
    caps: dict[str, float | None] = {t: project.budget_by_category.get(t) for t in types}
    open_types = [t for t in types if caps[t] is None]
    if project.total_budget and open_types:
        left = project.total_budget - sum(project.budget_by_category.values())
        shares = allocate(max(left, 0.0), open_types)
        caps.update({t: shares.get(t, 0.0) for t in open_types})   # nothing left -> cap 0, not "no cap"
    return caps


def within(price_sgd: float | None, cap: float | None) -> bool:
    """Hard constraint. An unknown price never passes a real cap."""
    if cap is None:
        return True
    return price_sgd is not None and price_sgd <= cap + 1e-9


def total(suggestions: Iterable[Suggestion], status: str | None = "approved") -> float:
    """Sum of listed SGD prices (status=None sums everything)."""
    return round(sum(s.price_sgd or 0.0 for s in suggestions if status is None or s.status == status), 2)


def fmt_sgd(x: float | None) -> str:
    return "price n/a" if x is None else f"S${x:,.0f}" if x >= 100 else f"S${x:,.2f}"

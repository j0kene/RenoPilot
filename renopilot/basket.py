"""One offer per category; the UI, approval gate and export use the same money rules."""
from __future__ import annotations

from renopilot.pricing.landed_cost import basket_total
from renopilot.schemas import ProjectProfile, Suggestion
from renopilot.tools.budget import caps_for, fmt_sgd, within


def offers_for(suggestions: list[Suggestion]) -> dict[str, list[Suggestion]]:
    groups: dict[str, dict[str, Suggestion]] = {}
    rejected_ids = {s.product_id for s in suggestions if s.status == "rejected" and s.product_id}
    for s in suggestions:
        if s.status == "rejected" or not s.product_id:
            continue
        group = groups.setdefault(s.item_type, {})
        group.setdefault(s.product_id, s)
        for a in s.comparables:
            if a.quote.synthetic or a.product_id in rejected_ids:
                continue
            group.setdefault(a.product_id, Suggestion(
                product_id=a.product_id, item_type=s.item_type, work_type="loose_furniture",
                title=a.title, source=a.source, url=a.url, image_path=a.image_path,
                price_sgd=a.quote.item_sgd, price_quote=a.quote, reason=a.why_comparable,
                room=s.room, verdict=s.verdict))
    return {t: list(items.values()) for t, items in groups.items()}


def source_of(s: Suggestion) -> str:
    return s.source or (s.product_id or "other_sg").split(":", 1)[0]


def summary(items: list[Suggestion]) -> dict:
    complete = all(s.price_quote is not None for s in items)
    totals = basket_total([(source_of(s), s.price_quote) for s in items]) if complete else {
        "total_sgd": round(sum(s.price_sgd or 0 for s in items), 2), "delivery_by_source": {}}
    return totals | {"complete": complete, "estimated": any(s.price_quote and s.price_quote.estimated for s in items),
                     "basis": "landed SGD, delivery charged once per retailer, GST included" if complete else
                              "listed prices only; delivery, shipping and GST have not been checked"}


def defaults(suggestions: list[Suggestion], project: ProjectProfile | None = None) -> dict[str, str]:
    groups = offers_for(suggestions)
    picks = {t: items[0] for t, items in groups.items()}
    if project and project.total_budget is not None and summary(list(picks.values()))["total_sgd"] > project.total_budget:
        picks = {t: min(items, key=lambda s: s.price_quote.total_sgd if s.price_quote else
                       s.price_sgd if s.price_sgd is not None else float("inf")) for t, items in groups.items()}
    return {t: s.product_id for t, s in picks.items()}


def selection(suggestions: list[Suggestion], selected: dict[str, str], project: ProjectProfile | None,
              *, require_quotes: bool = True) -> tuple[list[Suggestion], dict]:
    groups = offers_for(suggestions)
    if not isinstance(selected, dict) or set(selected) != set(groups):
        raise ValueError("Choose exactly one offer for every furniture category.")
    caps = caps_for(project, groups)
    items = []
    for t, pid in selected.items():
        s = next((s for s in groups[t] if s.product_id == pid), None)
        if s is None or not s.verdict or s.verdict.decision not in ("allow", "allow_with_permit"):
            raise ValueError(f"Choose a checked offer for {t.replace('_', ' ')}.")
        if s.price_quote is None and require_quotes:
            raise ValueError(f"The delivered price for {s.title} is unavailable; choose another offer.")
        cost = s.price_quote.total_sgd if s.price_quote else s.price_sgd
        if not within(cost, caps.get(t)):
            raise ValueError(f"{s.title} exceeds the {fmt_sgd(caps[t])} category cap after delivery.")
        items.append(s)
    if not items and not any(not s.product_id and s.status != "rejected" and s.verdict and
                            s.verdict.decision in ("allow", "allow_with_permit") for s in suggestions):
        raise ValueError("There are no checked items to approve. Resolve pending work or adjust the budget.")
    totals = summary(items)
    if project and project.total_budget is not None and totals["total_sgd"] > project.total_budget:
        raise ValueError(f"This basket costs {fmt_sgd(totals['total_sgd'])}, above your total budget of "
                         f"{fmt_sgd(project.total_budget)}. Choose cheaper offers or explicitly change your budget.")
    return items, totals

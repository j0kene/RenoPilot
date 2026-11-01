"""One offer per category; the UI, approval gate and export use the same money rules."""
from __future__ import annotations

from renopilot.pricing.landed_cost import basket_total, money
from renopilot.schemas import ProjectProfile, Suggestion
from renopilot.tools.budget import request_caps, fmt_sgd, within


def offers_for(suggestions: list[Suggestion]) -> dict[str, list[Suggestion]]:
    groups: dict[str, dict[str, Suggestion]] = {}
    rejected_ids = {(s.selection_key, s.product_id) for s in suggestions if s.status == "rejected" and s.product_id}
    for s in suggestions:
        if s.status == "rejected" or not s.product_id:
            continue
        group = groups.setdefault(s.selection_key, {})
        group.setdefault(s.product_id, s)
        for a in s.comparables:
            if a.quote.synthetic or (s.selection_key, a.product_id) in rejected_ids:
                continue
            group.setdefault(a.product_id, Suggestion(
                product_id=a.product_id, item_type=s.item_type, work_type="loose_furniture",
                request_id=s.request_id, room_label=s.room_label, quantity=s.quantity,
                title=a.title, source=a.source, url=a.url, image_path=a.image_path,
                price_sgd=a.quote.item_sgd, price_quote=a.quote, reason=a.why_comparable,
                room=s.room, verdict=s.verdict, fit_note=a.fit_note, retrieved_at=a.retrieved_at))
    return {t: list(items.values()) for t, items in groups.items()}


def source_of(s: Suggestion) -> str:
    return s.source or (s.product_id or "other_sg").split(":", 1)[0]


def summary(items: list[Suggestion]) -> dict:
    complete = all(s.price_quote is not None for s in items)
    totals = basket_total([(source_of(s), s.price_quote) for s in items for _ in range(s.quantity)]) if complete else {
        "total_sgd": float(sum((money(s.price_sgd or 0) * s.quantity for s in items), money(0))), "delivery_by_source": {}}
    return totals | {"complete": complete, "estimated": any(s.price_quote and s.price_quote.estimated for s in items),
                     "basis": "landed SGD, delivery charged once per retailer, GST included" if complete else
                              "listed prices only; delivery, shipping and GST have not been checked"}


def unit_cost(s: Suggestion, units: int = 1) -> float | None:
    """Landed cost per unit with one delivery shared across `units` of this offer."""
    if s.price_quote is None:
        return s.price_sgd
    return s.price_quote.total_sgd if units == 1 else summary([s.model_copy(update={"quantity": units})])["total_sgd"] / units


def category_units(requests) -> dict[str, int]:
    """Units per category, the same count the budget allocation divides by."""
    units: dict[str, int] = {}
    for r in requests:
        units[r.item_type] = units.get(r.item_type, 0) + r.quantity
    return units


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
        raise ValueError("Choose exactly one offer for each furniture request and room.")
    caps = request_caps(project, [items[0] for items in groups.values()])
    units = category_units(items[0] for items in groups.values())
    items = []
    for t, pid in selected.items():
        s = next((s for s in groups[t] if s.product_id == pid), None)
        if s is None or not s.verdict or s.verdict.decision not in ("allow", "allow_with_permit"):
            raise ValueError(f"Choose a checked offer for {groups[t][0].label}.")
        if s.price_quote is None and require_quotes:
            raise ValueError(f"The delivered price for {s.title} is unavailable; choose another offer.")
        if not within(unit_cost(s, units[s.item_type]), caps.get(t)):
            raise ValueError(f"{s.title} exceeds the {fmt_sgd(caps[t])} category cap after delivery.")
        items.append(s)
    if not items and not any(not s.product_id and s.status != "rejected" and s.verdict and
                            s.verdict.decision in ("allow", "allow_with_permit") for s in suggestions):
        raise ValueError("There are no checked items to approve. Resolve pending work or adjust the budget.")
    totals = summary(items)
    if project and project.total_budget is not None and totals["total_sgd"] > project.total_budget:
        raise ValueError(f"This basket costs {fmt_sgd(totals['total_sgd'])}, above your total budget of "
                         f"{fmt_sgd(project.total_budget)}. Choose cheaper offers or explicitly change your budget.")
    works = work_totals(project, suggestions)
    if works["remaining_sgd"] is not None and works["remaining_sgd"] < 0:
        raise ValueError("The entered work estimates exceed your works budget. Adjust the work scope or budget.")
    return items, totals


def work_totals(project: ProjectProfile | None, suggestions: list[Suggestion]) -> dict:
    """Owner-entered work estimates; a missing quote never becomes a zero-cost work."""
    works = {s.key: s.title for s in suggestions if not s.product_id and s.status != "rejected"}
    rows = [r for r in (project.work_estimates if project else []) if not r.request_id or r.request_id in works]
    missing = [title for key, title in works.items() if not any(r.request_id == key for r in rows)]
    unresolved = [r.description for r in rows if r.total_sgd is None or not r.source.strip() or r.checked_at is None]
    subtotal = float(sum((money(r.total_sgd) for r in rows if r.total_sgd is not None), money(0)))
    return {"subtotal_sgd": subtotal, "complete": not missing and not unresolved,
            "missing_requests": missing, "unpriced_rows": unresolved, "rows": rows,
            "remaining_sgd": None if not project or project.works_budget is None else
                float(money(project.works_budget) - money(subtotal))}

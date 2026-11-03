"""The pricing node: shortlist -> pricing -> planner.

For every live product suggestion:
  1. landed cost (deterministic)                         -> Suggestion.price_quote
  2. comparable offers at other retailers (model-judged) -> Suggestion.comparables: the 2 cheapest
     real offers, then at most 1 illustrative row, which can never push a real offer out
  3. budget re-check on LANDED cost (hard rule): over the cap -> removed (by "pricing");
     fewer than MIN_IN_BUDGET left for a type -> open issue for the owner
  4. FX older than 24 h -> escalation note
Already-priced suggestions are left alone, so a re-shortlist only prices the new ones.
"""
from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from renopilot import config, trace
from renopilot.basket import category_units, unit_cost
from renopilot.catalog import store
from renopilot.catalog.fit import fit_note
from renopilot.pricing.comparables import MAX_PAIRS_PER_CALL, find_candidates, judge_comparables
from renopilot.pricing.fx import FxProvider
from renopilot.pricing.landed_cost import landed_cost
from renopilot.pricing.rules import load_rules
from renopilot.schemas import PricedAlternative, Product
from renopilot.tools.budget import request_caps, fmt_sgd, within
from renopilot.vocab import LOOSE_FURNITURE

KEEP_COMPARABLES = 2
KEEP_ILLUSTRATIONS = 1   # made-up rows (synthetic=true) show the overseas method; they are not offers
COMPARE_TOP_N = 2        # comparables only for the first N suggestions per item type (the likely buys)


@lru_cache(maxsize=1)
def _catalog() -> tuple[dict[str, Product], list[Product]]:
    products, _ = store.load_all()
    return {p.id: p for p in products}, products


def make_pricing(products: dict[str, Product] | None = None, pool: list[Product] | None = None,
                 index=None, fx: FxProvider | None = None, judge_model=None, comparables: bool = True):
    """Factory so tests can inject a small catalog, a fake judge and a frozen FX."""

    def pricing(state: dict) -> dict:
        rules = load_rules()
        fxp = fx or FxProvider(stale_hours=rules.fx_stale_hours)
        by_id, all_products = (products, pool) if products is not None else _catalog()
        idx = index
        if idx is None and comparables and products is None:     # an injected catalog has no embeddings
            from renopilot.nodes.style import get_index
            idx = get_index()

        sugg = list(state.get("suggestions") or [])
        profile = state.get("style_profile")
        def spec_for(s):
            matches = [i for i in profile.items if (i.request_id == s.request_id if s.request_id else
                       i.item_type == s.item_type and i.room == s.room)] if profile else []
            return matches[0] if len(matches) == 1 else None
        todo = [i for i, s in enumerate(sugg) if s.status != "rejected" and s.product_id
                and s.work_type == "loose_furniture" and s.price_quote is None and s.product_id in by_id]
        stale = False
        for i in todo:
            try:
                q = landed_cost(by_id[sugg[i].product_id], rules, fxp)
            except Exception as e:                       # noqa: BLE001 - unpriceable stays unpriced, logged
                trace.log("pricing", "unpriced", product=sugg[i].product_id, error=str(e))
                continue
            stale |= q.fx_stale
            sugg[i] = sugg[i].model_copy(update={"price_quote": q, "source": by_id[sugg[i].product_id].source})

        # ---- comparables for the top suggestions per type; pairs batched into few judge calls
        if comparables and todo:
            seen_per_type: dict[str, int] = {}
            targets = []
            for i, s in enumerate(sugg):                 # shortlist order = rank order
                if s.status == "rejected" or s.work_type != "loose_furniture" or s.product_id not in by_id:
                    continue
                seen_per_type[s.item_type] = seen_per_type.get(s.item_type, 0) + 1
                if i in todo and seen_per_type[s.item_type] <= COMPARE_TOP_N:
                    targets.append(i)
            def excluded_for(s):
                return {r.get("product_id") for r in state.get("rejected") or [] if r.get("by") != "pricing"
                        and (not r.get("request_id") or r["request_id"] == s.request_id)}
            groups = [(by_id[sugg[i].product_id], find_candidates(by_id[sugg[i].product_id],
                        [p for p in all_products if p.id not in excluded_for(sugg[i])], idx))
                      for i in targets]
            groups = [g for g in groups if g[1]]
            chunks, cur, n = [], [], 0
            for g in groups:
                if cur and n + len(g[1]) > MAX_PAIRS_PER_CALL:
                    chunks.append(cur)
                    cur, n = [], 0
                cur.append(g)
                n += len(g[1])
            if cur:
                chunks.append(cur)
            with ThreadPoolExecutor(max_workers=3) as pool_:
                futures = [pool_.submit(contextvars.copy_context().run, judge_comparables, ch, judge_model)
                           for ch in chunks]
                verdicts = {}
                for f in futures:
                    verdicts.update(f.result())
            for i in todo:
                alts = []
                for cand, why in verdicts.get(sugg[i].product_id, []):
                    if cand.id in excluded_for(sugg[i]):
                        continue
                    spec = spec_for(sugg[i])
                    fits, note = fit_note(cand, spec) if spec else (None, "Fit not checked: size limits unavailable.")
                    if fits is False:
                        continue
                    try:
                        cq = landed_cost(cand, rules, fxp)
                    except Exception:                    # noqa: BLE001
                        continue
                    stale |= cq.fx_stale
                    alts.append(PricedAlternative(product_id=cand.id, source=cand.source, title=cand.title,
                                                  url=cand.url, quote=cq, why_comparable=why, image_path=cand.image_path,
                                                  fit_note=note, retrieved_at=cand.retrieved_at))
                alts.sort(key=lambda a: a.quote.total_sgd)
                real = [a for a in alts if not a.quote.synthetic][:KEEP_COMPARABLES]
                shown = real + [a for a in alts if a.quote.synthetic][:KEEP_ILLUSTRATIONS]
                sugg[i] = sugg[i].model_copy(update={"comparables": shown})

        # ---- budget on landed cost (hard rule)
        requests = [i for i in profile.items if i.item_type in LOOSE_FURNITURE] if profile else list(
            {s.selection_key: s for s in sugg if s.work_type == "loose_furniture"}.values())
        caps = request_caps(state.get("project"), requests)
        units = category_units(requests)
        removed, types_hit = [], set()
        for i, s in enumerate(sugg):
            q = s.price_quote
            if s.status == "rejected" or q is None or within(unit_cost(s, units.get(s.item_type, s.quantity)), caps.get(s.selection_key)):
                continue
            sugg[i] = s.model_copy(update={"status": "rejected"})
            types_hit.add(s.selection_key)
            removed.append({"product_id": s.product_id, "request_id": s.request_id, "item_type": s.item_type, "by": "pricing",
                            "reason": f"over budget after delivery/shipping/GST: {fmt_sgd(unit_cost(s, units.get(s.item_type, s.quantity)))} per unit "
                                      f"> {fmt_sgd(caps[s.selection_key])}"})
        issues = list(state.get("open_issues") or [])
        for t in sorted(types_hit):
            left = sum(1 for s in sugg if s.selection_key == t and s.status != "rejected")
            if left < config.MIN_IN_BUDGET:
                label = next(s for s in sugg if s.selection_key == t).label
                issues.append(f"Only {left} {label} option(s) stay within {fmt_sgd(caps[t])} once "
                              "delivery, shipping and GST are added: relax the budget for this item?")
        escalations = ["Exchange rates are more than 24 hours old; overseas prices may have moved."] if stale else []

        n_priced = sum(1 for s in sugg if s.price_quote is not None)
        n_alts = sum(len(s.comparables) for s in sugg)
        trace.log("pricing", "result", priced=n_priced, comparables=n_alts, removed=len(removed), fx_stale=stale)
        return {"suggestions": sugg, "rejected": removed, "open_issues": issues, "escalations": escalations,
                "trace": [{"node": "pricing", "action": "priced",
                           "why": f"{len(todo)} newly priced, {n_alts} comparables, {len(removed)} over budget"}]}

    return pricing


pricing = make_pricing()

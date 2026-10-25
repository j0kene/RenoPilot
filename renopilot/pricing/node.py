"""The pricing node: shortlist -> pricing -> planner.

For every live product suggestion:
  1. landed cost (deterministic)                         -> Suggestion.price_quote
  2. comparable offers at other retailers (model-judged) -> Suggestion.comparables (2 cheapest)
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
from renopilot.catalog import store
from renopilot.pricing.comparables import MAX_PAIRS_PER_CALL, find_candidates, judge_comparables
from renopilot.pricing.fx import FxProvider
from renopilot.pricing.landed_cost import landed_cost
from renopilot.pricing.rules import load_rules
from renopilot.schemas import PricedAlternative, Product
from renopilot.tools.budget import caps_for, fmt_sgd, within

KEEP_COMPARABLES = 2
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
        if idx is None and comparables:
            from renopilot.nodes.style import get_index
            idx = get_index()

        sugg = list(state.get("suggestions") or [])
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
            sugg[i] = sugg[i].model_copy(update={"price_quote": q})

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
            groups = [(by_id[sugg[i].product_id], find_candidates(by_id[sugg[i].product_id], all_products, idx))
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
                    try:
                        cq = landed_cost(cand, rules, fxp)
                    except Exception:                    # noqa: BLE001
                        continue
                    stale |= cq.fx_stale
                    alts.append(PricedAlternative(product_id=cand.id, source=cand.source, title=cand.title,
                                                  url=cand.url, quote=cq, why_comparable=why))
                alts.sort(key=lambda a: a.quote.total_sgd)
                sugg[i] = sugg[i].model_copy(update={"comparables": alts[:KEEP_COMPARABLES]})

        # ---- budget on landed cost (hard rule)
        caps = caps_for(state.get("project"), [s.item_type for s in sugg])
        removed, types_hit = [], set()
        for i, s in enumerate(sugg):
            q = s.price_quote
            if s.status == "rejected" or q is None or within(q.total_sgd, caps.get(s.item_type)):
                continue
            sugg[i] = s.model_copy(update={"status": "rejected"})
            types_hit.add(s.item_type)
            removed.append({"product_id": s.product_id, "item_type": s.item_type, "by": "pricing",
                            "reason": f"over budget after delivery/shipping/GST: {fmt_sgd(q.total_sgd)} "
                                      f"> {fmt_sgd(caps[s.item_type])}"})
        issues = list(state.get("open_issues") or [])
        for t in sorted(types_hit):
            left = sum(1 for s in sugg if s.item_type == t and s.status != "rejected")
            if left < config.MIN_IN_BUDGET:
                issues.append(f"Only {left} {t.replace('_', ' ')} option(s) stay within {fmt_sgd(caps[t])} once "
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

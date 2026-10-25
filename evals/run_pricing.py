"""Comparable-judging eval: does the pricing agent pair the RIGHT alternatives?

    uv run python evals/run_pricing.py [--n-runs 5]

28 labelled (shortlisted, candidate) pairs, all the same item type: 17 comparable and 11 hard
negatives (2-seater vs U-shape sectional, 80 cm round vs 180 cm oval table, ...). Labels were
set from size and form by Claude and are flagged for team review (evals/cases/comparables.jsonl).

Each run makes ONE batched judge call over all pairs (as the pricing node does), then the code
dims rule is applied, exactly as in production. A pair PASSES if the final verdict matches the label.

Baselines (deterministic, no model):
  type_only   every same-type candidate counts as comparable (what retrieval alone gives you)
  dims_only   comparable unless the dims rule rejects it
The proposal's bar for this capability is precision: a wrong "cheaper comparable" misleads the owner.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.harness import AgentResult, Case, CaseOutcome, Report, bootstrap_ci  # noqa: E402
from renopilot import config, trace  # noqa: E402
from renopilot.catalog import store  # noqa: E402
from renopilot.pricing.comparables import dims_compatible, judge_comparables  # noqa: E402


def load() -> list[Case]:
    rows = [json.loads(line) for line in (config.ROOT / "evals" / "cases" / "comparables.jsonl").open(encoding="utf-8")]
    return [Case(id=r["id"], input=r["input"], expected=r["expected"], tags=r["tags"]) for r in rows]


def prf(pred: dict[str, bool], cases: list[Case]) -> dict[str, float]:
    tp = sum(pred[c.id] and c.expected for c in cases)
    fp = sum(pred[c.id] and not c.expected for c in cases)
    fn = sum((not pred[c.id]) and c.expected for c in cases)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": p, "recall": r, "accuracy": sum(pred[c.id] == c.expected for c in cases) / len(cases)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=5)
    args = ap.parse_args()
    cases = load()
    by_id = {p.id: p for p in store.load_all()[0]}
    groups: dict[str, list] = {}
    for c in cases:
        groups.setdefault(c.input["shortlisted"], []).append(by_id[c.input["candidate"]])
    batch = [(by_id[s], cands) for s, cands in groups.items()]

    # deterministic baselines
    base = {"type_only": {c.id: True for c in cases},
            "dims_only": {c.id: dims_compatible(by_id[c.input["shortlisted"]].dims, by_id[c.input["candidate"]].dims)
                          for c in cases}}

    model_name = config.MODELS[config.BACKEND]["rerank"]
    out = config.ROOT / "evals" / "results" / f"pricing_comparables_{model_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    runs = [d["pred"] for d in done][: args.n_runs]
    for r in range(len(runs), args.n_runs):
        t0 = time.perf_counter()
        with trace.tracing() as t:
            verdict = judge_comparables(batch)
        yes = {(s, c.id) for s, lst in verdict.items() for c, _ in lst}
        pred = {c.id: (c.input["shortlisted"], c.input["candidate"]) in yes for c in cases}
        runs.append(pred)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run": r, "pred": pred, "latency_s": time.perf_counter() - t0,
                                "run_id": t.run_id, **t.totals()}) + "\n")
        print(f"  run {r + 1}/{args.n_runs}: {prf(pred, cases)}", flush=True)

    # report through the harness (per-pair pass = verdict matches label)
    outcomes = [CaseOutcome(c.id, pred[c.id] == c.expected, i, AgentResult(output=pred[c.id]), c.tags)
                for i, pred in enumerate(runs) for c in cases]
    rep = Report(name=f"judge ({model_name}) + dims rule", outcomes=outcomes, n_runs=len(runs))
    print("\n" + rep.render())
    print("\nprecision / recall / accuracy (mean over runs, sd):")
    for k in ("precision", "recall", "accuracy"):
        vals = [prf(p, cases)[k] for p in runs]
        print(f"  judge     {k:<9} {statistics.fmean(vals):.3f}  (sd {statistics.stdev(vals) if len(vals) > 1 else 0:.3f})")
    for name, pred in base.items():
        m = prf(pred, cases)
        print(f"  {name:<9} precision {m['precision']:.3f}  recall {m['recall']:.3f}  accuracy {m['accuracy']:.3f}")
    lo, hi = bootstrap_ci([statistics.fmean(p[c.id] == c.expected for p in runs) for c in cases])
    print(f"  judge per-pair accuracy 95% CI over pairs: {lo:.3f}-{hi:.3f}")


if __name__ == "__main__":
    main()

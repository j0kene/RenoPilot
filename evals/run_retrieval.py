"""Retrieval eval: does the right product come back in the top 10?

    uv run python evals/run_retrieval.py [--n-runs 5] [--cases 10] [--embed ViT-B-32:laion2b_s34b_b79k]

One VLM call per board per run (style extraction), then every search configuration is
scored on that same profile. The comparison between configurations is therefore paired:
same boards, same model output, only the retrieval differs.

  full          text_img + board_img + bm25, fused        (the system)
  text_only     bm25 on the VLM's item descriptions       (proposal baseline: describe in words, search text)
  clip_text     text_img only                             (ablation)
  image_only    board_img only, VLM-predicted type        (ablation)
  no_bm25       text_img + board_img                      (ablation)

A target counts as missed, tagged style_misread, if the VLM did not list its item type.
A case PASSES if recall@10 >= 0.6 on that board. This is an ABO retrieval diagnostic;
the proposal's final SG shortlist Precision@5 is measured separately by run_shortlist.py.
Results are appended to evals/results/ so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.harness import AgentResult, CaseOutcome, Report, load_cases, run_eval  # noqa: E402
from renopilot import config, trace  # noqa: E402
from renopilot.catalog.embed import get_embedder  # noqa: E402
from renopilot.catalog.search import CatalogIndex  # noqa: E402
from renopilot.nodes.style import extract_style_profile, query_text, resolve  # noqa: E402

CONFIGS = {
    "full": ("text_img", "board_img", "bm25"),
    "text_only": ("bm25",),
    "clip_text": ("text_img",),
    "image_only": ("board_img",),
    "no_bm25": ("text_img", "board_img"),
}
K = 10
PASS_AT = 0.6


def make_agent(index: CatalogIndex):
    emb = get_embedder(index.embed_spec)

    def agent(inp: dict) -> AgentResult:
        t0 = time.perf_counter()
        with trace.tracing() as t:
            profile, usage = extract_style_profile(inp["board"])
            board_emb = emb.images([resolve(p) for p in inp["board"]])
            specs = {s.item_type: s for s in profile.items}
            per_config: dict[str, list[dict]] = {}
            for name, signals in CONFIGS.items():
                rows = []
                for target in inp["_expected"]:
                    spec = specs.get(target["item_type"])
                    if spec is None:
                        rows.append({**target, "rank": None, "why": "style_misread"})
                        continue
                    hits = index.search(query_text(spec, profile), item_type=spec.item_type, board_emb=board_emb,
                                        sources=("abo",), signals=signals, k=50)
                    ids = [h.product.id for h in hits]
                    rank = ids.index(target["product_id"]) + 1 if target["product_id"] in ids else None
                    rows.append({**target, "rank": rank, "why": None if rank and rank <= K else "retrieval_miss"})
                per_config[name] = rows
            t.log("eval", "retrieval", per_config=per_config)
        metrics = {f"recall@{K}:{n}": sum(1 for r in rows if r["rank"] and r["rank"] <= K) / len(rows)
                   for n, rows in per_config.items()}
        metrics |= {f"mrr:{n}": sum(1 / r["rank"] for r in rows if r["rank"]) / len(rows)
                    for n, rows in per_config.items()}
        tags = sorted({r["why"] for r in per_config["full"] if r["why"]})
        return AgentResult(output={"profile": profile.model_dump(), "per_config": per_config}, metrics=metrics,
                           failure_tags=tags, input_tokens=usage["input_tokens"],
                           output_tokens=usage["output_tokens"], cost_usd=usage["cost_usd"],
                           latency_s=time.perf_counter() - t0, run_id=t.run_id)
    return agent


def project(report: Report, cfg: str) -> Report:
    """The same outcomes, seen through one configuration's metric."""
    outs = []
    for o in report.outcomes:
        m = o.result.metrics
        r = replace(o.result, metrics={f"recall@{K}": m.get(f"recall@{K}:{cfg}", 0.0), "mrr": m.get(f"mrr:{cfg}", 0.0)})
        outs.append(CaseOutcome(o.case_id, (not o.result.error) and r.metrics[f"recall@{K}"] >= PASS_AT,
                                o.run_index, r, o.tags))
    return Report(name=cfg, outcomes=outs, n_runs=report.n_runs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--cases", type=int, default=0, help="first N cases only (0 = all)")
    ap.add_argument("--embed", default=config.EMBED_MODEL)
    ap.add_argument("--tag", default="", help="suffix for the results file")
    args = ap.parse_args()

    cases = load_cases(config.ROOT / "evals" / "cases" / "retrieval.jsonl")
    if args.cases:
        cases = cases[: args.cases]
    for c in cases:                          # the adapter needs the targets to score itself
        c.input = {**c.input, "_expected": c.expected}
    index = CatalogIndex.load(args.embed)
    vision = config.MODELS[config.BACKEND]["vision"]
    out = config.ROOT / "evals" / "results" / f"retrieval_{vision}_{args.embed.split(':')[0]}{args.tag}.jsonl"
    print(f"{len(cases)} cases x {args.n_runs} runs; vision={vision}; embed={args.embed}; results -> {out}")

    raw = run_eval(make_agent(index), cases, lambda c, r: True, n_runs=args.n_runs, name="retrieval",
                   resume_file=out)
    reports = {cfg: project(raw, cfg) for cfg in CONFIGS}
    print()
    print(reports["text_only"].render())
    print()
    print(reports["full"].render(baseline=reports["text_only"]))
    print("\nablation (recall@10 mean, sd over runs):")
    for cfg, rep in reports.items():
        s = rep.metric(f"recall@{K}")
        print(f"  {cfg:<11} {s['mean']:.3f}  sd {s['sd']:.3f}  CI {s['ci_lo']:.3f}-{s['ci_hi']:.3f}   "
              f"pass^{args.n_runs} {rep.pass_hat_k:.1%}")
    summary = {cfg: rep.metric(f"recall@{K}") | {"pass_hat_k": rep.pass_hat_k} for cfg, rep in reports.items()}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Style-extraction eval: attribute F1 of the VLM's StyleProfile against two human labellers.

    uv run python evals/run_style.py [--n-runs 5] [--baseline]

Needs data/boards/<id>/ images and data/labels/style_labels_{A,B}.csv filled in
independently by two people (see data/labels/LABELLING.md). Boards labelled by only one
person are skipped.

Per board and run: F1 on each attribute set (style_tags, palette, materials, item_types)
against labeller A and against labeller B, averaged; the board's score is the mean over the
four attributes. A board passes at F1 >= 0.7 (proposal target). The A-vs-B agreement is
printed as the ceiling: the agent cannot be expected to agree with a person more than two
people agree with each other.

--baseline also runs the "no schema" baseline: one generic prompt for a free-text
description, mapped onto the vocabulary by exact word match.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import HumanMessage  # noqa: E402

from evals.harness import AgentResult, Case, run_eval  # noqa: E402
from renopilot import config, trace  # noqa: E402
from renopilot.llm import get_model, image_block, usage_of  # noqa: E402
from renopilot.nodes.style import extract_style_profile  # noqa: E402
from renopilot.vocab import COLOURS, ITEM_TYPES, MATERIALS, STYLE_TAGS  # noqa: E402

ATTRS = ("style_tags", "palette", "materials", "item_types")
VOCAB = {"style_tags": STYLE_TAGS, "palette": COLOURS, "materials": MATERIALS, "item_types": ITEM_TYPES}
PASS_AT = 0.7
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(gold))


def read_labels(who: str) -> dict[str, dict[str, set[str]]]:
    f = config.DATA / "labels" / f"style_labels_{who}.csv"
    out = {}
    if not f.exists():
        return out
    for r in csv.DictReader(f.open(encoding="utf-8")):
        sets = {a: {v.strip() for v in (r.get(a) or "").split(";") if v.strip()} for a in ATTRS}
        if sets["style_tags"]:                                   # an unlabelled row has no style tag
            bad = {a: s - set(VOCAB[a]) for a, s in sets.items() if s - set(VOCAB[a])}
            if bad:
                print(f"  warning: {who}/{r['board_id']} uses words outside the vocab: {bad}")
            out[r["board_id"]] = sets
    return out


def board_files(board_id: str) -> list[str]:
    d = config.BOARDS / board_id
    return sorted(str(p.relative_to(config.ROOT)) for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def score(pred: dict[str, set[str]], gold_a: dict, gold_b: dict) -> dict[str, float]:
    m = {a: (f1(pred[a], gold_a[a]) + f1(pred[a], gold_b[a])) / 2 for a in ATTRS}
    m["f1_mean"] = statistics.fmean(m[a] for a in ATTRS)
    return m


def agent_schema(inp: dict) -> AgentResult:
    t0 = time.perf_counter()
    with trace.tracing() as t:
        p, usage = extract_style_profile(inp["board"])
    pred = {"style_tags": set(p.style_tags), "palette": set(p.palette), "materials": set(p.materials),
            "item_types": {i.item_type for i in p.items}}
    return AgentResult(output={a: sorted(v) for a, v in pred.items()}, metrics=score(pred, inp["A"], inp["B"]),
                       latency_s=time.perf_counter() - t0, run_id=t.run_id, **usage)


GENERIC = "Describe the interior style, the colour palette, the materials and the furniture in this mood board."


def agent_baseline(inp: dict) -> AgentResult:
    t0 = time.perf_counter()
    msg = get_model("vision").invoke([HumanMessage(content=[{"type": "text", "text": GENERIC}] +
                                                   [image_block(config.ROOT / p) for p in inp["board"]])])
    text = msg.content.lower().replace("-", " ")
    pred = {a: {v for v in VOCAB[a] if v.replace("_", " ") in text} for a in ATTRS}
    return AgentResult(output={a: sorted(v) for a, v in pred.items()}, metrics=score(pred, inp["A"], inp["B"]),
                       latency_s=time.perf_counter() - t0, **usage_of(msg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()
    A, B = read_labels("A"), read_labels("B")
    both = sorted(set(A) & set(B))
    if not both:
        sys.exit("No board is labelled by both A and B yet: fill data/labels/style_labels_{A,B}.csv")
    agree = [score(A[b], B[b], B[b]) for b in both]          # A scored against B (B passed twice)
    print(f"{len(both)} boards labelled by both. Inter-rater F1 (ceiling): "
          + ", ".join(f"{a} {statistics.fmean(x[a] for x in agree):.2f}" for a in ATTRS + ("f1_mean",)))

    cases = [Case(id=b, input={"board": board_files(b), "A": A[b], "B": B[b]}, tags=[b.rsplit("_", 1)[0]])
             for b in both]
    grader = lambda c, r: r.metrics["f1_mean"] >= PASS_AT          # noqa: E731
    vision = config.MODELS[config.BACKEND]["vision"]
    res = config.ROOT / "evals" / "results"
    rep = run_eval(agent_schema, cases, grader, n_runs=args.n_runs, name="style (schema+vocab)",
                   resume_file=res / f"style_{vision}.jsonl")
    base = None
    if args.baseline:
        base = run_eval(agent_baseline, cases, grader, n_runs=args.n_runs, name="baseline (free text)",
                        resume_file=res / f"style_baseline_{vision}.jsonl")
        print("\n" + base.render())
    print("\n" + rep.render(baseline=base))


if __name__ == "__main__":
    main()

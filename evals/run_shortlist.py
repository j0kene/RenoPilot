"""Human-rated Precision@5 of final, priced SG shortlists against a paired BM25 baseline.

--prepare runs REAL models/catalog, freezes the actual top-five outputs, and creates a
CSV + local HTML review sheet. --score never calls a model and refuses missing labels.
No model-derived labels are substituted for human relevance judgements.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.harness import AgentResult, CaseOutcome, Report, evaluation_fingerprints, load_cases, run_eval  # noqa: E402
from renopilot import config, trace  # noqa: E402
from renopilot.nodes.style import build_shortlist, extract_style_profile  # noqa: E402
from renopilot.pricing.node import pricing  # noqa: E402
from renopilot.schemas import ProjectProfile  # noqa: E402

RESULTS = config.ROOT / "evals" / "results" / "shortlist_sg.jsonl"
LABELS = config.DATA / "labels" / "shortlist_relevance.csv"
FIELDS = ["fingerprint", "case_id", "run_index", "variant", "rank", "board_id", "item_type", "product_id",
          "title", "image_path", "url", "relevant", "labelled_by", "notes"]


def revision():
    return f"SG-top5:{config.BACKEND}:{config.MODELS[config.BACKEND]}:{config.PICKER}:{config.EMBED_MODEL}"


def agent(inp):
    t0 = time.perf_counter()
    board = [str(p) for p in sorted((config.BOARDS / inp["board_id"]).glob("*.jpg"))]
    with trace.tracing() as t:
        profile, _ = extract_style_profile(board, brief=f"Find a {inp['item_type'].replace('_', ' ')} matching this board.")
        project = ProjectProfile.model_validate(inp["project"])
        variants = {}
        for name, kwargs in (("agent", {}), ("text_only", {"signals": ("bm25",), "use_rerank": False})):
            picks, _ = build_shortlist(profile, project, board, item_types=[inp["item_type"]], **kwargs)
            priced = pricing({"suggestions": picks, "style_profile": profile, "project": project})["suggestions"]
            variants[name] = [s.model_dump(mode="json") for s in priced if s.product_id and s.status != "rejected"][:5]
        totals = t.totals()
    return AgentResult(output=variants, latency_s=time.perf_counter() - t0, run_id=t.run_id,
        input_tokens=totals["input_tokens"], output_tokens=totals["output_tokens"], cost_usd=totals["cost_usd"])


def label_key(row):
    return tuple(str(row[k]) for k in ("fingerprint", "case_id", "run_index", "variant", "rank", "product_id"))


def prepare(cases, n_runs):
    report = run_eval(agent, cases, lambda c, r: True, n_runs=n_runs,
                      name="shortlist preparation (not relevance scoring)", resume_file=RESULTS, revision=revision())
    old = {label_key(r): r for r in csv.DictReader(LABELS.open(encoding="utf-8"))} if LABELS.exists() else {}
    by_id = {c.id: c for c in cases}
    rows, sections = [], []
    for outcome in report.outcomes:
        if outcome.result.error:
            continue
        case = by_id[outcome.case_id]
        board_id = case.input["board_id"]
        board_html = "".join(f'<img width="160" src="{html.escape(os.path.relpath(p, LABELS.parent))}">'
            for p in sorted((config.BOARDS / board_id).glob("*.jpg")))
        cards = []
        for variant, suggestions in outcome.result.output.items():
            for rank, s in enumerate(suggestions, 1):
                row = dict.fromkeys(FIELDS, "") | {"fingerprint": outcome.fingerprint, "case_id": case.id,
                    "run_index": outcome.run_index, "variant": variant, "rank": rank, "board_id": board_id,
                    "item_type": case.input["item_type"], "product_id": s["product_id"], "title": s["title"],
                    "image_path": s["image_path"], "url": s["url"]}
                row |= {k: old.get(label_key(row), {}).get(k, "") for k in ("relevant", "labelled_by", "notes")}
                rows.append(row)
                photo = html.escape(os.path.relpath(config.ROOT / s["image_path"], LABELS.parent)) if s["image_path"] else ""
                cards.append(f'<figure><img width="160" src="{photo}"><figcaption>{html.escape(variant)} rank {rank}: '
                    f'{html.escape(s["title"])}<br>{html.escape(s["product_id"])}</figcaption></figure>')
        sections.append(f'<details><summary>{case.id}, run {outcome.run_index}: {board_id}, '
            f'{html.escape(case.input["item_type"])}</summary>{board_html}<div class="cards">{"".join(cards)}</div></details>')
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    # Preserve annotations for other batches/revisions instead of overwriting a labeller's work.
    keys = {label_key(r) for r in rows}
    rows += [r for k, r in old.items() if k not in keys]
    with LABELS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    LABELS.with_suffix(".html").write_text('<!doctype html><meta charset="utf-8"><title>Shortlist review</title>'
        '<style>body{font:16px sans-serif;margin:30px}img{object-fit:contain;height:150px}.cards{display:flex;flex-wrap:wrap}'
        'figure{width:180px}details{margin:24px 0}</style><h1>Human shortlist relevance review</h1>'
        '<p>Fill relevant=1 or 0 and labelled_by in shortlist_relevance.csv. Judge fit to the board, category, '
        'form, colour and material. Ignore model reasons. See docs/HUMAN_HANDOFF.md for the rubric.</p>'
        + "".join(sections), encoding="utf-8")
    print(f"Prepared {len(report.outcomes)} case-runs. Label {LABELS}; preview {LABELS.with_suffix('.html')}.")
    errors = [o for o in report.outcomes if o.result.error]
    if errors:
        raise SystemExit(f"{len(errors)} preparation runs failed; inspect {RESULTS} before labelling.")


def score(cases, n_runs):
    fingerprints = evaluation_fingerprints(cases, revision())
    snapshots = {}
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            row = json.loads(line)
            if row.get("fingerprint") == fingerprints.get(row["case_id"]):
                snapshots[row["case_id"], row["run_index"]] = row
    labels = {label_key(r): r for r in csv.DictReader(LABELS.open(encoding="utf-8"))} if LABELS.exists() else {}
    reports = {variant: Report(variant, [], n_runs) for variant in ("agent", "text_only")}
    missing = []
    for case in cases:
        for run_index in range(n_runs):
            record = snapshots.get((case.id, run_index))
            if not record or record["result"]["error"]:
                missing.append(f"{case.id}/run {run_index}: prepare a successful snapshot")
                continue
            for variant in reports:
                relevant = 0
                for rank, item in enumerate(record["result"]["output"][variant], 1):
                    key = tuple(str(v) for v in (record["fingerprint"], case.id, run_index, variant, rank, item["product_id"]))
                    label = labels.get(key, {})
                    if label.get("relevant") not in ("0", "1") or not label.get("labelled_by", "").strip():
                        missing.append(f"{case.id}/run {run_index}/{variant}/{rank}: human label")
                    else:
                        relevant += int(label["relevant"])
                p5 = relevant / 5  # absent ranks count as misses, never inflate short shortlists
                result = AgentResult(output={}, metrics={"precision@5": p5})
                reports[variant].outcomes.append(CaseOutcome(case.id, p5 >= .6, run_index, result, case.tags))
    if missing:
        raise SystemExit(f"Pending {len(missing)} preparation/label entries; no relevance score reported.\n" + "\n".join(missing[:10]))
    print(reports["text_only"].render())
    print(reports["agent"].render(baseline=reports["text_only"]))


def main():
    ap = argparse.ArgumentParser()
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--score", action="store_true")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--cases", type=int, default=0)
    args = ap.parse_args()
    cases = load_cases(config.ROOT / "evals" / "cases" / "shortlist_sg.jsonl")
    if args.cases:
        cases = cases[:args.cases]
    (prepare if args.prepare else score)(cases, args.n_runs)


if __name__ == "__main__":
    main()

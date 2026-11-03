# RenoPilot evaluation runs, 2 November 2026

The final evaluation on the frozen code. Two machines ran it: Jolene ran every suite except end-to-end live, and a
teammate ran end-to-end live. The runs from 30 to 31 October, on older models and code, are in `../results_10_30/`.

- Code: `nov-eval-codefreeze` at `417b9ec`. `master` differs from it only in `README.md`. The teammate's end-to-end
  fingerprints do not match `417b9ec`, so the exact version of that run is still to be confirmed.
- Backend: `claude_cli` (Claude Code CLI 2.1.316, `claude -p`, subscription login).
- Models: vision, planner, rerank and intake `claude-opus-5-5`; shoppers `claude-sonnet-5-5`; effort `medium`;
  `RENOPILOT_FX=snapshot`. Effort is not part of a result's fingerprint, so it is stated here.
- Catalog: the committed `data/sg_catalog.csv` snapshot, SigLIP `ViT-B-16-SigLIP:webli`; retrieval uses the ABO subset.
- Langfuse was on for graph and end-to-end runs. Every run's trace is also written to `runs/*.jsonl`; the traces for
  the guard-removed and no-tools end-to-end rows are copied to `traces/`.

## Results

Cost is the API-equivalent price the CLI reports, not what the subscription charges.

| Suite | File | Size | Result | pass^k | Mean latency | Cost |
|---|---|---|---|---|---|---|
| End-to-end, live (teammate) | `end_to_end_live.jsonl` | 10 × 5 | **100%**, 0 violations | 100% | 100.2s | $46.37 |
| Compliance Q&A, live | `compliance_qa_live.jsonl` | 55 × 5 | **98.2%**, sd 0 (CI 94.5–100%) | 98.2% | 5.8s | $5.45 |
| Compliance Q&A, no tools | `compliance_qa_no_tools.jsonl` | 55 × 5 | 70.9% decision correct, 20.7% wrong go-aheads | 63.6% | 12.4s | $8.34 |
| Red-team, live | `compliance_redteam_live.jsonl` | 17 × 5 | **100%** | 100% | 5.5s | $1.50 |
| Red-team, no tools | `compliance_redteam_no_tools.jsonl` | 17 × 5 | 88.2%, 10.6% wrong go-aheads | 88.2% | 10.9s | $2.30 |
| Prompt injection, live | `compliance_injection_live.jsonl` | 10 × 5 | **0 of 50 attacks succeeded** | 100% | 28.2s | $4.62 |
| Contract: Q&A, red-team, injection | `compliance_*_contract.jsonl` | 55, 20, 10 × 5 | 100% | 100% | – | $0 |
| Graph, LLM planner | `graph_llm_claude-opus-5-5_planner-tools-v2.jsonl` | 14 × 5 | **100%** | 100% | 23.7s | $5.45 |
| Graph, fixed planner | `graph_fixed_na.jsonl` | 14 × 1 | 92.9% (fails g05, owner feedback) | – | – | $0 |
| Retrieval | `retrieval_claude-opus-5-5_ViT-B-16-SigLIP.jsonl` | 50 × 5 | recall@10 **0.647** (CI 0.582–0.709); text-only 0.447 | 54% | 10.8s | $7.97 |
| Comparable judge | `pricing_comparables_claude-opus-5-5.jsonl` | 56 pairs × 5 | real pairs precision and accuracy 1.00 (type-only 0.61); illustrative 0.929 | – | – | not recorded |
| Guard removed, live | `end_to_end_live_no_guard.jsonl` | 8 × 2 | 1.06 uncleared works per journey (CI 0.69–1.31) | – | 168.5s | $22.18 |
| No-tools end-to-end | `end_to_end_no_tools.jsonl` | 10 × 2 | rubric 100%, but 0 of 114 products on the catalog | – | 74.0s | $3.56 |
| Demo, with and without guard | `end_to_end_demo*.jsonl` | 10 × 5 each | 100%; without guard 20%, 1.1 violations per run | – | – | $0 |
| Shortlist prepare | `shortlist_sg.jsonl`, `labels/` | 30 × 1 | 300 picks waiting for human relevance labels | – | 53.9s | $8.95 |

Across live Q&A and red-team, the agent gave 0 wrong go-aheads in 360 runs, against 66 of 360 for the same model
with no tools. The no-tools baselines are graded on the decision alone, because they cannot cite a rule.

## Failure modes

| Failure | Where | How often | Cause |
|---|---|---|---|
| Style misread | Retrieval | 219 of 880 targets (24.9%) | The vision model left the item type out, so it was never searched |
| Retrieval miss | Retrieval | 89 of 880 (10.1%); 38 ranked 11–20 | Listed and searched, but ranked below 10 |
| Over-escalation | Q&A `cq13` | 5 of 5 runs | The guard's text check reads “no gas pipe enclosed” as both yes and no and escalates. Safe side; found during the freeze, not fixed |
| Comparable missed | Judge, illustrative pairs | 10 of 140 | Genuine matches rejected; no wrong match accepted |

Removing BM25 leaves recall unchanged and raises MRR by 0.083 (CI 0.046–0.122).

## How the runs were checked

- Every counted row has cost above zero and no error, so the models really ran. A run where the CLI failed but code
  fallbacks still passed injection at $0 was discarded.
- Usage limits hit during the runs. Jolene's 8 affected rows were removed and rerun; the teammate's 37 quota-error
  attempts stay in `end_to_end_live.jsonl`, and the scores use the latest completed row per case and run.
- A limit in the middle of a journey can make a shopper fall back silently. No counted row shows this: no limit errors
  in the traces, and no item type whose offers all came from the fallback.

## Run counts

Five runs per case, except guard removed and no-tools end-to-end (2 runs, to manage cost and usage limits), graph
fixed planner (1 run, deterministic) and shortlist prepare (1 run, for human labelling).

## Files

- `*.jsonl`: one line per run of a case (`case_id`, `run_index`, `passed`, `result`, `fingerprint`).
- `end_to_end_live_report.pdf`: the teammate's summary of the end-to-end live run.
- `traces/`: run traces for the guard-removed and no-tools end-to-end rows.
- `labels/`: the shortlist relevance sheet (300 rows) and its preview.
- `logs/`: console output of each suite run, ending in its report.

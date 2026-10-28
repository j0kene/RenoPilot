# Human activities still required

The code changes do not replace independent labels, source verification or an authenticated live run.
The two style sheets have intentionally been left blank. Compliance expected labels are engineering
test expectations derived from the frozen cards; they are not independently verified legal labels.
Scripted-demo and injection-contract results must not be reported as live-model accuracy or resistance.

## 1. Restore model access and complete the live evaluation

**Owner:** the person with the team's Claude account or API credentials. **Required before** generating
shortlist relevance sheets and claiming live end-to-end performance.

The attempted live smoke check returned `Not logged in · Please run /login`. In a local terminal, run
`claude`, complete `/login`, then exit; alternatively configure the supported token/key in `.env` as
described in `.env.example`. Keep credentials out of Git and report artifacts. Verify model access with
`RENOPILOT_LIVE=1 python -m uv run pytest tests/test_llm_live.py` (PowerShell: set the environment
variable with `$env:RENOPILOT_LIVE='1'` first).

Then run the real suites with the team's chosen model configuration:

```bash
python -m uv run python evals/run_end_to_end.py --mode live --n-runs 5
python -m uv run python evals/run_compliance.py --live --n-runs 5
```

Inspect each failure and its `run_id` in `runs/` or Langfuse. Confirm that the output really uses the
uploaded/example images, that a mixed living/bathroom board assigns rooms per item, that requested
demolition is not rewritten as construction, and that the approved basket matches the displayed total.
Repeat after fixing real failures; keep failed outcomes in the record. Code/data/case fingerprints
prevent old successful runs from being reused for changed code. A started or partial run is not pass^5.

**Complete when:** saved live reports cover all intended cases and five repetitions, failures are
explained or fixed, and one team member has inspected the resulting outputs and costs. Passing
scripted contracts alone does not complete this activity.

## 2. Independently label the style boards

**Owners:** two different human labellers, A and B. **Workload:** 27 boards × two people = 54 board
reviews, approximately 1–1.5 hours per person. Both sheets currently have zero completed boards.

Open all four photos in each `data/boards/<board_id>/` folder. A fills
`data/labels/style_labels_A.csv`; B fills `data/labels/style_labels_B.csv`. Use
`data/labels/LABELLING.md` for the exact vocabulary and rubric. Each row needs 1–3 style tags, 2–6
palette entries, visible materials and the distinct visible item types; separate values with semicolons.
Use notes for conflicting images, unclear objects or images that do not belong in the board.

Label independently, without discussing the rows, reading model predictions, or treating the folder
name as the answer. Do not fill one sheet by copying the other. Identify each labeller in the report.
Freeze both sheets before running the scorer:

```bash
python -m uv run python evals/run_style.py --n-runs 5 --baseline
```

Review inter-rater agreement and disagreements after both sheets are frozen. Preserve original labels
if any adjudicated version is created; do not revise labels to improve the agent's score.

**Complete when:** every board has two independent valid rows and the report contains agreement,
per-attribute F1, baseline comparison and repeated-run results. The proposal mentioned 30 boards;
either accurately report the actual 27, or curate three more boards and label them independently
before claiming the 30-board target.

## 3. Judge final Singapore shortlist relevance

**Owner:** a human familiar with the relevance rubric; a second person should resolve uncertain
judgements. **Workload:** 30 item/board cases × five runs × two variants × up to five items = at most
1,500 product judgements. This is separate from style attribute labelling and ABO retrieval Recall@10.

After model access works, prepare real results from the frozen SG catalog:

```bash
python -m uv run python evals/run_shortlist.py --prepare --n-runs 5
```

Preparation produces `evals/results/shortlist_sg.jsonl`,
`data/labels/shortlist_relevance.csv`, and a local HTML review sheet at
`data/labels/shortlist_relevance.html`. Open the HTML file in a browser to compare the board photos
with each result. Use `case_id`, `run_index`, `variant` and `rank` to locate its CSV row. The variants
are the actual agent shortlist and a paired BM25 text-only baseline using the same extracted profile.
Do not rate a guessed product from an id alone; inspect the photo/title and product page where needed.

Set `relevant=1` only when the item is the requested category and is a plausible visual match in form,
colour and material to the board. Set `relevant=0` for the wrong kind of object or a clear mismatch.
Judge visible evidence, not the model's match reason, brand prestige, or which variant produced it.
Enter your name/identifier in `labelled_by`; put ambiguous cases in `notes` and have the second person
resolve them before scoring. Do not alter ids, ranks, fingerprints or the saved outputs.

```bash
python -m uv run python evals/run_shortlist.py --score --n-runs 5
```

The scorer refuses missing human labels and outdated/missing snapshots. Missing ranks count as misses
in Precision@5; a short result list does not earn an inflated score. Preparation preserves existing
annotations, and annotations only apply to the matching frozen output fingerprint.

**Complete when:** all prepared products have a 0/1 label and labeller, ambiguous cases are resolved,
and the final report includes Precision@5 (target ≥0.60), five-run reliability and the paired baseline.

## 4. Independently verify HDB cards and compliance gold answers

**Owner:** the compliance workstream member; ask HDB or a suitably qualified practitioner when the
source does not resolve a technical condition. **Scope:** all 22 rule cards, 50 QA cases and 20
adversarial request cases. Prompt-injection cases also need a check that they test genuine attacks.

Use `data/labels/compliance_review.csv` to record the reviewer, review date, confirmed status and
correction notes for every rule. Open each source URL from `data/compliance/sources.json` and compare
the exact quote, scope, permit decision, method, conditions and alternative with
`data/compliance/hdb_rules.jsonl`. A quote appearing on a saved page proves provenance, not that the
whole rule interpretation is correct. Secondary contractor/blog sources need confirmation against an
official source before presenting their statements as official guidance.

Pay particular attention to bathroom floor versus wall work, overlay versus removal/replacement,
the correct block-completion date rather than a guessed key-collection date, unknown structural
status, reinforced-concrete elements, and household-shelter exceptions. Check the day before/on the
three-year boundary and the documented 29-February handling. The guard deliberately escalates
unknown facts and treats shelter requests conservatively; confirm that the UI wording makes that clear.

Review the expected verdict and cited rule for every row of `evals/cases/compliance_qa.jsonl` and
`evals/cases/compliance_redteam.jsonl` independently of the current implementation. Correct bad gold
answers and add counterexamples rather than changing expectations merely to make a run pass.

```bash
python -m uv run python evals/run_compliance.py --n-runs 5
```

**Complete when:** all review rows have a human sign-off or an explicit unresolved issue, expected
labels have been independently reviewed, and the report clearly distinguishes deterministic rule
enforcement from live natural-language interpretation. Do not describe unresolved conditions as cleared.

## 5. Replace synthetic overseas offers before claiming real savings

**Owner:** the catalog/pricing workstream member with access to the seller listings. All 18 current
`data/taobao_manual.csv` rows are sample fixtures. The app excludes them from purchasable selections.

For each real replacement, capture a stable product URL/id, listing title, correct item type, seller,
current price/currency and capture date. Match the actual size/variant/material, not only a similar
thumbnail. Record dimensions, packed weight/volume, shipping mode and a seller/forwarder shipping
quote where available; record whether last-mile delivery, insurance and tax are included. Preserve
the source evidence and distinguish quoted values from assumptions. Set `synthetic=false` only for
verified real rows, rebuild the index, and check that the resulting delivered comparison is like for like.

Also verify the freight assumptions in `data/pricing/rules.json`: the existing sea estimate is based
on an old secondary source and the air estimate is an assumption. Obtain current quotes or keep
them prominently marked as estimates. Have the pricing owner check the frozen delivery, FX and
tax-rule inputs against their authoritative sources before making current-price claims.

**Complete when:** the demonstration's real overseas comparisons have working links, correct variants,
dated evidence and defensible delivered-cost inputs. If this work is deferred, present those examples
as synthetic pricing illustrations and do not claim observed savings from real overseas offers.

## 6. Rehearse the final owner journey and approve the report claims

**Owners:** presenter/integration lead, with one other person acting as the homeowner.

Run the app on the actual presentation machine. Confirm product photos and links load, Langfuse is
reachable if shown, and live credentials/internet work. Use offline mode only for the labelled
`japandi_01` scripted scenario; other boards/uploads need Live. Rehearse: change a product selection,
choose a real comparable, reject with feedback, adjust a tight budget, compare old/new/unknown
bathroom dates, explicitly propose a permitted alternative to blocked work, and download/open the CSV.
Verify one chosen offer per category, exact cents, shared delivery, remaining budget, and unresolved
works in the final screen/export. Check that omitted products or pending works are not described as
an entirely completed renovation plan. Sessions currently use an in-memory checkpointer, so a server
restart/reset does not preserve the in-progress project.

Align the slides/report with completed evidence: 27 currently available style boards, the actual SG
catalog snapshot, the synthetic overseas limitation, fixed-date contract versus authenticated live
results, and pending human labels. Preserve model names, case counts, run counts, cost/latency, failures
and baseline comparisons. The current code does not certify a renovation or purchase anything;
actual works still need the owner's fact confirmation, relevant approvals and contractor checks.

**Complete when:** two people can complete the rehearsed journey on the demo machine, the CSV agrees
with the screen, and every performance/data claim in the presentation is backed by the appropriate
saved result and completed human review.

# Changes from the proposal

The proposal listed its targets as hypotheses and the course lets the final submission change them. These items
were dropped on purpose. Each entry says what was promised, why it went, and what the report cannot claim as a result.

| Proposed | Decision | Why | What this costs |
|---|---|---|---|
| Item spotting on COCO indoor photos (at least 80% of labelled items) | Dropped | The product works from the owner's mood board and brief; detection on COCO would measure a capability the journey does not use. | No object-detection figure. Whether the right items were read from a board is covered by style F1. |
| Renovation glossary (about 100 terms) | Not built | Prompts and outputs are constrained to the fixed vocabularies in `renopilot/vocab.py`, so the model cannot answer in a term the catalog or rule table does not know. | Local terms in a free-text brief ("L-box", "foot run") depend on what the model already knows. This was not measured. |
| nDCG@10 for shortlist relevance | Dropped | The owner sees five offers per item, and humans label those five. Ranks 6 to 10 have no labels, so nDCG@10 cannot be computed. | Relevance is reported as Precision@5 only, which ignores order within the five. |
| Faithfulness check on compliance answers | Dropped | The compliance answer is built in code from the rule card (`decision: reason (citation)`), and the model only reads the request. There is no model-written claim to check against the rule. | Whether a rule card itself reflects HDB's text rests on the human review in `docs/HUMAN_HANDOFF.md`, step 4. |
| LLM judge for end-to-end scenarios, calibrated on 15 cases | Replaced | Every end-to-end criterion is a checkable fact (approved or not, within budget, which rule fired, whether the run stopped at the gate), so a rule-based grader is cheaper, repeatable and needs no calibration. | The end-to-end score says nothing about how well the reply is written or how well the picks match the board; relevance comes from the human Precision@5 suite. |
| Real Taobao listings in the price catalog | Replaced with illustrations | Taobao sits behind a login and real listings may not be collected in time. | No observed overseas prices, so no overseas saving can be claimed. See below. |
| Contractor due diligence (stretch) | Not built | It was a stretch objective, and the three core objectives took priority. | The agent gives no contractor checks and must not be described as doing so. |

## Objective 3 and the overseas rows

Proposed: "Price local and overseas items on the shortlist on full landed cost in SGD: price, exchange rate, shipping,
local delivery and GST where it applies."

Now: **Price real local items (IKEA SG, Castlery) on full landed cost in SGD, including delivery. The overseas method
(exchange rate, shipping, GST) is built and unit-tested, and is shown on clearly labelled illustrative listings because
no real overseas prices were collected.**

| The report may say | The report may not say |
|---|---|
| Landed cost is correct on 30 hand-worked cases, including overseas ones | RenoPilot finds cheaper overseas offers, or any saving figure |
| Local totals use real listed prices from the dated snapshot, with delivery | The catalog holds real Taobao listings, or 312 products (it holds 294 real ones) |
| Exchange rates older than 24 hours are flagged | Overseas totals predict a real delivered price: the freight rates are assumptions |
| The comparable judge scored X on real IKEA SG/Castlery pairs | The illustrative pair scores show accuracy on real overseas listings |

How each part of the evaluation is affected:

| Eval or proposal item | Effect |
|---|---|
| Landed-cost unit tests, FX tests | None. They test arithmetic and rules on hand-written inputs, not listings. |
| Comparable judging | The real pairs are the reported result; the illustrative pairs are reported separately. Labels need human review, and Castlery has no dimensions, so the size rule cannot fire on the real set. |
| Shortlist Precision@5 and retrieval | None. Illustrative rows have no image embedding, so they are not in the search index and never appear in a shortlist. |
| End-to-end scenarios | The proposal's example with two Taobao items is not used; no case depends on overseas rows. The rubric now fails a run if an illustrative row is approved, or appears in the reply or the export. |
| Datasets slide | 294 real SGD listings and 18 illustrative CNY rows, not "price in SGD or CNY" for real items. |
| Autonomy table | FX, GST and landed cost still act on their own, as code. On overseas rows that code only ever acts on illustrations. |

If real listings do arrive, enter them with `synthetic=false` and rebuild the index; they then behave as real offers
with no code change, and `docs/HUMAN_HANDOFF.md` step 5 lists what to capture.

## How the baselines differ from the graph's grading

The spec asks for a baseline, so these exist, and they are not graded identically to the graph. This has to be stated
wherever their numbers appear.

- **No-tools compliance** (`run_compliance.py --baseline`) is graded on the decision alone, because a model with no
  rule table cannot give a rule id. Compare it to the agent's `decision_correct` and `unsafe_clear`, not to the
  agent's strict pass rate. It uses the agent's own intake model, as the proposal specifies. The injection suite is
  skipped because it attacks the shopper's tools, which this baseline does not have.
- **No-tools end-to-end** (`run_end_to_end.py --mode no_tools`) passes even with invented products; `catalog_match`
  reports how many it names are in the snapshot. This favours the baseline. A chat reply also cannot commit to
  anything, so the "owner walks away" case passes without showing any enforced gate.
- **Guard removed** (`run_end_to_end.py --mode live --no-guard`) clears everything it is given. Its `violations`
  metric counts approved work that the real guard would not clear.

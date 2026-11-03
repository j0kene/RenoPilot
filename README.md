# RenoPilot

RenoPilot helps new BTO owners in Singapore go from a mood board to a furniture shortlist they can actually
buy, with prices they can trust and renovation plans that respect HDB rules.

## What it does

You give RenoPilot a few inspiration images, a budget, and some details about your flat. It then:

1. **Reads the style** of your mood board (palette, materials, overall look).
2. **Finds matching products** from real IKEA SG and Castlery listings.
3. **Works out the landed price**, including delivery, which is shared across everything you buy from the
   same retailer.
4. **Checks your renovation plans against HDB rules**, such as the 3-year wait before redoing a bathroom.
5. **Lets you choose.** You pick one offer per item (the same kind of furniture can appear in different
   rooms), set quantities, and export the final list as a CSV.

Renovation costs are tracked separately from the furniture budget. If HDB rules block a piece of work, it
stays blocked. If you choose a different way of doing it, RenoPilot shows you a preview and runs the same
compliance check again before anything is approved.

## Getting started

You'll need Python 3.12. We use [uv](https://docs.astral.sh/uv/) to manage dependencies.

```bash
python -m pip install --user uv
python -m uv sync                  # creates .venv from uv.lock
copy .env.example .env             # on macOS/Linux: cp .env.example .env
python -m uv run pytest            # offline tests, no model calls
```

All configuration lives in `.env`. The comments in [.env.example](.env.example) explain every option, but
these are the ones you're most likely to touch:

- `RENOPILOT_BACKEND` chooses the model provider. The default, `claude_cli`, uses your Claude Code login or
  token. Set it to `openai` to use `OPENAI_API_KEY` instead.
- `RENOPILOT_MODEL_VISION`, `_PLANNER`, `_RERANK`, `_SHOPPER` and `_INTAKE` let you pick a different model
  for each role.
- `RENOPILOT_FX=snapshot` uses our saved exchange rates, so results are reproducible and work offline. If you
  leave it unset, live daily rates come from Frankfurter.

To run the tests that call a real model:
`set RENOPILOT_LIVE=1 && python -m uv run pytest tests/test_llm_live.py`
(on macOS/Linux: `RENOPILOT_LIVE=1 python -m uv run pytest tests/test_llm_live.py`).

## Running the app

The quickest way to see RenoPilot is the Streamlit app in offline mode:

```bash
python -m uv run --group ui streamlit run app/streamlit_app.py
```

Offline mode walks through one scripted Japandi example (`japandi_01`). The models are played by code, so it
shows how the workflow fits together rather than how well the system understands images. Switch to Live mode
to try the other example boards or upload your own.

There is also a command-line version:

```bash
python -m uv run python -m renopilot.run_cli --board data/boards/japandi_01 --budget sofa=1500,coffee_table=400
python -m uv run python -m renopilot.run_cli --board a.jpg b.jpg c.jpg --total 6000 --planner fixed
python -m uv run python -m renopilot.run_cli --board data/boards/japandi_01 --total 6000 \
    --completed 2024-06-01 --flat-type 4-room --brief "we also want to redo the bathroom floor"
```

A few other useful commands:

```bash
python -m uv run python evals/demo_trace.py      # run the offline demo and send the trace to Langfuse
python -m uv run python -m renopilot.graph       # print the agent graph as a Mermaid diagram
```

Every run also saves a local trace to `runs/<run_id>.jsonl`, with one line per step or model call.

### Design decisions worth knowing

**The owner stays in control of the facts.** Budgets, quantities, rooms and renovation details come from form
fields, not from guessing. The flat's completion date matters most, because it decides whether HDB's
3-year bathroom rule applies, so the owner has to confirm it in the date field. A sentence in the free-text
brief isn't enough.

**When unsure, we ask.** If a renovation request isn't supported, or we don't know the work method or the
room, the item is left pending for the owner to review. We never assume it belongs with the most common room
or quietly pick the easiest option.

**Prices are honest about what they don't know.**

- Furniture prices are per unit, and delivery is charged once per retailer order.
- Renovation estimates record quantity, unit, rate, source and the date they were checked. If any of that is
  missing, the estimate is marked incomplete.
- We only use dimensions that are clearly labelled (width, depth, height). Anything ambiguous is shown as
  unknown rather than guessed.
- Every offer shows the date its price was captured.

## Data

All data comes from frozen snapshots, so results are reproducible. Large files aren't committed; the scripts
below rebuild them. [data/SOURCES.md](data/SOURCES.md) lists every source, its licence and when it was
captured.

```bash
python -m uv run python data/scripts/fetch_abo.py              # Amazon Berkeley Objects metadata (~90 MB)
python -m uv run python data/scripts/build_abo_subset.py       # furniture subset + images
python -m uv run python data/scripts/collect_sg_catalog.py     # IKEA SG + Castlery -> data/sg_catalog.csv
python -m uv run python data/scripts/build_index.py            # SQLite database + SigLIP image embeddings
python -m uv run python data/scripts/make_retrieval_boards.py  # retrieval evaluation cases
python -m uv run python data/scripts/fetch_unsplash_boards.py  # style evaluation boards (needs UNSPLASH_ACCESS_KEY)
python -m uv run python data/scripts/fetch_hdb_sources.py      # HDB renovation guidelines behind the rule cards
```

`collect_sg_catalog.py` also accepts `--details-per-type N` to fetch a limited number of product pages per
retailer and item type.

The IKEA SG and Castlery prices are real. The overseas listings are made up purely to illustrate the
landed-cost calculation and should never be read as real offers (more on this in
[docs/SCOPE_CHANGES.md](docs/SCOPE_CHANGES.md)). Manual price checks, each with a date and source link, go in
`data/catalog_overrides.csv`.

## Tracing with Langfuse

We self-host Langfuse to see exactly what the agents are doing.

```bash
docker compose up -d     # Langfuse at http://localhost:3000
```

Log in with `dev@renopilot.local` / `renopilot-local`. The first time it starts, Langfuse creates the project
using the `LANGFUSE_*` keys from your `.env`, so the app and server always match. To turn tracing off, comment
those keys out.

Each conversation from the CLI, the Streamlit app or `evals/run_graph.py` appears as a single trace,
including every pause for owner input. Inside it you can see:

- the master agent's tool calls and each subagent step
- the shopper agents and the tools they call (`search_catalog`, `product_details`, `landed_price`)
- the deterministic tools underneath (catalog search, landed cost, basket total, comparable candidates, HDB
  rule check)
- every model call, with its prompt, reply and token count

The CLI prints a link to the trace when it finishes. [docs/INTERFACES.md](docs/INTERFACES.md) describes how the
master agent and subagents fit together.

## Evaluation

### Main suites

| Command | What it measures | How it's graded |
|---|---|---|
| `python -m uv run python evals/run_retrieval.py --n-runs 5` | Recall@10 on boards built from known ABO products, compared with a text-only baseline and ablations | deterministic |
| `python -m uv run python evals/run_style.py --n-runs 5 --baseline` | Style attribute F1 against human labellers | deterministic |
| `python -m uv run python evals/run_graph.py --n-runs 5` | Whether the agent takes the right route at each decision point (including pricing edge cases: going over budget after landed costs, stale exchange rates, pricing switched off), LLM planner vs a fixed pipeline | trajectory assertions |
| `python -m uv run python evals/run_pricing.py --n-runs 5` | Judging whether two products are comparable, on 28 real IKEA SG/Castlery pairs (17 matches, 11 hard negatives), plus 28 illustrative overseas pairs reported separately, against type-only and dimensions-only baselines | label match |
| `python -m uv run pytest tests/test_landed_cost.py` | 30 landed-cost cases worked out by hand, exact to the cent | exact |

### Compliance, end-to-end and shortlist quality

```bash
# Compliance: 55 rule questions, 20 adversarial requests, 10 prompt-injection cases
python -m uv run python evals/run_compliance.py --n-runs 5
python -m uv run python evals/run_compliance.py --live --n-runs 5        # with a real model
python -m uv run python evals/run_compliance.py --baseline --n-runs 5    # same model, no tools or rule table

# End-to-end: 10 complete shopping journeys on the SG catalog
python -m uv run python evals/run_end_to_end.py --mode demo --n-runs 5
python -m uv run python evals/run_end_to_end.py --mode live --n-runs 5              # real vision, planner, shoppers, pricing
python -m uv run python evals/run_end_to_end.py --mode no_tools --n-runs 5          # baseline: no tools
python -m uv run python evals/run_end_to_end.py --mode live --no-guard --n-runs 5   # ablation: HDB guard removed

# Shortlist quality: real SG top-five results alongside a BM25 baseline
python -m uv run python evals/run_shortlist.py --prepare --n-runs 5
python -m uv run python evals/run_shortlist.py --score --n-runs 5
```

On macOS or Linux, `run_live_eval.sh` starts Langfuse if it isn't already running and then runs the live
evaluations:

```bash
./run_live_eval.sh                           # live end-to-end
./run_live_eval.sh compliance --n-runs 5     # live compliance suites
./run_live_eval.sh cr07 e04                  # just the cases you name
```

Results are appended to `evals/results/*.jsonl`, so an interrupted run picks up where it left off. Each result
is tagged with a fingerprint of the code, data and test case, and is recomputed if any of those change. A
case only counts as passing all five runs (pass^5) if all five actually finished.

### Limitations

We want to be upfront about what these numbers do and don't show:

- The compliance contract tests show that the rules are enforced in code. They don't show that a live model
  resists prompt injection on its own.
- ABO Recall@10 is a diagnostic for the retrieval step. Final shortlist quality on the SG catalog
  (Precision@5) comes from `run_shortlist.py` and depends on human judgements.


## Project structure

```
renopilot/
  config, llm (model per role), schemas + vocab + state (shared contracts), trace
  basket.py      one offer per item; shared money rules for the UI, approval and export
  compliance.py  HDB rule cards (data/compliance/hdb_rules.jsonl)
  catalog/       store (SQLite), embed (open_clip/SigLIP), search (hybrid + RRF), taxonomy, fit
  nodes/         intake, planner (master agent), style + shopper (ReAct subagents), guard (HDB rules),
                 approval (+ ask_user), respond
  pricing/       rules, fx, landed_cost (deterministic), comparables (model-judged), node
  tools/         budget (deterministic money maths)
  demo.py        the offline demo: models played by code over the frozen SG catalog
  graph.py       LangGraph wiring
  run_cli.py     command-line demo
app/             Streamlit UI
data/            scripts/, sg_catalog.csv, boards/, labels/, compliance/, pricing/
                 (raw/, images/ and index/ are rebuilt by the scripts)
evals/           harness.py (pass^k, confidence intervals, resume), cases/, run_*.py
tests/           offline unit and graph tests, live backend smoke tests
docs/            interfaces, scope changes, human handoff
docker-compose.yml   self-hosted Langfuse
```

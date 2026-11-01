# RenoPilot

From mood board to a compliant, priced shortlist for BTO owners in Singapore (CS621 group project).

The integrated demo runs mood board → style and retrieval → landed pricing → HDB check → owner selection.
Choose one offer per furniture category, see the delivered basket total and remaining budget, and export a
CSV shopping list. Each renovation item carries its own room and requested method. Blocked work stays
blocked; an owner-chosen alternative is a separate proposal checked by the guard again.

## Setup (Windows, Python 3.12)

```bash
python -m pip install --user uv
python -m uv sync                      # creates .venv from uv.lock
copy .env.example .env                 # model and Langfuse settings
python -m uv run pytest                # offline tests, no model calls
```

Set `RENOPILOT_BACKEND=claude_cli` (default) or `openai` in `.env`. Claude CLI uses a Claude Code login or token; the OpenAI API backend uses `OPENAI_API_KEY`. Per-role models can be set with `RENOPILOT_MODEL_VISION`, `RENOPILOT_MODEL_PLANNER`, `RENOPILOT_MODEL_RERANK`, `RENOPILOT_MODEL_SHOPPER`, and `RENOPILOT_MODEL_INTAKE`. Live tests: `set RENOPILOT_LIVE=1 && python -m uv run pytest tests/test_llm_live.py`.

## Data (frozen snapshots; large files are gitignored and rebuilt by script)

```bash
python -m uv run python data/scripts/fetch_abo.py              # ABO metadata (~90 MB)
python -m uv run python data/scripts/build_abo_subset.py       # furniture subset + images
python -m uv run python data/scripts/collect_sg_catalog.py     # IKEA SG + Castlery -> data/sg_catalog.csv
python -m uv run python data/scripts/build_index.py            # SQLite + SigLIP embeddings
python -m uv run python data/scripts/make_retrieval_boards.py  # retrieval eval cases
python -m uv run python data/scripts/fetch_unsplash_boards.py  # style eval boards (needs UNSPLASH_ACCESS_KEY)
```

Sources, licences and snapshot dates: [data/SOURCES.md](data/SOURCES.md).

## Run

```bash
python -m uv run --group ui streamlit run app/streamlit_app.py     # offline mode: one labelled scripted Japandi example
python -m uv run python evals/demo_trace.py                          # the same offline demo, scripted, to Langfuse
python -m uv run python -m renopilot.run_cli --board data/boards/japandi_01 --budget sofa=1500,coffee_table=400
python -m uv run python -m renopilot.run_cli --board a.jpg b.jpg c.jpg --total 6000 --planner fixed
python -m uv run python -m renopilot.run_cli --board data/boards/japandi_01 --total 6000 --completed 2024-06-01 --flat-type 4-room --brief "we also want to redo the bathroom floor"
python -m uv run python -m renopilot.graph      # print the graph as mermaid
```

The owner's details go in two ways, merged at intake: form fields / CLI flags (flat type, block completion date,
budgets) always win, and a free-text brief ("What would you like?" / `--brief`) fills anything they leave empty
(read by the intake model; by a small parser in demo mode). The completion date drives HDB's 3-year bathroom rule;
items the brief asks for (e.g. bathroom tiles) are added to the board's items.
Offline mode only accepts `japandi_01`; its fixed profile and code-driven models demonstrate the workflow,
not image understanding. Use Live for other example boards or uploads. Unknown work methods and rooms
stay pending instead of inheriting a majority-room classification or silently taking the easiest route.

Prices: `RENOPILOT_FX=snapshot` uses the frozen FX file (reproducible, offline); the default fetches daily
rates from Frankfurter. Real prices are IKEA SG and Castlery; the overseas rows are made-up illustrations of the
landed-cost method, never offers (see `docs/SCOPE_CHANGES.md`).

Every run writes a JSONL trace to `runs/<run_id>.jsonl`, one row per node or model call.

## Tracing in Langfuse (self-hosted, docker-compose.yml)

`docker compose up -d` starts Langfuse at http://localhost:3000. Log in as `dev@renopilot.local` /
`renopilot-local`; on first start the stack creates the project with the `LANGFUSE_*` keys from `.env`, so the app and
the server always agree (comment the keys out to turn tracing off). Graph runs from the CLI, the Streamlit app and `evals/run_graph.py` then show up as one trace per conversation,
with all the interrupt/resume turns in it: the master agent's tool calls, each subagent node, the shopper agents
and their tool calls (`search_catalog`, `product_details`, `landed_price`), the code-level tools under them
(`@trace.tool`: catalog search, landed cost, basket total, comparable candidates, HDB rule check), and every model
call (prompt, reply, tokens). See [docs/INTERFACES.md](docs/INTERFACES.md) for the master/subagent layout. The CLI prints the trace link at the end.

## Evals

| command | what | grader |
|---|---|---|
| `python -m uv run python evals/run_retrieval.py --n-runs 5` | Recall@10 on boards of known ABO products, vs text-only baseline + ablations | deterministic |
| `python -m uv run python evals/run_style.py --n-runs 5 --baseline` | style attribute F1 vs two human labellers | deterministic |
| `python -m uv run python evals/run_graph.py --n-runs 5` | routes at each boundary (incl. pricing: landed-cost budget removal, stale FX, pricing off), LLM planner vs fixed pipeline | trajectory assertions |
| `python -m uv run python evals/run_pricing.py --n-runs 5` | comparable judging on 28 real IKEA SG/Castlery pairs (17 positive, 11 hard negatives), plus 28 with illustrative overseas rows reported separately, vs type-only and dims-only baselines | label match |
| `python -m uv run pytest tests/test_landed_cost.py` | 30 hand-worked landed-cost cases, exact to the cent | exact |

Results append to `evals/results/*.jsonl`, so an interrupted run resumes where it stopped.
New outcomes include a fingerprint of the code, grounding data, case inputs and expected labels;
outdated outcomes are not reused after changes. Incomplete repetitions do not count as pass^5.

Additional evaluation commands:

```bash
python -m uv run python evals/run_compliance.py --n-runs 5          # 50 QA + 20 adversarial + 10 injection contracts
python -m uv run python evals/run_compliance.py --live --n-runs 5   # real request interpretation / injected shopper
python -m uv run python evals/run_end_to_end.py --mode demo --n-runs 5  # 10 complete SG-catalog journeys
python -m uv run python evals/run_end_to_end.py --mode live --n-runs 5  # real vision/planner/shoppers/pricing
python -m uv run python evals/run_compliance.py --baseline --n-runs 5   # the same model with no tools or rule table
python -m uv run python evals/run_end_to_end.py --mode no_tools --n-runs 5        # the end-to-end no-tools baseline
python -m uv run python evals/run_end_to_end.py --mode live --no-guard --n-runs 5 # the ablation: guard removed (--mode demo is free)
python -m uv run python evals/run_shortlist.py --prepare --n-runs 5    # real SG top-five results + paired BM25 baseline
python -m uv run python evals/run_shortlist.py --score --n-runs 5      # only after humans label the prepared results
```

Contract results establish deterministic enforcement, not live-model injection resistance. The ABO
Recall@10 suite is a retrieval diagnostic; `run_shortlist.py` measures final SG Precision@5 from human
judgements. Both human style-label sheets are still pending; no automatic substitute labels were added.
The complete human task list, review rubrics, file locations and acceptance criteria are in
[docs/HUMAN_HANDOFF.md](docs/HUMAN_HANDOFF.md). What was dropped from the proposal, and how the baselines
are graded, is in [docs/SCOPE_CHANGES.md](docs/SCOPE_CHANGES.md).

## Layout

```
renopilot/   config, llm (model per role), schemas + vocab + state (shared contracts), trace
  catalog/   store (SQLite), embed (open_clip/SigLIP), search (hybrid + RRF), taxonomy (title -> type)
  nodes/     intake, planner (master agent), style + shopper (ReAct subagent), guard (HDB rules), approval (+ ask_user), respond
  demo.py    the offline demo world: models played by code, over the frozen SG catalog snapshot
  tools/     budget (deterministic money maths)
  pricing/   rules, fx, landed_cost (deterministic), comparables (model-judged), node
  graph.py   the LangGraph wiring;  run_cli.py  terminal demo
data/        scripts/, sg_catalog.csv, boards/, labels/ (raw/, images/, index/ are rebuilt)
evals/       harness.py (week-3 contract + pass^k + CI + resume), cases/, run_*.py
tests/       offline unit + graph tests (fakes.py), live backend smoke tests
docker-compose.yml   self-hosted Langfuse for traces
```

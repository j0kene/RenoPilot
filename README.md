# RenoPilot

From mood board to a compliant, priced shortlist for BTO owners in Singapore (CS621 group project).

This repo currently holds **P0, mood board → shortlist**: the shared LangGraph skeleton plus the Style agent, with
its data and evals. Pricing is in (`renopilot/pricing/`); Compliance plugs into the same graph (see [docs/INTERFACES.md](docs/INTERFACES.md)).

## Setup (Windows, Python 3.12)

```bash
python -m pip install --user uv
python -m uv sync                      # creates .venv from uv.lock
copy .env.example .env                 # optional overrides
python -m uv run pytest                # offline tests, no model calls
```

The model backend is the Claude Code CLI (`claude -p`) for every role while we build and evaluate, so you must
be logged in to Claude Code. Live tests: `set RENOPILOT_LIVE=1 && python -m uv run pytest tests/test_llm_live.py`.
Switching to OpenAI later: set `RENOPILOT_BACKEND=openai` and `OPENAI_API_KEY` in `.env`.

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
python -m uv run --group ui streamlit run app/streamlit_app.py     # demo UI (temporary)
python -m uv run python -m renopilot.run_cli --board data/boards/japandi_01 --budget sofa=1500,coffee_table=400
python -m uv run python -m renopilot.run_cli --board a.jpg b.jpg c.jpg --total 6000 --planner fixed
python -m uv run python -m renopilot.graph      # print the graph as mermaid
```

Prices: `RENOPILOT_FX=snapshot` uses the frozen FX file (reproducible, offline); the default fetches daily
rates from Frankfurter. Overseas rows are currently synthetic sample data (see `data/SOURCES.md`).

Every run writes a JSONL trace to `runs/<run_id>.jsonl`, one row per node or model call.

## Evals

| command | what | grader |
|---|---|---|
| `python -m uv run python evals/run_retrieval.py --n-runs 5` | Recall@10 on boards of known ABO products, vs text-only baseline + ablations | deterministic |
| `python -m uv run python evals/run_style.py --n-runs 5 --baseline` | style attribute F1 vs two human labellers | deterministic |
| `python -m uv run python evals/run_graph.py --n-runs 5` | routes at each boundary (incl. pricing: landed-cost budget removal, stale FX, pricing off), LLM planner vs fixed pipeline | trajectory assertions |
| `python -m uv run python evals/run_pricing.py --n-runs 5` | comparable judging on 28 labelled pairs (17 positive, 11 hard negatives) vs type-only and dims-only baselines | label match |
| `python -m uv run pytest tests/test_landed_cost.py` | 30 hand-worked landed-cost cases, exact to the cent | exact |

Results append to `evals/results/*.jsonl`, so an interrupted run resumes where it stopped.

## Layout

```
renopilot/   config, llm (model per role), schemas + vocab + state (shared contracts), trace
  catalog/   store (SQLite), embed (open_clip/SigLIP), search (hybrid + RRF), taxonomy (title -> type)
  nodes/     intake, planner, style, guard_stub, approval (+ ask_user), respond
  tools/     budget (deterministic money maths)
  pricing/   rules, fx, landed_cost (deterministic), comparables (model-judged), node
  graph.py   the LangGraph wiring;  run_cli.py  terminal demo
data/        scripts/, sg_catalog.csv, boards/, labels/ (raw/, images/, index/ are rebuilt)
evals/       harness.py (week-3 contract + pass^k + CI + resume), cases/, run_*.py
tests/       offline unit + graph tests (fakes.py), live backend smoke tests
```

# Shared interfaces: how the components plug in

The graph, the state and the data contracts are fixed so the workstreams build in parallel
without touching each other's files. If you need to change one of these, agree it first:
`renopilot/schemas.py`, `renopilot/state.py`, `renopilot/vocab.py`.

## The graph

```
START -> intake -> planner -> {style_extract | ask_user} -> planner ...
                   planner -> shortlist -> pricing -> planner
                   planner -> compliance_guard -> approval -> respond -> END
                                                  approval -> planner   (owner rejected something)
```

`compliance_guard -> approval` is a fixed edge: nothing reaches the owner without a verdict,
and no agent can route around it. `shortlist -> pricing` is also fixed: every new shortlist is
priced (landed cost, budget re-check) before the planner sees it. `build_graph(pricing=False)`
drops the pricing node (the P0 graph, used as the ablation).

## Master agent and subagents

The **planner is the master agent**. Each step it makes one tool call (`extract_style`, `shortlist`,
`ask_user`, `finalize`, defined in `nodes/planner.py`), and the call hands the work to a subagent node:
`Command(goto=...)` routes there, the subagent writes its result into the state, and control comes back.
Code checks every call first (`check_action`); a refused call goes back to the master as that tool's
result, once, then the fixed policy decides. The subagents are graph nodes, not tools inside one agent,
so the gates stay graph edges: the master can hand over to Compliance (`finalize`) but cannot skip it.

| subagent | node(s) | how it works | its tools |
|---|---|---|---|
| Style agent | `style_extract`, `shortlist` | one vision call for the profile; then a **shopper** ReAct agent per item type (`nodes/shopper.py`, `PICKER=react`) or one rerank call (`PICKER=rerank`) | `search_catalog`, `product_details`, `landed_price` (+ `Picks` to submit) |
| Pricing agent | `pricing` | deterministic, plus a model judge for comparables | `landed_cost`, `find_candidates`, `basket_total` |
| Compliance | `compliance_guard` | deterministic gate over the HDB rule cards, every suggestion | `hdb_rule_check` |

The shopper's hard rules (item type, budget cap, retailers, excluded ids) are bound into its tools when it is
built, so its tool arguments can only ever narrow a search. Its picks must be ids its searches returned.

On the Claude CLI backend, tool calling is emulated (the calls come back as JSON
under an output schema); with `RENOPILOT_BACKEND=openai` it is native. Either way the agents see standard
`AIMessage.tool_calls`. `RENOPILOT_LIVE=1 uv run pytest tests/test_llm_live.py -k tool` checks the CLI path.

## Data each node sees

`Suggestion` (in `schemas.py`) is the unit everything passes around:

| field | written by | notes |
|---|---|---|
| `product_id`, `title`, `price_sgd`, `url`, `image_path`, `reason`, `score` | Style agent | `product_id=None` for a finish or built-in the board asked for but that isn't a product |
| `item_type`, `work_type` | Style agent | `work_type` comes from `vocab.WORK_TYPE_OF`: `loose_furniture`, `finish`, `built_in` |
| `verdict: Verdict` | **Compliance guard** | `decision` = allow / allow_with_permit / block / escalate, plus `reason`, `rule_id`, `alternative`, `citation` |
| `price_quote: PriceQuote` | **Pricing agent** | landed cost of this item: `item_sgd`, `shipping_sgd`, `local_delivery_sgd` (if bought alone), `gst_sgd`, `gst_basis`, `total_sgd`, `lines`, `assumptions`, `estimated`, `synthetic`, `fx_rate/fx_date/fx_stale` |
| `comparables: list[PricedAlternative]` | **Pricing agent** | up to 2 comparable offers at other retailers, each with its own `PriceQuote` and a `why_comparable` |
| `status` | approval gate | proposed / approved / rejected |

`ProjectProfile` holds `flat_type`, `completion_date` (optional: it narrows the bathroom routes) and the budgets.
Only the owner (through an interrupt's resume value) can change the budget; no model output can.

## Compliance guard (implemented: `renopilot/nodes/guard.py`)

The rules are 22 cards in `data/compliance/hdb_rules.jsonl`, each quoting the HDB page it rests on
(`data/compliance/sources.json`, fetched by `data/scripts/fetch_hdb_sources.py`; a test checks every quote
against the saved page). `compliance.routes(item_type, room, completion_date)` returns the cards that apply.

Most finishes can be done more than one way, and HDB treats the ways differently: overlaying floor tiles needs
no permit, hacking them out needs one, raising the floor is not allowed, and bathroom tiles cannot be replaced in
the block's first 3 years. So each card is a **route**, and `Verdict.options` holds the guard's verdict for every
route, easiest first. The verdict in effect is the easiest legal route; at the approval gate the owner can switch to
another listed route (resume `{"routes": {item_type: rule_id}}`), never to a blocked one. Facts narrow the list:
the board's room (a bathroom card replaces the general card for the same method) and `completion_date`, which
drops the route for the other block age. Without a date both are listed, and picking one states the fact.

| case | verdict |
|---|---|
| loose furniture | `allow`, no options |
| finish / built-in with cards | easiest route in effect, all routes in `options`, each with `rule_id`, `route`, `conditions`, `citation` |
| every route blocked | `block`: the suggestion is rejected (by "guard") with the reason and the card's `alternative` |
| no card covers it | `escalate` with `HDB-NOT-LISTED` (ask HDB); the guard never guesses |

The red-team / injection evals belong to this workstream: follow `evals/run_graph.py` for the pattern
(cases JSONL + a trajectory grader + `run_eval`).

## Pricing agent (implemented: `renopilot/pricing/`)

| file | role | model? |
|---|---|---|
| `rules.py` + `data/pricing/rules.json` | GST, low-value-goods / import-relief rules, delivery fee tables, shipping rates, typical sizes; each with source URL and date checked | no |
| `fx.py` | Frankfurter rates, cached daily; `RENOPILOT_FX=snapshot` uses the frozen `data/pricing/fx_snapshot.json`; stale (> 24 h) is flagged, unknown currency raises | no |
| `landed_cost.py` | `landed_cost(product) -> PriceQuote`, `basket_total(...)` (delivery once per retailer) | no |
| `comparables.py` | `find_candidates` (same type, other retailers) + `judge_comparables` (model, ids constrained) + `dims_compatible` (code rule, overrides the model) | judge only |
| `node.py` | `make_pricing(...)` / `pricing`: prices new suggestions, attaches the 2 cheapest comparables for the top 2 per type, removes items over budget on LANDED cost (`rejected` with `by: "pricing"`), raises an open issue if fewer than 3 remain, warns on stale FX | via comparables |

Overseas listings come from `data/taobao_manual.csv` (loaded by `build_index.py`, not embedded). The current
rows are **synthetic sample fixtures** (`synthetic=true`, titles start `[SAMPLE]`, no URL); they are labelled
"sample data" everywhere they appear and must be replaced with real rows before any real-user claim.

The shortlist's hard budget filter still uses the listed price (a cheap pre-filter); the pricing node re-checks
on landed cost. A pricing removal is not final: if the owner raises the budget, the item can come back
(the shortlist only excludes rejections by the owner or the guard).

## Tracing (JSONL + Langfuse)

Nodes keep logging compact rows with `trace.log(...)` (the JSONL the evals read). Langfuse needs nothing from
a node: with `trace.run_config(...)` as the graph config, every node, model call, agent and LangChain tool call
is traced automatically, nested where it ran: master generation (its tool call) -> subagent node ->
shopper agent -> its tool call -> the code-level tool underneath (e.g. `search_catalog` -> `catalog_search`,
which shows the hard filters the code added to the model's query).
The one thing to add is `@trace.tool(name, inputs=(...))` on a new **tool**, i.e. deterministic code a node
calls (search, a rule lookup, a cost function), so it appears as a tool call under that node. Name only small
arguments in `inputs` (never the catalog or an embedding), and pass `output=` to summarise a large result.
Outside a traced run the decorator is a plain call, so unit tests are unaffected. The compliance rule lookup
already carries `@trace.tool("hdb_rule_check", ...)` on `verdict_for`: keep it when replacing the body.

## Tests and evals

- Offline graph tests use `tests/fakes.py` (`FAKE_NODES`) so they run without the model or the catalog.
  Pass your own fakes through `build_graph(nodes={...})`.
- Every eval goes through `evals/harness.py`: `Case`, `AgentResult`, `run_eval(..., n_runs=5, resume_file=...)`,
  then `Report.render(baseline=...)`. Put failure tags from `vocab.FAILURE_TAGS` in `AgentResult.failure_tags`.

# Shared interfaces: how Compliance and Pricing plug in

The graph, the state and the data contracts are fixed so the workstreams build in parallel
without touching each other's files. If you need to change one of these, agree it first:
`renopilot/schemas.py`, `renopilot/state.py`, `renopilot/vocab.py`.

## The graph

```
START -> intake -> planner -> {style_extract | shortlist | ask_user} -> planner ...
                   planner -> compliance_guard -> approval -> respond -> END
                                                  approval -> planner   (owner rejected something)
```

`compliance_guard -> approval` is a fixed edge: nothing reaches the owner without a verdict,
and no agent can route around it. The planner cannot call the guard on its own, and nothing
after the guard may change a verdict.

## Data each node sees

`Suggestion` (in `schemas.py`) is the unit everything passes around:

| field | written by | notes |
|---|---|---|
| `product_id`, `title`, `price_sgd`, `url`, `image_path`, `reason`, `score` | Style agent | `product_id=None` for a finish or built-in the board asked for but that isn't a product |
| `item_type`, `work_type` | Style agent | `work_type` comes from `vocab.WORK_TYPE_OF`: `loose_furniture`, `finish`, `built_in` |
| `verdict: Verdict` | **Compliance guard** | `decision` = allow / allow_with_permit / block / escalate, plus `reason`, `rule_id`, `alternative`, `citation` |
| `price_quote: dict` | **Pricing agent** (reserved) | landed-cost breakdown; suggested keys: `listed`, `currency`, `fx_rate`, `fx_date`, `shipping`, `local_delivery`, `gst`, `total_sgd` |
| `status` | approval gate | proposed / approved / rejected |

`ProjectProfile` holds `flat_type`, `completion_date` (needed for the 3-year rules) and the budgets.
Only the owner (through an interrupt's resume value) can change the budget; no model output can.

## Compliance guard: what to replace

`renopilot/nodes/guard_stub.py` already has the final node signature. Replace `verdict_for(s, project)`:

```python
def verdict_for(s: Suggestion, project: ProjectProfile | None) -> Verdict:
    # loose furniture -> allow (keep this)
    # finish / built_in -> look up the rule table by s.item_type; date rules in code
    #   (e.g. project.completion_date + 3 years for bathroom finishes); cite the passage
    # not covered by the table -> Verdict(decision="escalate", ...)  never guess
```

A `block` verdict marks the suggestion rejected (by "guard") and records the reason, so the planner can
re-plan with a legal alternative. The red-team / injection evals belong to this workstream: follow
`evals/run_graph.py` for the pattern (cases JSONL + a trajectory grader + `run_eval`).

## Pricing agent: how to add it

1. Write `renopilot/nodes/pricing.py` with `def pricing(state) -> dict` that fills `price_quote` on live
   suggestions, using deterministic, unit-tested tools in `renopilot/tools/` (FX, GST, landed cost).
2. In `graph.py`, add the node between `shortlist` and the planner, or between `planner` and
   `compliance_guard`, whichever is agreed. It is one `add_node` and one edge change.
3. `tools/budget.py` currently filters on the listed SGD price (`price_sgd`). Once landed cost exists,
   the budget filter should switch to `price_quote["total_sgd"]`.

## Tests and evals

- Offline graph tests use `tests/fakes.py` (`FAKE_NODES`) so they run without the model or the catalog.
  Pass your own fakes through `build_graph(nodes={...})`.
- Every eval goes through `evals/harness.py`: `Case`, `AgentResult`, `run_eval(..., n_runs=5, resume_file=...)`,
  then `Report.render(baseline=...)`. Put failure tags from `vocab.FAILURE_TAGS` in `AgentResult.failure_tags`.

"""Complete SG-catalog journeys. --mode live uses the real vision/planner/shoppers/pricing.

Demo runs the production graph and tools with the labelled scripted Japandi example; its
results establish integration only. Fixed dates keep bathroom-rule results reproducible.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from evals.harness import AgentResult, load_cases, run_eval  # noqa: E402
from evals.run_graph import resolve_step  # noqa: E402
from renopilot import config, trace  # noqa: E402
from renopilot.basket import selection  # noqa: E402
from renopilot.demo import DEMO_BOARD, demo_nodes  # noqa: E402
from renopilot.graph import build_graph  # noqa: E402
from renopilot.nodes.respond import shopping_list_csv  # noqa: E402
from renopilot.schemas import ProjectProfile  # noqa: E402


def make_agent(mode):
    def agent(inp):
        t0 = time.perf_counter()
        graph = build_graph(nodes=demo_nodes() if mode == "demo" else None)
        board = [str(p) for p in sorted((config.BOARDS / inp.get("board", DEMO_BOARD.name)).glob("*.jpg"))]
        gates, resumes = [], []
        with trace.tracing() as t:
            cfg = trace.run_config("renopilot-eval-e2e", recursion_limit=100, tags=["eval:e2e", mode])
            out = graph.invoke({"board_images": board, "project": ProjectProfile.model_validate(inp["project"]),
                                "messages": [HumanMessage(inp.get("brief", ""))]}, cfg)
            script = list(inp["script"])
            for _ in range(12):
                if "__interrupt__" not in out:
                    break
                gate = out["__interrupt__"][0].value
                gates.append(gate)
                if not script or gate.get("error"):
                    break
                if gate["type"] == "question" and not script[0].startswith("answer"):
                    reply = {"answer": "Keep the budget and accept fewer options."}
                else:
                    reply = resolve_step(script.pop(0), gate)
                resumes.append(reply)
                out = graph.invoke(Command(resume=reply), cfg)
            state = graph.get_state(cfg).values
            totals = t.totals()
        return AgentResult(output={"state": state, "gates": gates, "parked": "__interrupt__" in out,
            "export": shopping_list_csv(state), "mode": mode}, latency_s=time.perf_counter() - t0,
            run_id=t.run_id, input_tokens=totals["input_tokens"], output_tokens=totals["output_tokens"],
            cost_usd=totals["cost_usd"])
    return agent


def grade(case, result):
    state, e = result.output["state"], case.expected
    failures = []
    if bool(state.get("approved")) != e.get("approved", True):
        failures.append("premature_stop")
    if state.get("approved"):
        try:
            picks, _ = selection(state["suggestions"], state["selected"], state["project"])
            if not picks:
                failures.append("retrieval_miss")
        except (ValueError, KeyError):
            failures.append("over_budget")
        if any(s.status == "approved" and (not s.verdict or s.verdict.decision not in
               ("allow", "allow_with_permit")) for s in state["suggestions"]):
            failures.append("rule_violation")
    if e.get("blocked_rule") and not any(s.verdict and s.verdict.rule_id == e["blocked_rule"] and
            s.status == "rejected" for s in state.get("suggestions", [])):
        failures.append("rule_violation")
    if e.get("permit_rule") and not any(s.verdict and s.verdict.rule_id == e["permit_rule"] and
            s.verdict.decision == "allow_with_permit" for s in state.get("suggestions", [])):
        failures.append("rule_violation")
    if e.get("reshortlist") and not any(d.get("refinement") for d in state.get("trace", [])):
        failures.append("feedback_dropped")
    if e.get("parked") and not result.output["parked"]:
        failures.append("under_escalation")
    result.failure_tags = sorted(set(failures))
    return not failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "live"], default="demo")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--cases", type=int, default=0)
    args = ap.parse_args()
    cases = load_cases(config.ROOT / "evals" / "cases" / "end_to_end.jsonl")
    if args.cases:
        cases = cases[:args.cases]
    rep = run_eval(make_agent(args.mode), cases, grade, n_runs=args.n_runs, name=f"e2e ({args.mode})",
        revision=f"{args.mode}:{config.BACKEND}:{config.MODELS[config.BACKEND]}:{config.PICKER}",
        resume_file=config.ROOT / "evals" / "results" / f"end_to_end_{args.mode}.jsonl")
    print(rep.render())
    if rep.mean_pass < 1:
        sys.exit(1)


if __name__ == "__main__":
    main()

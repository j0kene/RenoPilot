"""Graph trajectory eval: does the agent take the right ROUTE at each boundary?

    uv run python evals/run_graph.py [--planner llm|fixed|both] [--n-runs 5] [--real-style]

By default the style nodes are the offline fakes (tests/fakes.py), so the thing under test
is the planner and the gates: is the board checked before any model call, does a tight
budget stop and ask, does a rejection re-search only that item type, does the owner's
feedback reach the search (refinement), is nothing approved without the owner?
`--real-style` swaps in the real Style agent (slow; for end-to-end runs on real boards).

Graders are trajectory assertions read from the state's decision log, not from the reply text.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langgraph.types import Command  # noqa: E402
from PIL import Image  # noqa: E402

from evals.harness import AgentResult, load_cases, run_eval  # noqa: E402
from renopilot import config, trace  # noqa: E402
from renopilot.graph import build_graph  # noqa: E402
from renopilot.nodes.planner import PROMPT_VERSION  # noqa: E402
from renopilot.schemas import ProjectProfile  # noqa: E402
from renopilot.tools.budget import caps_for  # noqa: E402
from tests.fakes import FAKE_NODES, fake_pricing_node  # noqa: E402

TMP = config.CACHE / "eval_boards"


def fake_board(n: int) -> list[str]:
    TMP.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = TMP / f"b{i}.jpg"
        if not p.exists():
            Image.new("RGB", (64, 64), (200, 180, 150)).save(p)
        paths.append(str(p))
    return paths


def resolve_step(step: str, gate: dict) -> dict:
    """Script step -> resume value. 'reject:sofa:1:reason|coffee_table:1:reason' picks the
    n-th live suggestion of that item type at the gate, whatever its id is this run."""
    if step == "approve":
        return {"approve": "all"}
    if step.startswith("answer:add_images:"):
        return {"answer": "added more images", "board_images": fake_board(int(step.rsplit(":", 1)[1]))}
    if step.startswith("answer:raise:"):
        k, v = step.split(":", 2)[2].split("=")
        return {"answer": f"ok, up to {v} for the {k}", "budget_by_category": {k: float(v)}}
    if step.startswith("answer:"):
        return {"answer": step.split(":", 1)[1]}
    if step.startswith("reject:"):
        rejects = []
        for part in step.split(":", 1)[1].split("|"):
            t, n, reason = part.split(":", 2)
            cards = [c for c in gate.get("suggestions", []) if c["item_type"] == t and c["product_id"]]
            if len(cards) >= int(n):
                rejects.append({"product_id": cards[int(n) - 1]["product_id"], "reason": reason})
        return {"reject": rejects}
    raise ValueError(step)


def make_agent(planner: str, real_style: bool):
    def agent(inp: dict) -> AgentResult:
        t0 = time.perf_counter()
        mode = inp.get("pricing", "fake")            # fake (no-op) | landed | stale_fx | off
        nodes = None if real_style else dict(FAKE_NODES)
        if nodes is not None and mode in ("landed", "stale_fx"):
            nodes["pricing"] = fake_pricing_node(stale=mode == "stale_fx")
        g = build_graph(planner=planner, nodes=nodes, pricing=mode != "off")
        project = ProjectProfile.model_validate({"budget_by_category": inp.get("budget", {}),
            "total_budget": inp.get("total"), "as_of": inp.get("as_of", "2026-10-28")})
        board = (inp.get("board") or [str(p) for p in sorted((config.BOARDS / "japandi_01").glob("*.jpg"))][:inp["n_images"]]
                 ) if real_style else fake_board(inp["n_images"])
        gates, shown, resumes = [], [], []
        with trace.tracing() as t:
            cfg = trace.run_config("renopilot-eval-graph", recursion_limit=80, run_id=t.run_id,
                                   tags=["eval:graph", f"planner:{planner}", f"pricing:{mode}"])
            out = g.invoke({"board_images": board, "project": project, "messages": []}, cfg)
            script = list(inp["script"])
            while "__interrupt__" in out:
                gate = out["__interrupt__"][0].value
                gates.append(gate)
                if gate["type"] == "approval":
                    shown.append({c["product_id"] for c in gate["suggestions"] if c["product_id"]})
                if not script:
                    break                                   # owner walks away: the run must stay parked
                if gate["type"] == "question" and not script[0].startswith("answer"):
                    resumes.append({"answer": "No strong preference, use your judgement."})   # unscripted question
                else:
                    resumes.append(resolve_step(script.pop(0), gate))
                out = g.invoke(Command(resume=resumes[-1]), cfg)
            state = g.get_state(cfg).values
        totals = t.totals()
        return AgentResult(
            output={"state": state, "gates": gates, "shown": shown, "resumes": resumes,
                    "parked": "__interrupt__" in out},
            tool_calls=[d["action"] for d in state.get("trace", []) if d["node"] == "planner"],
            steps=len(state.get("trace", [])), input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"], cost_usd=totals["cost_usd"],
            latency_s=time.perf_counter() - t0, run_id=t.run_id)
    return agent


def grade(case, result) -> bool:
    e, o = case.expected, result.output
    state, gates = o["state"], o["gates"]
    planner_log = [d for d in state.get("trace", []) if d["node"] == "planner"]
    fails = []
    if "first_gate" in e and (not gates or gates[0]["type"] != e["first_gate"]):
        fails.append("under_escalation" if e["first_gate"] == "question" else "over_escalation")
    if "question_mentions" in e and (not gates or e["question_mentions"] not in gates[0].get("question", "")):
        fails.append("format_error")
    if "planner_actions" in e and result.tool_calls != e["planner_actions"]:
        fails.append("wrong_tool")
    if e.get("no_extract_under_min_images"):            # the board rule holds even after the owner answers
        if len(state.get("board_images") or []) < config.MIN_BOARD_IMAGES and                 any(d["node"] == "style_extract" for d in state.get("trace", [])):
            fails.append("rule_violation")
    if "reshortlist_types" in e:
        # a RE-search is a search of a type that was already searched before; first-time searches
        # of a type (in any order) are not re-searches, and grouping is an efficiency question
        seen, redo = set(), []
        for d in planner_log:
            if d["action"] == "shortlist":
                redo += [t for t in d["item_types"] if t in seen]
                seen |= set(d["item_types"])
        if sorted(redo) != sorted(t for x in e["reshortlist_types"] for t in x):
            fails.append("wrong_tool")
    if "refinement_for" in e:
        if not any(d["action"] == "shortlist" and e["refinement_for"] in d["item_types"] and d.get("refinement")
                   for d in planner_log):
            fails.append("feedback_dropped")
    if "max_questions" in e and sum(g["type"] == "question" for g in gates) > e["max_questions"]:
        fails.append("over_escalation")
    if "approved" in e and bool(state.get("approved")) != e["approved"]:
        fails.append("under_escalation" if state.get("approved") else "premature_stop")
    if e.get("stops_at_gate") and not o["parked"]:
        fails.append("under_escalation")
    if e.get("all_verdicts") and not all(s.verdict for s in state.get("suggestions", []) if s.status != "rejected"):
        fails.append("rule_violation")
    if "finish_verdict" in e:
        fin = [s for s in state.get("suggestions", []) if s.work_type != "loose_furniture"]
        if not fin or any(s.verdict is None or s.verdict.decision != e["finish_verdict"] for s in fin):
            fails.append("rule_violation")
    if e.get("never_reshown_rejected"):                  # rejected at gate i -> absent from every later gate
        approval_resumes = [r for g, r in zip(gates, o["resumes"]) if g["type"] == "approval"]
        gone: set[str] = set()
        for i, ids in enumerate(o["shown"]):
            if gone & ids:
                fails.append("wrong_tool")
                break
            if i < len(approval_resumes):
                gone |= {r["product_id"] for r in approval_resumes[i].get("reject", [])}
    if "escalation_mentions" in e:
        if not any(e["escalation_mentions"] in x for x in state.get("escalations") or []):
            fails.append("under_escalation")
    if e.get("removed_by_pricing"):
        if not any(r.get("by") == "pricing" for r in state.get("rejected") or []):
            fails.append("over_budget")
    if e.get("no_pricing_node") and any(d["node"] == "pricing" for d in state.get("trace", [])):
        fails.append("wrong_tool")
    if e.get("within_caps"):
        caps = caps_for(state.get("project"), [s.item_type for s in state.get("suggestions", []) if s.product_id])
        if any(caps.get(s.item_type) is not None and
               (s.price_quote.total_sgd if s.price_quote else s.price_sgd or 0) > caps[s.item_type]
               for s in state.get("suggestions", []) if s.status != "rejected" and s.product_id):
            fails.append("over_budget")
    result.failure_tags = sorted(set(fails))
    return not fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["llm", "fixed", "both"], default="both")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--real-style", action="store_true")
    args = ap.parse_args()
    cases = load_cases(config.ROOT / "evals" / "cases" / "graph_trajectory.jsonl")
    planner_model = config.MODELS[config.BACKEND]["planner"]
    reports = {}
    for planner in (["fixed", "llm"] if args.planner == "both" else [args.planner]):
        n = 1 if planner == "fixed" and not args.real_style else args.n_runs     # fixed + fakes is deterministic
        tag = f"{planner_model}_{PROMPT_VERSION}" if planner == "llm" else "na"
        out = config.ROOT / "evals" / "results" / f"graph_{planner}_{tag}{'_real' if args.real_style else ''}.jsonl"
        reports[planner] = run_eval(make_agent(planner, args.real_style), cases, grade, n_runs=n,
                                    name=f"planner={planner}", resume_file=out,
                                    revision=f"{planner}:{args.real_style}:{config.BACKEND}:{config.MODELS[config.BACKEND]}")
    print()
    for name, rep in reports.items():
        print(rep.render(baseline=reports.get("fixed") if name == "llm" else None))
        print()


if __name__ == "__main__":
    main()

"""Complete SG-catalog journeys. --mode live uses the real vision/planner/shoppers/pricing.

Demo runs the production graph and tools with the labelled scripted Japandi example; its
results establish integration only. Fixed dates keep bathroom-rule results reproducible.

The proposal's baseline and ablation:
    --mode no_tools   the same vision model with no tools, catalog or rule table: one chat
    --no-guard        our graph (demo or live) with the compliance guard removed
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from evals.harness import AgentResult, load_cases, run_eval  # noqa: E402
from evals.run_graph import resolve_step  # noqa: E402
from renopilot import config, trace  # noqa: E402
from renopilot.basket import selection  # noqa: E402
from renopilot.demo import DEMO_BOARD, demo_nodes  # noqa: E402
from renopilot.graph import build_graph  # noqa: E402
from renopilot.llm import get_model, image_block, usage_of  # noqa: E402
from renopilot.nodes.guard import verdict_for  # noqa: E402
from renopilot.nodes.respond import shopping_list_csv  # noqa: E402
from renopilot.schemas import ItemType, ProjectProfile, Verdict  # noqa: E402
from renopilot.tools.budget import caps_for, within  # noqa: E402

CLEARED = ("allow", "allow_with_permit")


def no_guard(state):
    """The ablation: the guard's place in the graph stays, and it clears whatever it is given."""
    v = Verdict(decision="allow", reason="Ablation: the compliance guard is removed.")
    return {"suggestions": [s if s.status == "rejected" else s.model_copy(update={"verdict": v})
                            for s in state.get("suggestions") or []],
            "trace": [{"node": "compliance_guard", "action": "ablated", "why": "no guard"}]}


def make_agent(mode, guard=True):
    def agent(inp):
        t0 = time.perf_counter()
        graph = build_graph(nodes=(demo_nodes() if mode == "demo" else {}) | ({} if guard else {"compliance_guard": no_guard}))
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


def violations(state):
    """Approved work the real guard would not clear: what reaches the owner without it."""
    return sum(not s.product_id and s.status == "approved" and verdict_for(s, state.get("project")).decision not in CLEARED
               for s in state.get("suggestions") or [])


def illustration_as_real(state, export):
    """Made-up overseas rows may appear on an approval card, labelled. Never bought, exported or in the reply."""
    reply = state["messages"][-1].content if state.get("messages") else ""
    return any(s.status == "approved" and s.price_quote and s.price_quote.synthetic for s in state.get("suggestions") or []
               ) or "[SAMPLE]" in export or (state.get("approved") and "[SAMPLE]" in str(reply))


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
    if e.get("escalated") and not any(not s.product_id and s.verdict and s.verdict.decision == "escalate"
                                      for s in state.get("suggestions", [])):
        failures.append("under_escalation")
    if e.get("reshortlist") and not any(d.get("refinement") for d in state.get("trace", [])):
        failures.append("feedback_dropped")
    if e.get("parked") and not result.output["parked"]:
        failures.append("under_escalation")
    result.metrics["violations"] = float(violations(state))
    if result.metrics["violations"]:
        failures.append("rule_violation")
    if illustration_as_real(state, result.output["export"]):
        failures.append("unsupported_claim")
    result.failure_tags = sorted(set(failures))
    return not failures


# --------------------------------------------------------------------------- the no-tools baseline
NO_TOOLS_SYSTEM = """You help a first-time BTO flat owner in Singapore turn a mood board into a renovation plan.
You have no tools, no product catalog and no rule documents: answer from what you already know.
products: real items to buy in Singapore for the board's look, one per item type, with the retailer, the price in
SGD and the link. Stay within the project's budgets. If a budget leaves no room, leave products empty and ask.
works: every finish, built-in or renovation the brief or board asks for, with your decision under HDB rules:
allow, allow_with_permit, block if it is not permitted, or escalate if you cannot tell from the facts given.
questions: anything you need the owner to decide."""


class PlannedProduct(BaseModel):
    item_type: ItemType
    title: str
    retailer: str = ""
    price_sgd: float
    url: str = ""


class PlannedWork(BaseModel):
    request: str
    decision: Literal["allow", "allow_with_permit", "block", "escalate"]
    reason: str = ""


class Plan(BaseModel):
    products: list[PlannedProduct] = Field(default_factory=list)
    works: list[PlannedWork] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)


def no_tools_agent(inp):
    """The same conversation as the graph's script, as one chat. Nothing stops an answer from being
    wrong, so nothing is enforced here; the grader holds it to what a chat reply can be held to."""
    t0 = time.perf_counter()
    model = get_model("vision").with_structured_output(Plan, include_raw=True)
    project = ProjectProfile.model_validate(inp["project"])
    board = sorted((config.BOARDS / inp.get("board", DEMO_BOARD.name)).glob("*.jpg"))

    def ask(messages):
        t1 = time.perf_counter()
        out = model.invoke(messages)
        trace.log("no_tools", "model_call", latency_s=round(time.perf_counter() - t1, 2), **usage_of(out["raw"]))
        if out["parsed"] is None:
            raise ValueError(f"format_error: {out['parsing_error']}")
        return out["parsed"]

    with trace.tracing() as t:
        text = f"Project: {project.model_dump_json(exclude_none=True)}\nBrief: {inp.get('brief') or 'none'}\nMood board:"
        messages = [SystemMessage(NO_TOOLS_SYSTEM),
                    HumanMessage([{"type": "text", "text": text}] + [image_block(p) for p in board])]
        plans, approved = [ask(messages)], False
        for step in inp["script"]:
            if step == "approve":
                approved = True
                break
            if step.startswith("answer:raise:"):
                item, value = step.split(":", 2)[2].split("=")
                project = project.model_copy(update={"budget_by_category": project.budget_by_category | {item: float(value)}})
                note = f"I can go up to S${value} for the {item.replace('_', ' ')}. Please revise the plan."
            else:
                item, _, reason = step.split(":", 1)[1].split(":", 2)
                note = f"I reject your {item.replace('_', ' ')}: {reason}. Please revise the plan."
            messages += [AIMessage(plans[-1].model_dump_json()), HumanMessage(note)]
            plans.append(ask(messages))
        totals = t.totals()
    return AgentResult(output={"plans": [p.model_dump() for p in plans], "project": project.model_dump(mode="json"),
                               "approved": approved, "mode": "no_tools"},
                       latency_s=time.perf_counter() - t0, run_id=t.run_id, input_tokens=totals["input_tokens"],
                       output_tokens=totals["output_tokens"], cost_usd=totals["cost_usd"])


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


@lru_cache(maxsize=1)
def _catalog_keys():
    with (config.DATA / "sg_catalog.csv").open(encoding="utf-8") as f:
        return {_norm(v) for r in csv.DictReader(f) for v in (r["title"], r["url"]) if v}


def grade_no_tools(case, result):
    """The rubric with the graph's rule ids replaced by decisions, which this baseline can give.
    Generous on purpose: an invented product still passes, and `catalog_match` reports how many
    are real. The graph's products come from the catalog by construction."""
    o, e = result.output, case.expected
    plan, project = o["plans"][-1], ProjectProfile.model_validate(o["project"])
    products, decisions, failures = plan["products"], {w["decision"] for w in plan["works"]}, []
    if e.get("approved", True) and not products:
        failures.append("retrieval_miss")
    caps = caps_for(project, [p["item_type"] for p in products])
    total = sum(p["price_sgd"] for p in products)
    if any(not within(p["price_sgd"], caps.get(p["item_type"])) for p in products) or (
            project.total_budget is not None and total > project.total_budget):
        failures.append("over_budget")
    needed = "block" if e.get("blocked_rule") else "allow_with_permit" if e.get("permit_rule") else \
        "escalate" if e.get("escalated") else None
    if needed and needed not in decisions:
        failures.append("under_escalation" if needed == "escalate" else "rule_violation")
    # work the rule table does not clear, stated as allowed or permitted, is what would reach the owner
    result.metrics["violations"] = float(needed in ("block", "escalate") and needed not in decisions and bool(decisions & set(CLEARED)))
    if e.get("reshortlist"):
        first, last = ({p["title"] for p in x["products"] if p["item_type"] == "sofa"} for x in (o["plans"][0], plan))
        if first == last:
            failures.append("feedback_dropped")
    if products:
        result.metrics["catalog_match"] = sum(_norm(p["title"]) in _catalog_keys() or _norm(p["url"]) in _catalog_keys()
                                              for p in products) / len(products)
    result.failure_tags = sorted(set(failures))
    return not failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "live", "no_tools"], default="demo")
    ap.add_argument("--no-guard", action="store_true", help="ablation: remove the compliance guard")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--cases", type=int, default=0)
    args = ap.parse_args()
    if args.no_guard and args.mode == "no_tools":
        ap.error("--no-guard needs the graph: use it with --mode demo or live")
    cases = load_cases(config.ROOT / "evals" / "cases" / "end_to_end.jsonl")
    if args.cases:
        cases = cases[:args.cases]
    label = args.mode + ("_no_guard" if args.no_guard else "")
    agent, grader = (no_tools_agent, grade_no_tools) if args.mode == "no_tools" else (make_agent(args.mode, not args.no_guard), grade)
    rep = run_eval(agent, cases, grader, n_runs=args.n_runs, name=f"e2e ({label})",
        revision=f"{label}:{config.BACKEND}:{config.MODELS[config.BACKEND]}:{config.PICKER}",
        resume_file=config.ROOT / "evals" / "results" / f"end_to_end_{label}.jsonl")
    print(rep.render())
    if rep.mean_pass < 1 and label in ("demo", "live"):     # a baseline is expected to fail cases
        sys.exit(1)


if __name__ == "__main__":
    main()

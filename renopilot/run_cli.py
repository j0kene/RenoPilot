"""Terminal demo: run the graph on a board, answer its questions, approve or reject.

    uv run python -m renopilot.run_cli --board data/boards/japandi_01 --budget sofa=1500,coffee_table=400
    uv run python -m renopilot.run_cli --board img1.jpg img2.jpg img3.jpg --total 6000 --planner fixed
    uv run python -m renopilot.run_cli --board data/boards/japandi_01 --total 6000 --auto-approve

At the approval gate type:  a            approve all
                            r 3 too big  reject item #3 with a reason (several: r 1,3 reason)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

from renopilot import config, trace
from renopilot.graph import build_graph
from renopilot.schemas import ProjectProfile

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def board_paths(args: list[str]) -> list[str]:
    out = []
    for a in args:
        p = Path(a) if Path(a).is_absolute() else config.ROOT / a
        out += sorted(str(f) for f in p.iterdir() if f.suffix.lower() in IMG_EXT) if p.is_dir() else [str(p)]
    return out


def parse_budget(s: str | None) -> dict[str, float]:
    if not s:
        return {}
    return {k.strip(): float(v) for k, v in (kv.split("=") for kv in s.split(","))}


def show_approval(gate: dict) -> list[dict]:
    cards = gate["suggestions"]
    print("\n--- SHORTLIST (approval needed) ---")
    last = None
    for i, c in enumerate(cards, 1):
        if c["item_type"] != last:
            print(f"\n{c['item_type'].replace('_', ' ').upper()}")
            last = c["item_type"]
        flag = "" if c["verdict"] in (None, "allow") else f"  [{c['verdict']}] {c['verdict_reason']}"
        print(f"  {i:>2}. {c['title'][:70]:<70} {c['price']:>10}{flag}")
        if c["product_id"]:
            print(f"      {c['reason']}")
    for e in gate.get("escalations") or []:
        print(f"\n  note: {e}")
    return cards


def ask_approval(cards: list[dict], auto: bool) -> dict:
    if auto:
        return {"approve": "all"}
    raw = input("\n[a]pprove all, or r <numbers> <reason>: ").strip()
    if not raw or raw.lower().startswith("a"):
        return {"approve": "all"}
    _, nums, *reason = raw.split(" ", 2) + [""]
    picked = [cards[int(n) - 1] for n in nums.split(",") if n.strip().isdigit() and 0 < int(n) <= len(cards)]
    return {"reject": [{"product_id": c["product_id"], "reason": " ".join(reason).strip()}
                       for c in picked if c["product_id"]]}


def ask_question(gate: dict, auto: bool) -> dict:
    print(f"\n--- QUESTION ---\n{gate['question']}")
    if auto:
        return {"answer": "keep my budget"}
    answer = input("answer> ").strip()
    budget = input("new budgets (e.g. sofa=2000) or blank> ").strip()
    return {"answer": answer} | ({"budget_by_category": parse_budget(budget)} if budget else {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", nargs="+", required=True, help="image files or a folder")
    ap.add_argument("--budget", help="per-item caps, e.g. sofa=1500,coffee_table=400")
    ap.add_argument("--total", type=float, help="total furniture budget (SGD)")
    ap.add_argument("--planner", choices=["llm", "fixed"], default="llm")
    ap.add_argument("--auto-approve", action="store_true")
    args = ap.parse_args()

    graph = build_graph(planner=args.planner)
    cfg = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 60}
    project = ProjectProfile(budget_by_category=parse_budget(args.budget), total_budget=args.total)
    with trace.tracing() as t:
        out = graph.invoke({"board_images": board_paths(args.board), "project": project, "messages": []}, cfg)
        while "__interrupt__" in out:
            gate = out["__interrupt__"][0].value
            if gate["type"] == "approval":
                resume = ask_approval(show_approval(gate), args.auto_approve)
            else:
                resume = ask_question(gate, args.auto_approve)
            out = graph.invoke(Command(resume=resume), cfg)
        print("\n=== REPLY ===\n" + out["messages"][-1].content)
        tot = t.totals()
        print(f"\n[{tot['model_calls']} model calls, {tot['input_tokens']:,} in / {tot['output_tokens']:,} out tokens, "
              f"~${tot['cost_usd']:.3f} API-equivalent]  trace: {t.path}")
        print("[decisions] " + " -> ".join(f"{d['node']}:{d['action']}" for d in out.get("trace", [])))


if __name__ == "__main__":
    main()

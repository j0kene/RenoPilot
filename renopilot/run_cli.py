"""Terminal demo: run the graph on a board, answer its questions, approve or reject.

    uv run python -m renopilot.run_cli --board data/boards/japandi_01 --budget sofa=1500,coffee_table=400
    uv run python -m renopilot.run_cli --board img1.jpg img2.jpg img3.jpg --total 6000 --planner fixed
    uv run python -m renopilot.run_cli --board data/boards/japandi_01 --total 6000 --auto-approve
    uv run python -m renopilot.run_cli --board data/boards/japandi_01 --total 6000 --completed 2024-06-01         --flat-type 4-room --brief "warm Japandi living room; we also want to redo the bathroom floor"

At the approval gate type:  a            approve all
                            r 3 too big  reject item #3 with a reason (several: r 1,3 reason)
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from langchain_core.messages import HumanMessage
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
        price = f"{c['landed']} landed{' (est.)' if c.get('estimated') else ''}" if c.get("landed") else c["price"]
        print(f"  {i:>2}. {c['title'][:62]:<62} {price:>20}{flag}")
        if c["product_id"]:
            print(f"      {c['reason']}")
            for a in c.get("comparables") or []:
                tag = " [SAMPLE DATA]" if a["sample"] else ""
                print(f"      {'cheaper' if a['cheaper'] else 'similar'} at {a['source']}: {a['title'][:50]} "
                      f"{a['landed']} landed{tag}")
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
    ap.add_argument("--completed", type=date.fromisoformat, help="block completion date, YYYY-MM-DD (3-year bathroom rule)")
    ap.add_argument("--flat-type", choices=["2-room", "3-room", "4-room", "5-room", "executive", "other"])
    ap.add_argument("--brief", default="", help="what you'd like, in your own words (fills fields left empty)")
    args = ap.parse_args()

    graph = build_graph(planner=args.planner)
    project = ProjectProfile(budget_by_category=parse_budget(args.budget), total_budget=args.total,
                             completion_date=args.completed, flat_type=args.flat_type)
    messages = [HumanMessage(args.brief.strip())] if args.brief.strip() else []
    with trace.tracing() as t:
        cfg = trace.run_config("renopilot-cli", tags=["cli", f"planner:{args.planner}"], run_id=t.run_id)
        out = graph.invoke({"board_images": board_paths(args.board), "project": project, "messages": messages}, cfg)
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
        if url := trace.langfuse_url(cfg):
            print(f"[langfuse] {url}")


if __name__ == "__main__":
    main()

"""Langfuse demo that needs no model and no catalog (the offline demo world, renopilot/demo.py):
the real graph, shopper agents, pricing and compliance code, with the models played by code
over a made-up sample catalog. The Streamlit app's demo mode runs the same thing, clickable.

    docker compose up -d langfuse-web
    uv run python evals/demo_trace.py                  # prints the trace link

The story: the master reads the board, shortlists, finalizes; the owner rejects the first sofa
("too bulky, prefer darker wood"); the master re-shortlists sofas with that as a refinement,
which reaches the shopper's search; the owner approves. All of it is one trace.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langgraph.types import Command  # noqa: E402

from renopilot import config, trace  # noqa: E402
from renopilot.demo import demo_nodes  # noqa: E402
from renopilot.graph import build_graph  # noqa: E402
from renopilot.schemas import ProjectProfile  # noqa: E402

for k, v in {"LANGFUSE_PUBLIC_KEY": "pk-lf-renopilot-local", "LANGFUSE_SECRET_KEY": "sk-lf-renopilot-local",
             "LANGFUSE_BASE_URL": "http://localhost:3000"}.items():
    os.environ.setdefault(k, v)                       # the keys docker-compose.yml creates by default


def main() -> None:
    g = build_graph(nodes=demo_nodes())
    cfg = trace.run_config("renopilot-demo", tags=["demo"])
    board = [str(p) for p in sorted((config.BOARDS / "japandi_01").glob("*.jpg"))]
    out = g.invoke({"board_images": board, "messages": [],
                    "project": ProjectProfile(total_budget=6000, budget_by_category={"sofa": 1500, "coffee_table": 400})},
                   cfg)
    sofa = next(c for c in out["__interrupt__"][0].value["suggestions"] if c["item_type"] == "sofa")
    print(f"owner rejects {sofa['title']}: too bulky, prefer darker wood")
    out = g.invoke(Command(resume={"reject": [{"product_id": sofa["product_id"],
                                               "reason": "too bulky, prefer darker wood"}]}), cfg)
    print("owner approves the new shortlist")
    out = g.invoke(Command(resume={"approve": "all"}), cfg)
    print("\n" + out["messages"][-1].content)
    print(f"\nLangfuse trace: {trace.langfuse_url(cfg) or 'Langfuse not reachable: is the container up?'}")


if __name__ == "__main__":
    main()

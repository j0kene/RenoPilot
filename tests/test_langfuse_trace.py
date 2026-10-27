"""Langfuse wiring, offline: spans go to an in-memory exporter instead of the server.
Checks that tool calls nest under the node (or agent tool) that made them, that the master's
tool calls and the shopper subagents land in the same tree, and that resuming an interrupt
continues the same trace."""
import json

import pytest
from langfuse import Langfuse
from langgraph.types import Command
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from PIL import Image

from renopilot import trace
from renopilot.graph import build_graph
from renopilot.nodes.planner import make_planner
from renopilot.nodes.style import make_shortlist
from renopilot.pricing.landed_cost import landed_cost
from renopilot.schemas import ProjectProfile
from tests.fakes import (FAKE_NODES, FakeShopperModel, ScriptedToolModel, fake_index, fake_priced_catalog,
                         fake_pricing_node)

TYPE, INPUT = "langfuse.observation.type", "langfuse.observation.input"
EXPORTER = InMemorySpanExporter()     # Langfuse keeps one client per key per process: share its exporter
BUDGET = ProjectProfile(budget_by_category={"sofa": 2000, "coffee_table": 400})


@pytest.fixture
def spans(monkeypatch):
    """Langfuse on, exporting to memory. Yields a function that flushes and returns the spans."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")   # never reached: spans stay in memory
    client = Langfuse(span_exporter=EXPORTER, mask=trace.omit_images)   # the first client for this key wins
    trace._langfuse.cache_clear()
    EXPORTER.clear()

    def finished():
        client.flush()
        return EXPORTER.get_finished_spans()

    yield finished
    trace._langfuse.cache_clear()


def _board(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = tmp_path / f"b{i}.jpg"
        Image.new("RGB", (32, 32), (200, 180, 150)).save(p)
        paths.append(str(p))
    return paths


def test_tools_are_plain_calls_when_langfuse_is_off(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    q = landed_cost(fake_priced_catalog()["sofa0"])
    assert q.total_sgd == 635.0                           # S$600 + S$35 home delivery
    assert "callbacks" not in trace.run_config("t")


def test_images_are_masked_but_keep_their_place():
    msg = {"role": "user", "content": [{"type": "text", "text": "board"},
                                       {"type": "image", "base64": "A" * 4096, "mime_type": "image/jpeg"}]}
    out = trace.omit_images(data=[msg])
    assert out[0]["content"][0] == {"type": "text", "text": "board"}
    assert out[0]["content"][1] == {"type": "image", "base64": "<3 KB omitted>", "mime_type": "image/jpeg"}


def _run(g, tmp_path):
    cfg = trace.run_config("test", tags=["test"])
    out = g.invoke({"board_images": _board(tmp_path), "messages": [], "project": BUDGET}, cfg)
    assert out["__interrupt__"][0].value["type"] == "approval"
    g.invoke(Command(resume={"approve": "all"}), cfg)                # same cfg: same trace


def test_graph_run_is_one_trace_with_tools_under_their_nodes(tmp_path, spans):
    _run(build_graph(planner="fixed", nodes=FAKE_NODES | {"pricing": fake_pricing_node()}), tmp_path)
    got = spans()
    assert len({s.context.trace_id for s in got}) == 1               # the resume joined the first trace
    by_id = {s.context.span_id: s for s in got}
    assert [s.name for s in got if s.parent.span_id not in by_id] == ["test", "test"]   # parked, resumed
    parent = lambda s: by_id[s.parent.span_id].name                  # noqa: E731
    tools = {(s.name, parent(s)) for s in got if s.attributes.get(TYPE) == "tool"}
    assert tools == {("landed_cost", "pricing"), ("hdb_rule_check", "compliance_guard"), ("basket_total", "respond")}
    assert {"intake", "planner", "approval", "respond"} <= {s.name for s in got}


def test_master_and_subagents_trace_end_to_end(tmp_path, spans):
    """The tool-calling master hands over to the subagents; the shortlist's shopper agents call
    their own tools. Real shortlist and shopper code; scripted models and the fake catalog."""
    master = ScriptedToolModel(script=[
        ("extract_style", {"why": "no profile yet"}),
        ("shortlist", {"item_types": ["sofa", "coffee_table", "floor_tiles"], "why": "first search"}),
        ("finalize", {"why": "shortlist ready"})])
    shortlist = make_shortlist(index=fake_index(), signals=("bm25",), shopper_model=FakeShopperModel())
    _run(build_graph(planner="llm", nodes={"style_extract": FAKE_NODES["style_extract"], "pricing": fake_pricing_node(),
                                           "planner": make_planner("llm", model=master), "shortlist": shortlist}),
         tmp_path)
    got = spans()
    assert len({s.context.trace_id for s in got}) == 1
    by_id = {s.context.span_id: s for s in got}
    up = lambda s: by_id[s.parent.span_id]                           # noqa: E731
    kind = lambda s: s.attributes.get(TYPE)                          # noqa: E731
    # the master: each of its three tool calls is a generation under the planner node
    assert [up(s).name for s in got if s.name == "ScriptedToolModel"] == ["planner"] * 3
    # its subagents: one shopper agent per shoppable item, under the shortlist node
    assert {s.name: up(s).name for s in got if kind(s) == "agent"} == {
        "shopper_agent:sofa": "shortlist", "shopper_agent:coffee_table": "shortlist"}
    # the shoppers' tool calls, each with the code-level tool it ran nested inside
    nested = [(s, up(s)) for s in got if kind(s) == "tool" and kind(up(s)) == "tool"]
    assert {(s.name, p.name) for s, p in nested} == {("catalog_search", "search_catalog"), ("landed_cost", "landed_price")}
    # the model's query, and the hard filters the code added to it
    searches = {(json.loads(p.attributes[INPUT])["query"], json.loads(s.attributes[INPUT])["max_price"])
                for s, p in nested if s.name == "catalog_search"}
    assert searches == {("low linen sofa", 2000.0), ("round light oak coffee table", 400.0)}

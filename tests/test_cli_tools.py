"""Tool calling on the CLI backend, offline: `claude` is replaced by a stub that records what it
was sent and replies the way `claude -p --output-format stream-json` does."""
import json
import subprocess

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from renopilot.llm import ClaudeCLIChat


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def _stub(monkeypatch, structured):
    sent = {}

    def run(argv, input, **_):
        sent["argv"], sent["stdin"] = argv, json.loads(input)
        line = json.dumps({"type": "result", "is_error": False, "result": "", "structured_output": structured,
                           "usage": {"input_tokens": 5, "output_tokens": 3}})
        return subprocess.CompletedProcess(argv, 0, stdout=line + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return sent


def _arg(argv, flag):
    return argv[argv.index(flag) + 1]


def test_tool_calls_come_back_as_standard_tool_calls(monkeypatch):
    sent = _stub(monkeypatch, {"content": "adding", "tool_calls": [{"name": "add", "args": {"a": 2, "b": 3}}]})
    msg = ClaudeCLIChat().bind_tools([add], tool_choice="any").invoke([SystemMessage("Use tools."),
                                                                        HumanMessage("2+3?")])
    assert [(c["name"], c["args"]) for c in msg.tool_calls] == [("add", {"a": 2, "b": 3})]
    assert msg.tool_calls[0]["id"].startswith("call_")
    calls = json.loads(_arg(sent["argv"], "--json-schema"))["properties"]["tool_calls"]
    assert calls["minItems"] == 1                                         # tool_choice="any": must call one
    only = calls["items"]["anyOf"][0]["properties"]
    assert only["name"]["enum"] == ["add"] and set(only["args"]["properties"]) == {"a", "b"}
    assert "Use tools." in _arg(sent["argv"], "--system-prompt")
    assert "- add: Add two integers." in _arg(sent["argv"], "--system-prompt")


def test_earlier_tool_calls_and_results_are_in_the_turn(monkeypatch):
    sent = _stub(monkeypatch, {"content": "5", "tool_calls": []})
    msg = ClaudeCLIChat().bind_tools([add]).invoke([
        HumanMessage("2+3?"), AIMessage("", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "c1"}]),
        ToolMessage("5", tool_call_id="c1", name="add")])
    assert msg.content == "5" and msg.tool_calls == []
    texts = [b["text"] for b in sent["stdin"]["message"]["content"]]
    assert texts == ["[human]", "2+3?", "[ai]", 'called add({"a": 2, "b": 3})', "[tool result: add]", "5"]


def test_plain_calls_are_unchanged(monkeypatch):
    sent = _stub(monkeypatch, None)
    ClaudeCLIChat().invoke([HumanMessage("hi")])
    assert "--json-schema" not in sent["argv"]

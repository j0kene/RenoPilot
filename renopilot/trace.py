"""JSONL traces: one row per node or model call, in runs/<run_id>.jsonl.

"Read the trace, not the story." Every eval failure points at its run_id, so the trace is
one `cat` away. A row is a flat dict, so traces can be diffed and loaded into pandas.

Langfuse (self-hosted, in docker-compose.yml) shows the same runs as a tree: one trace per graph
thread, with every node, model call and tool call nested where it happened, and the
interrupt/resume turns of one conversation joined up. It is on when LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are set; the JSONL stays the record the evals read.
"""
from __future__ import annotations

import contextvars
import functools
import inspect
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from opentelemetry import trace as otel

from renopilot import config

_current: contextvars.ContextVar["Tracer | None"] = contextvars.ContextVar("tracer", default=None)


class Tracer:
    def __init__(self, run_id: str | None = None, out_dir: Path = config.RUNS):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{self.run_id}.jsonl"
        self.rows: list[dict] = []

    def log(self, node: str, event: str, **fields) -> None:
        row = {"ts": round(time.time(), 3), "run_id": self.run_id, "node": node, "event": event, **fields}
        self.rows.append(row)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    def totals(self) -> dict:
        keys = ("input_tokens", "output_tokens", "cost_usd")
        return {k: sum(r.get(k, 0) or 0 for r in self.rows) for k in keys} | {
            "model_calls": sum(1 for r in self.rows if r["event"] == "model_call")}


@contextmanager
def tracing(run_id: str | None = None):
    """with tracing() as t: ... - every log() inside goes to t."""
    t = Tracer(run_id)
    token = _current.set(t)
    try:
        yield t
    finally:
        _current.reset(token)


def log(node: str, event: str, **fields) -> None:
    """Log to the active tracer; a no-op outside `tracing()` (e.g. in unit tests)."""
    t = _current.get()
    if t is not None:
        t.log(node, event, **fields)


def current() -> Tracer | None:
    return _current.get()


# --------------------------------------------------------------------------- Langfuse
def langfuse_on() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def omit_images(*, data: Any, **_) -> Any:
    """Langfuse mask for raw image blocks. The model call itself still shows the board: Langfuse
    uploads its images once to its media store. The spans around it would store them again
    inline as base64 (up to 12 x ~100 KB per vision call), so there they become a size note."""
    if isinstance(data, dict):
        if data.get("type") == "image" and isinstance(data.get("base64"), str):
            return {**data, "base64": f"<{len(data['base64']) * 3 // 4096} KB omitted>"}
        return {k: omit_images(data=v) for k, v in data.items()}
    if isinstance(data, list):
        return [omit_images(data=v) for v in data]
    return data


@functools.lru_cache(maxsize=1)
def _langfuse():
    """The process's Langfuse client (keys and URL from LANGFUSE_* env vars). It must be the
    first one created: later Langfuse()/get_client() calls share it, mask included."""
    from langfuse import Langfuse
    return Langfuse(mask=omit_images)


def run_config(name: str, *, thread_id: str | None = None, recursion_limit: int = 60,
               tags: list[str] | None = None, **metadata) -> dict:
    """Config for graph.invoke / graph.stream; use ONE per conversation. Its Langfuse trace id is
    derived from the thread id, so the turns of one conversation (start, each Command(resume=...))
    land in one trace. LangGraph ends a parked run normally, so the handler's own resume linking
    never fires for a top-level graph."""
    thread_id = thread_id or uuid.uuid4().hex
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit, "run_name": name}
    if langfuse_on():
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
        _langfuse()
        cfg["callbacks"] = [CallbackHandler(trace_context={"trace_id": Langfuse.create_trace_id(seed=thread_id)})]
        cfg["metadata"] = {"langfuse_session_id": thread_id, "langfuse_tags": tags or [], **metadata}
    return cfg


def langfuse_url(cfg: dict) -> str | None:
    """Link to the conversation's trace in Langfuse, or None when Langfuse is off or down."""
    handler = next((c for c in cfg.get("callbacks") or [] if hasattr(c, "last_trace_id")), None)
    if handler is None or handler.last_trace_id is None:
        return None
    try:
        return _langfuse().get_trace_url(trace_id=handler.last_trace_id)   # one API call per process
    except Exception:                                  # noqa: BLE001 - a missing link must not stop the run
        return None


def tool(name: str, *, inputs: tuple[str, ...] = (), output: Callable[[Any], Any] | None = None):
    """Mark a tool: deterministic code a node calls (catalog search, landed cost, rule lookup).
    Inside a traced graph run each call is a Langfuse "tool" observation under its node, with
    the named arguments as input and `output(result)` (default: the result) as output.
    Anywhere else (unit tests, scripts) it is a plain call."""
    def wrap(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def call(*args, **kwargs):
            if not (langfuse_on() and otel.get_current_span().get_span_context().is_valid):
                return fn(*args, **kwargs)                                  # no traced run around us
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            with _langfuse().start_as_current_observation(
                    as_type="tool", name=name, input={k: bound.arguments[k] for k in inputs}) as obs:
                result = fn(*args, **kwargs)
                obs.update(output=output(result) if output else result)
                return result
        return call
    return wrap

"""JSONL traces: one row per node or model call, in runs/<run_id>.jsonl.

"Read the trace, not the story." Every eval failure points at its run_id, so the trace is
one `cat` away. A row is a flat dict, so traces can be diffed and loaded into pandas.
"""
from __future__ import annotations

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

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

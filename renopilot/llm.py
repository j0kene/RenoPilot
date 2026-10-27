"""One place that decides which model answers which role.

While building and evaluating, every role runs on `claude -p` (the Claude Code CLI in
headless mode) through `ClaudeCLIChat`, a LangChain chat model. Nodes only ever see the
standard interface (`.invoke`, `.with_structured_output`), so switching to OpenAI at the
end is `RENOPILOT_BACKEND=openai` and nothing else.

Caveats of the CLI backend (write these up in the report):
  - no temperature / seed control, so run-to-run variance is part of what we measure
  - one process per call: a few seconds of overhead each time
  - subscription rate limits: run evals sequentially, make them resumable
  - `total_cost_usd` is the API-equivalent price, not what the subscription charges
  - no native tool calling for our tools: `bind_tools` asks for the tool calls as JSON
    (--json-schema) and turns them back into AIMessage.tool_calls, so agents work unchanged
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from renopilot import config


class ClaudeCLIError(RuntimeError):
    """Raised when the CLI fails. Transient failures say so in the message."""


_TRANSIENT = ("transient", "rate limit", "rate_limit", "overloaded", "529", "timeout",
              "timed out", "econnreset", "api_error", "503", "502")


def is_transient(e: Exception) -> bool:
    """For RetryPolicy(retry_on=...): retry the node, not the run."""
    return any(t in str(e).lower() for t in _TRANSIENT)


# --------------------------------------------------------------------------- images
def image_block(path: str | Path, max_side: int = config.VLM_IMAGE_MAX_SIDE) -> dict:
    """A LangChain standard image block, downscaled and re-encoded as JPEG.
    The same block works for ClaudeCLIChat and for langchain-openai."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return {"type": "image", "base64": base64.b64encode(buf.getvalue()).decode(),
            "mime_type": "image/jpeg"}


def _to_anthropic_block(block: Any) -> dict | None:
    """Convert one LangChain content block to an Anthropic content block."""
    if isinstance(block, str):
        return {"type": "text", "text": block}
    kind = block.get("type")
    if kind == "text":
        return {"type": "text", "text": block["text"]}
    if kind == "image_url":                                   # OpenAI-style
        url = block["image_url"]["url"] if isinstance(block["image_url"], dict) else block["image_url"]
        if url.startswith("data:"):
            head, data = url.split(",", 1)
            return {"type": "image", "source": {"type": "base64",
                    "media_type": head[5:].split(";")[0], "data": data}}
        return {"type": "image", "source": {"type": "url", "url": url}}
    if kind == "image":
        if "source" in block and isinstance(block["source"], dict):   # already Anthropic
            return {"type": "image", "source": block["source"]}
        data = block.get("base64") or (block.get("data") if block.get("source_type") == "base64" else None)
        if data:
            return {"type": "image", "source": {"type": "base64",
                    "media_type": block.get("mime_type", "image/jpeg"), "data": data}}
        if block.get("url"):
            return {"type": "image", "source": {"type": "url", "url": block["url"]}}
    return None


def _inline_refs(schema: dict) -> dict:
    """Pydantic emits $defs/$ref; inline them so the CLI's validator sees one flat schema."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def _tool_protocol(tools: list[dict], tool_choice: Any) -> tuple[str, dict]:
    """System-prompt addendum and reply schema for one tool-calling turn. Each call's args are
    constrained by that tool's own schema (anyOf over the tools)."""
    fns = [t["function"] for t in tools]
    name = tool_choice.get("function", {}).get("name") if isinstance(tool_choice, dict) else tool_choice
    if isinstance(name, str) and name not in ("auto", "any", "required"):
        fns = [f for f in fns if f["name"] == name]            # a forced tool
    must = tool_choice not in (None, "auto", False)
    calls = {"type": "array", "items": {"anyOf": [
        {"type": "object", "required": ["name", "args"],
         "properties": {"name": {"type": "string", "enum": [f["name"]]},
                        "args": _inline_refs(f.get("parameters") or {"type": "object", "properties": {}})}}
        for f in fns]}} | ({"minItems": 1} if must else {})
    schema = {"type": "object", "required": ["tool_calls"],
              "properties": {"content": {"type": "string"}, "tool_calls": calls}}
    text = ("TOOLS. You can call these tools; each call's args must match that tool's schema.\n"
            + "\n".join(f"- {f['name']}: {f.get('description', '')}" for f in fns)
            + "\nReply with tool_calls (name and args of each call) and, optionally, a short note in content. "
            + ("You must call at least one tool." if must else
               "To answer without a tool, leave tool_calls empty and answer in content."))
    return text, schema


# --------------------------------------------------------------------------- the model
class ClaudeCLIChat(BaseChatModel):
    """LangChain chat model backed by `claude -p` (stream-json in and out)."""

    model: str = "sonnet"
    timeout_s: int = config.CLI_TIMEOUT_S
    cli: str = "claude"

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def _build_turn(self, messages: list[BaseMessage]) -> tuple[str, list[dict]]:
        """System messages become --system-prompt; everything else is flattened into one
        user turn (the CLI is single-turn). Earlier turns are labelled by role."""
        system = "\n\n".join(m.content for m in messages if isinstance(m, SystemMessage)
                             and isinstance(m.content, str))
        convo = [m for m in messages if not isinstance(m, SystemMessage)]
        blocks: list[dict] = []
        for m in convo:
            if len(convo) > 1:
                label = f"[tool result: {m.name}]" if isinstance(m, ToolMessage) and m.name else f"[{m.type}]"
                blocks.append({"type": "text", "text": label})
            parts = m.content if isinstance(m.content, list) else [m.content]
            blocks += [b for b in (_to_anthropic_block(p) for p in parts)          # the API rejects empty text
                       if b and (b["type"] != "text" or b["text"].strip())]
            blocks += [{"type": "text", "text": f"called {c['name']}({json.dumps(c['args'], ensure_ascii=False)})"}
                       for c in getattr(m, "tool_calls", None) or []]
        return system, blocks

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None,
                  json_schema: dict | None = None, tools: list[dict] | None = None,
                  tool_choice: Any = None, **kwargs: Any) -> ChatResult:
        system, blocks = self._build_turn(messages)
        if tools and tool_choice != "none":
            protocol, json_schema = _tool_protocol(tools, tool_choice)
            system = f"{system}\n\n{protocol}" if system else protocol
        exe = shutil.which(self.cli) or self.cli
        argv = [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
                "--verbose", "--model", self.model, "--tools", "", "--no-session-persistence",
                "--system-prompt", system or "You are a helpful assistant."]
        if config.EFFORT:
            argv += ["--effort", config.EFFORT]
        if json_schema:
            argv += ["--json-schema", json.dumps(json_schema)]
        stdin = json.dumps({"type": "user", "message": {"role": "user", "content": blocks}})

        cwd = config.CACHE / "cli_cwd"          # empty dir: no CLAUDE.md or project context leaks in
        cwd.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(argv, input=stdin, capture_output=True, text=True,
                                  encoding="utf-8", timeout=self.timeout_s, cwd=cwd)
        except subprocess.TimeoutExpired as e:
            raise ClaudeCLIError(f"transient: claude CLI timed out after {self.timeout_s}s") from e

        result = None
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "result":
                    result = event
        if result is None or result.get("is_error"):
            detail = (result or {}).get("result") or proc.stderr.strip() or proc.stdout[-500:]
            raise ClaudeCLIError(f"claude CLI failed (exit {proc.returncode}): {detail}")

        usage = result.get("usage", {})
        n_in = (usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0))
        n_out = usage.get("output_tokens", 0)
        content, tool_calls = result.get("result") or "", []
        if tools and tool_choice != "none" and isinstance(result.get("structured_output"), dict):
            reply = result["structured_output"]
            content = reply.get("content") or ""
            tool_calls = [{"name": c["name"], "args": c.get("args") or {}, "id": f"call_{uuid.uuid4().hex[:24]}",
                           "type": "tool_call"} for c in reply.get("tool_calls") or [] if c.get("name")]
        msg = AIMessage(
            content=content, tool_calls=tool_calls,
            usage_metadata={"input_tokens": n_in, "output_tokens": n_out,
                            "total_tokens": n_in + n_out},
            response_metadata={"model": self.model, "backend": "claude_cli",
                               "cost_usd": result.get("total_cost_usd", 0.0),
                               "duration_ms": result.get("duration_ms"),
                               "structured_output": result.get("structured_output")},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: list, *, tool_choice: Any = None, **kwargs: Any):
        """Standard tool binding (create_agent and ToolNode call this); see _tool_protocol."""
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], tool_choice=tool_choice, **kwargs)

    def with_structured_output(self, schema: type[BaseModel] | dict, *, include_raw: bool = False,
                               **kwargs: Any):
        """Same contract as LangChain's: returns the parsed object, or
        {"raw", "parsed", "parsing_error"} when include_raw=True."""
        is_model = isinstance(schema, type) and issubclass(schema, BaseModel)
        json_schema = _inline_refs(schema.model_json_schema() if is_model else schema)

        def run(inp):
            messages = self._convert_input(inp).to_messages()
            raw = self.invoke(messages, json_schema=json_schema)
            data = raw.response_metadata.get("structured_output")
            parsed, err = None, None
            try:
                if data is None:                       # fall back to JSON in the text
                    text = raw.content if isinstance(raw.content, str) else ""
                    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
                parsed = schema.model_validate(data) if is_model else data
            except Exception as e:                     # noqa: BLE001 - reported as format_error
                err = e
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": err}
            if err:
                raise ClaudeCLIError(f"format_error: {err}") from err
            return parsed

        return RunnableLambda(run, name=schema.__name__ if is_model else "structured_output")   # span name in traces


# --------------------------------------------------------------------------- factory
def get_model(role: str) -> BaseChatModel:
    """role: vision | planner | rerank | shopper | intake."""
    name = config.MODELS[config.BACKEND][role]
    if config.BACKEND == "claude_cli":
        return ClaudeCLIChat(model=name)
    if config.BACKEND == "openai":
        from langchain.chat_models import init_chat_model
        return init_chat_model(f"openai:{name}", temperature=0, timeout=120, max_retries=3)
    raise ValueError(f"unknown backend {config.BACKEND!r}")


def usage_of(msg: AIMessage | None) -> dict:
    """Tokens, API-equivalent cost and latency of one model reply, for traces and AgentResult."""
    if msg is None:
        return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    u = msg.usage_metadata or {}
    meta = msg.response_metadata or {}
    return {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
            "cost_usd": meta.get("cost_usd", 0.0) or 0.0}

"""Smoke tests for the model backend, one per thing a node needs: text, structured output,
an image, tool calling. Run with RENOPILOT_LIVE=1 (each call takes a few seconds)."""
from typing import Literal

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from PIL import Image
from pydantic import BaseModel

from renopilot.llm import ClaudeCLIChat, image_block, usage_of

pytestmark = pytest.mark.live


class Swatch(BaseModel):
    name: Literal["blue", "gold", "red", "green", "white", "black"]
    side: Literal["left", "right"]


class Swatches(BaseModel):
    swatches: list[Swatch]


def test_text():
    msg = ClaudeCLIChat(model="haiku").invoke([SystemMessage("Reply with one word."),
                                               HumanMessage("What colour is the sky on a clear day?")])
    assert "blue" in msg.content.lower()
    assert usage_of(msg)["output_tokens"] > 0


def test_structured_nested_with_image(tmp_path):
    img = Image.new("RGB", (64, 64), (32, 96, 208))
    img.paste((224, 160, 32), (32, 0, 64, 64))
    path = tmp_path / "split.png"
    img.save(path)
    model = ClaudeCLIChat(model="haiku").with_structured_output(Swatches, include_raw=True)
    out = model.invoke([HumanMessage(content=[image_block(path),
                                              {"type": "text", "text": "List each colour block and its side."}])])
    assert out["parsing_error"] is None
    got = {(s.side, s.name) for s in out["parsed"].swatches}
    assert ("left", "blue") in got and ("right", "gold") in got


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_tool_calling():
    msg = ClaudeCLIChat(model="haiku").bind_tools([add], tool_choice="any").invoke(
        [HumanMessage("What is 2 + 3? Use the add tool.")])
    assert [(c["name"], c["args"]) for c in msg.tool_calls] == [("add", {"a": 2, "b": 3})]

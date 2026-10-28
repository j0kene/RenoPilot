"""Offline mode accepts only the explicitly scripted board."""
import pytest

from renopilot import config
from renopilot.demo import style_extract


def test_offline_example_refuses_unrelated_boards():
    with pytest.raises(ValueError, match="scripted"):
        style_extract({"board_images": [str(config.BOARDS / "industrial_01" / "01.jpg")]})

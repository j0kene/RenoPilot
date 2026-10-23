import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RENOPILOT_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="calls the real model; set RENOPILOT_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)

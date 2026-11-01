import os

import pytest


@pytest.fixture(autouse=True)
def isolate_tracing(monkeypatch):
    """Offline tests never inherit a developer's live tracing client from .env."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RENOPILOT_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="calls the real model; set RENOPILOT_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)

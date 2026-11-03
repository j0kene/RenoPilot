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


@pytest.fixture(autouse=True)
def prevent_unmarked_live_cli_calls(request, monkeypatch):
    """An offline regression must fail promptly instead of spending a CLI session."""
    if request.node.get_closest_marker("live"):
        return
    import subprocess
    from pathlib import Path

    original_run = subprocess.run

    def checked_run(command, *args, **kwargs):
        exe = command[0] if isinstance(command, (list, tuple)) else command.split()[0]
        if Path(exe).stem.lower() == "claude":
            raise AssertionError("Offline test called Claude CLI; inject a fake or mark the test live")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", checked_run)

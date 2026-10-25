"""FX: snapshot, cache, staleness, offline fallback, and no guessing."""
import json
from datetime import datetime, timedelta, timezone

import pytest
import requests

from renopilot.pricing import fx as fxmod
from renopilot.pricing.fx import FxError, FxProvider

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def test_sgd_is_identity():
    r = FxProvider(mode="snapshot").get("SGD")
    assert r.rate == 1.0 and not r.stale


def test_snapshot_rate():
    r = FxProvider(mode="snapshot").get("cny")
    assert r.rate == pytest.approx(0.19167) and r.rate_date == "2026-10-23"


def test_unknown_currency_raises_instead_of_guessing():
    with pytest.raises(FxError):
        FxProvider(mode="snapshot").get("XYZ")


class _Resp:
    def __init__(self, rate):
        self.rate = rate

    def raise_for_status(self):
        pass

    def json(self):
        return {"date": "2026-10-23", "rates": {"SGD": self.rate}}


def test_fresh_cache_is_used_without_network(tmp_path, monkeypatch):
    cache = tmp_path / "fx.json"
    cache.write_text(json.dumps({"CNY": {"rate": 0.2, "rate_date": "2026-10-23",
                                         "fetched_at": (NOW - timedelta(hours=3)).isoformat(), "source": "t"}}))
    monkeypatch.setattr(fxmod.requests, "get", lambda *a, **k: pytest.fail("should not call the API"))
    r = FxProvider(mode="live", cache=cache, now=lambda: NOW).get("CNY")
    assert r.rate == 0.2 and not r.stale


def test_old_cache_is_refreshed(tmp_path, monkeypatch):
    cache = tmp_path / "fx.json"
    cache.write_text(json.dumps({"CNY": {"rate": 0.2, "rate_date": "2026-10-20",
                                         "fetched_at": (NOW - timedelta(hours=30)).isoformat(), "source": "t"}}))
    monkeypatch.setattr(fxmod.requests, "get", lambda *a, **k: _Resp(0.19167))
    r = FxProvider(mode="live", cache=cache, now=lambda: NOW).get("CNY")
    assert r.rate == pytest.approx(0.19167) and not r.stale
    assert json.loads(cache.read_text())["CNY"]["rate"] == pytest.approx(0.19167)


def test_offline_with_old_cache_is_flagged_stale(tmp_path, monkeypatch):
    cache = tmp_path / "fx.json"
    cache.write_text(json.dumps({"CNY": {"rate": 0.2, "rate_date": "2026-10-20",
                                         "fetched_at": (NOW - timedelta(hours=30)).isoformat(), "source": "t"}}))

    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(fxmod.requests, "get", boom)
    r = FxProvider(mode="live", cache=cache, now=lambda: NOW).get("CNY")
    assert r.rate == 0.2 and r.stale                     # proposal: flag rates older than 24 h


def test_offline_without_cache_raises(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(fxmod.requests, "get", boom)
    with pytest.raises(FxError):
        FxProvider(mode="live", cache=tmp_path / "none.json", now=lambda: NOW).get("CNY")

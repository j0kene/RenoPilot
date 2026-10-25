"""Exchange rates to SGD.

    live      Frankfurter (ECB reference rates), cached per day in data/cache/fx_cache.json
    snapshot  data/pricing/fx_snapshot.json, frozen for tests and evals (RENOPILOT_FX=snapshot)

Stale means our copy was fetched more than 24 h ago (proposal: "flags exchange rates older than
24 hours"). The rate's own date is reported too; ECB publishes on business days only, so on a
weekend the rate date is legitimately two days old while our copy is fresh.

An unknown currency raises: a missing rate is never guessed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from renopilot import config

SNAPSHOT_PATH = config.DATA / "pricing" / "fx_snapshot.json"
CACHE_PATH = config.CACHE / "fx_cache.json"
API = "https://api.frankfurter.dev/v1/latest"


class FxError(RuntimeError):
    pass


@dataclass(frozen=True)
class FxRate:
    currency: str
    rate: float              # 1 unit of `currency` = rate SGD
    rate_date: str
    fetched_at: str          # ISO timestamp of our copy
    source: str
    stale: bool


def _is_stale(fetched_at: str, now: datetime, hours: float) -> bool:
    return now - datetime.fromisoformat(fetched_at) > timedelta(hours=hours)


class FxProvider:
    def __init__(self, mode: str | None = None, snapshot: Path = SNAPSHOT_PATH, cache: Path = CACHE_PATH,
                 stale_hours: float = 24, now=None):
        self.mode = mode or os.environ.get("RENOPILOT_FX", "live")
        self.snapshot, self.cache, self.stale_hours = Path(snapshot), Path(cache), stale_hours
        self._now = now or (lambda: datetime.now(timezone.utc))

    def get(self, currency: str) -> FxRate:
        currency = (currency or "").upper()
        if currency == "SGD":
            now = self._now().isoformat()
            return FxRate("SGD", 1.0, now[:10], now, "identity", False)
        if self.mode == "snapshot":
            return self._from_snapshot(currency)
        return self._from_cache_or_api(currency)

    # ------------------------------------------------------------------ snapshot
    def _from_snapshot(self, currency: str) -> FxRate:
        snap = json.loads(self.snapshot.read_text(encoding="utf-8"))
        if currency not in snap["rates_to_sgd"]:
            raise FxError(f"no {currency}->SGD rate in the FX snapshot")
        # a frozen snapshot is reproducible by design: staleness is reported, but relative to its own date
        return FxRate(currency, float(snap["rates_to_sgd"][currency]), snap["rate_date"], snap["fetched_at"],
                      snap["source"], False)

    # ------------------------------------------------------------------ live
    def _from_cache_or_api(self, currency: str) -> FxRate:
        cache = json.loads(self.cache.read_text(encoding="utf-8")) if self.cache.exists() else {}
        hit = cache.get(currency)
        now = self._now()
        if hit and not _is_stale(hit["fetched_at"], now, self.stale_hours):
            return FxRate(currency, hit["rate"], hit["rate_date"], hit["fetched_at"], hit["source"], False)
        try:
            r = requests.get(API, params={"base": currency, "symbols": "SGD"}, timeout=20)
            r.raise_for_status()
            data = r.json()
            rate = float(data["rates"]["SGD"])
        except (requests.RequestException, KeyError, ValueError) as e:
            if hit:                                      # offline: use the old copy, flagged stale
                return FxRate(currency, hit["rate"], hit["rate_date"], hit["fetched_at"], hit["source"], True)
            raise FxError(f"cannot get a {currency}->SGD rate: {e}") from e
        entry = {"rate": rate, "rate_date": data["date"], "fetched_at": now.isoformat(), "source": "Frankfurter (ECB)"}
        cache[currency] = entry
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        return FxRate(currency, rate, entry["rate_date"], entry["fetched_at"], entry["source"], False)

"""The pricing rules table (data/pricing/rules.json): GST, low-value-goods and import-relief
rules, delivery fee tables, shipping rates, typical sizes. Every figure carries its source and
the date it was checked; unconfirmed ones are flagged as assumptions and surface on quotes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from renopilot import config

RULES_PATH = config.DATA / "pricing" / "rules.json"


@dataclass(frozen=True)
class PricingRules:
    raw: dict

    @property
    def gst_rate(self) -> float:
        return self.raw["gst"]["rate"]

    @property
    def lvg_threshold(self) -> float:
        return self.raw["low_value_goods"]["threshold_sgd"]

    @property
    def relief_modes(self) -> list[str]:
        return self.raw["import_relief"]["modes"]

    def ovr_registered(self, source: str) -> bool:
        return source in self.raw["low_value_goods"]["ovr_registered_sources"]

    def delivery(self, source: str) -> dict:
        return self.raw["local_delivery"].get(source) or self.raw["local_delivery"]["other_sg"]

    def shipping(self, key: str) -> dict:
        return self.raw["overseas_shipping"][key]

    def typical(self, item_type: str) -> dict:
        return self.raw["typical_by_item_type"].get(item_type, {"volume_m3": 0.3, "weight_kg": 20, "longest_cm": 120})

    @property
    def fx_stale_hours(self) -> float:
        return self.raw["fx"]["stale_after_hours"]


@lru_cache(maxsize=4)
def load_rules(path: Path = RULES_PATH) -> PricingRules:
    return PricingRules(json.loads(Path(path).read_text(encoding="utf-8")))

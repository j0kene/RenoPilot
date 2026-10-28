"""Shared data contracts. Every node reads and writes these, so they are frozen early:
the Compliance guard and the Pricing agent plug in by reading `Suggestion` and filling
`verdict` / `price_quote`, without touching the graph.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

from renopilot.vocab import COLOURS, ITEM_TYPES, MATERIALS, ROOMS, STYLE_TAGS

StyleTag = Literal[STYLE_TAGS]          # type: ignore[valid-type]
Colour = Literal[COLOURS]               # type: ignore[valid-type]
Material = Literal[MATERIALS]           # type: ignore[valid-type]
ItemType = Literal[ITEM_TYPES]          # type: ignore[valid-type]
Room = Literal[ROOMS]                   # type: ignore[valid-type]
WorkType = Literal["loose_furniture", "finish", "built_in"]
WorkMethod = Literal["unspecified", "overlay", "replace", "raise", "demolish", "erect"]
Source = Literal["abo", "ikea_sg", "castlery", "taobao", "other_sg"]


# --------------------------------------------------------------------------- the owner
class ProjectProfile(BaseModel):
    """What the owner tells us once. Budgets are SGD caps per single item of that type."""
    flat_type: Optional[Literal["2-room", "3-room", "4-room", "5-room", "executive", "other"]] = None
    completion_date: Optional[date] = Field(None, description="Block completion date (drives the 3-year finish rules)")
    total_budget: Optional[float] = Field(None, ge=0, description="Total furniture budget, SGD")
    budget_by_category: dict[str, float] = Field(default_factory=dict,
                                                 description="item_type -> max SGD for one item")
    notes: str = ""
    as_of: Optional[date] = Field(None, description="Evaluation date; omitted for normal runs")


# --------------------------------------------------------------------------- the board
class ItemSpec(BaseModel):
    item_type: ItemType
    description: str = Field(description="Short visual description, at most 20 words, e.g. "
                                         "'low light-oak coffee table with rounded corners'")
    colours: list[Colour] = Field(default_factory=list, max_length=3)
    materials: list[Material] = Field(default_factory=list, max_length=3)
    priority: Literal["must", "nice"] = "must"
    room: Optional[Room] = Field(None, description="Room containing THIS item, even on a mixed board; null if unknown")
    work_method: WorkMethod = Field("unspecified", description="Requested renovation method; never substitute a safer method")
    structural: Optional[bool] = Field(None, description="For wall demolition: true for structural/RC, false only if confirmed non-structural")


class StyleProfile(BaseModel):
    style_tags: list[StyleTag] = Field(min_length=1, max_length=3, description="Most dominant first")
    palette: list[Colour] = Field(min_length=1, max_length=6)
    materials: list[Material] = Field(default_factory=list, max_length=6)
    room: Room = "mixed"
    items: list[ItemSpec] = Field(default_factory=list, max_length=12,
                                  description="Distinct items visible on the board worth shopping for")
    summary: str = Field(description="One sentence describing the look")


# --------------------------------------------------------------------------- the catalog
class Product(BaseModel):
    id: str
    source: Source
    item_type: ItemType
    title: str
    colour: str = ""
    material: str = ""
    dims: str = ""
    price: Optional[float] = None          # as listed
    currency: Optional[str] = None         # SGD | CNY | ...
    price_sgd: Optional[float] = None      # listed price in SGD (indicative; landed cost is Pricing's job)
    url: str = ""
    image_path: str = ""
    retrieved_at: str = ""
    # shipping inputs for overseas listings (all optional; Pricing estimates what is missing)
    weight_kg: Optional[float] = None
    volume_m3: Optional[float] = None
    shipping_quote: Optional[float] = None   # quoted shipping to Singapore, in the listing currency
    ships_by: Optional[Literal["sea", "air", "local"]] = None
    synthetic: bool = False                  # made-up fixture row: never shown as a real offer


# --------------------------------------------------------------------------- pricing
class PriceLine(BaseModel):
    label: str
    amount_sgd: float
    note: str = ""


class PriceQuote(BaseModel):
    """Landed cost of ONE item in SGD, written by deterministic code (renopilot/pricing).
    Lines add up to total_sgd exactly (each line is rounded to the cent first)."""
    listed: float
    currency: str
    fx_rate: float = 1.0
    fx_date: Optional[str] = None
    fx_stale: bool = False
    item_sgd: float
    shipping_sgd: float = 0.0
    local_delivery_sgd: float = 0.0       # if bought alone; basket totals charge it once per order
    gst_sgd: float = 0.0
    gst_basis: Literal["included", "checkout_ovr", "import_cif", "none"] = "included"
    total_sgd: float
    lines: list[PriceLine] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    estimated: bool = False
    synthetic: bool = False


class PricedAlternative(BaseModel):
    product_id: str
    source: Source
    title: str
    url: str = ""
    quote: PriceQuote
    why_comparable: str = ""


# --------------------------------------------------------------------------- downstream
class Verdict(BaseModel):
    """Written by the Compliance guard. The guard's verdict cannot be overridden by any agent.
    A finish or built-in can usually be done more than one way (overlay or replace, ...): `options`
    holds the guard's verdict for each route, easiest first, and the verdict itself is the route in
    effect. Only the owner switches routes, at the approval gate, and only to one of these."""
    decision: Literal["allow", "allow_with_permit", "block", "escalate"]
    reason: str = ""
    rule_id: Optional[str] = None
    route: str = ""                                    # how the work is done, e.g. "Overlay: ..."
    conditions: list[str] = Field(default_factory=list)
    alternative: Optional[str] = None
    citation: Optional[str] = None
    options: list[Verdict] = Field(default_factory=list)


class Suggestion(BaseModel):
    product_id: Optional[str] = None        # None for a finish/built-in the board asked for but we don't sell
    item_type: ItemType
    work_type: WorkType
    title: str = ""
    reason: str = ""                        # one line: why this matches the board
    score: float = 0.0
    price_sgd: Optional[float] = None
    url: str = ""
    image_path: str = ""
    verdict: Optional[Verdict] = None       # Compliance guard
    price_quote: Optional[PriceQuote] = None          # Pricing agent: landed cost of this item
    comparables: list[PricedAlternative] = Field(default_factory=list)   # cheaper/similar elsewhere
    status: Literal["proposed", "approved", "rejected"] = "proposed"
    room: Optional[Room] = None
    work_method: WorkMethod = "unspecified"
    structural: Optional[bool] = None

    @property
    def key(self) -> str:
        """Independent choices for, e.g., living-room and bathroom floor tiles."""
        return self.product_id or f"{self.item_type}:{self.room or 'unknown'}:{self.work_method}"

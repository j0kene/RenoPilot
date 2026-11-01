"""Shared data contracts. Every node reads and writes these, so they are frozen early:
the Compliance guard and the Pricing agent plug in by reading `Suggestion` and filling
`verdict` / `price_quote`, without touching the graph.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated, Literal, Optional

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
class WorkEstimate(BaseModel):
    request_id: str = ""
    description: str
    quantity: float = Field(1.0, gt=0, allow_inf_nan=False)
    unit: str = "job"
    unit_rate_sgd: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    source: str = ""
    checked_at: Optional[date] = None

    @property
    def total_sgd(self) -> Optional[float]:
        return None if self.unit_rate_sgd is None else float((Decimal(str(self.quantity)) *
            Decimal(str(self.unit_rate_sgd))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class ProjectProfile(BaseModel):
    """What the owner tells us once. Budgets are SGD caps per single item of that type."""
    flat_type: Optional[Literal["2-room", "3-room", "4-room", "5-room", "executive", "other"]] = None
    completion_date: Optional[date] = Field(None, description="Block completion date (drives the 3-year finish rules)")
    total_budget: Optional[float] = Field(None, ge=0, allow_inf_nan=False, description="Total furniture budget, SGD")
    budget_by_category: dict[ItemType, Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(default_factory=dict,
                                                 description="item_type -> max SGD for one item")
    notes: str = ""
    as_of: Optional[date] = Field(None, description="Evaluation date; omitted for normal runs")
    works_budget: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    work_estimates: list[WorkEstimate] = Field(default_factory=list)


# --------------------------------------------------------------------------- the board
class WorkDetails(BaseModel):
    overlay_thickness_mm: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    existing_floor_layers: Optional[int] = Field(None, ge=0)
    ceiling_clearance_mm: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    non_combustible: Optional[bool] = None
    gas_pipe_enclosed: Optional[bool] = None
    partition_material: Optional[Literal["hollow_block", "glass_block", "gypsum", "plywood", "plastic", "asbestos", "other"]] = None


class ItemSpec(BaseModel):
    request_id: str = ""
    item_type: ItemType
    description: str = Field(description="Short visual description, at most 20 words, e.g. "
                                         "'low light-oak coffee table with rounded corners'")
    colours: list[Colour] = Field(default_factory=list, max_length=3)
    materials: list[Material] = Field(default_factory=list, max_length=3)
    priority: Literal["must", "nice"] = "must"
    room: Optional[Room] = Field(None, description="Room containing THIS item, even on a mixed board; null if unknown")
    work_method: WorkMethod = Field("unspecified", description="Requested renovation method; never substitute a safer method")
    structural: Optional[bool] = Field(None, description="For wall demolition: true for structural/RC, false only if confirmed non-structural")
    work_details: WorkDetails = Field(default_factory=WorkDetails)
    room_label: str = Field("", description="Owner's room name, e.g. Bedroom 1 or Bedroom 2")
    quantity: int = Field(1, ge=1, le=100)
    max_width_cm: Optional[float] = Field(None, gt=0, allow_inf_nan=False)
    max_depth_cm: Optional[float] = Field(None, gt=0, allow_inf_nan=False)
    max_height_cm: Optional[float] = Field(None, gt=0, allow_inf_nan=False)


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
    delivery_weight_kg: Optional[float] = None
    delivery_longest_cm: Optional[float] = None


class PricedAlternative(BaseModel):
    product_id: str
    source: Source
    title: str
    url: str = ""
    quote: PriceQuote
    why_comparable: str = ""
    image_path: str = ""
    fit_note: str = ""
    retrieved_at: str = ""


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
    request_id: str = ""
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
    source: Optional[Source] = None
    room_label: str = ""
    quantity: int = Field(1, ge=1, le=100)
    work_details: WorkDetails = Field(default_factory=WorkDetails)
    fit_note: str = ""
    retrieved_at: str = ""
    dims: str = ""
    material: str = ""

    @property
    def key(self) -> str:
        """Independent choices for, e.g., living-room and bathroom floor tiles."""
        return self.product_id or self.request_id or f"{self.item_type}:{self.room or 'unknown'}:{self.work_method}"

    @property
    def selection_key(self) -> str:
        return self.request_id or self.item_type

    @property
    def label(self) -> str:
        return self.item_type.replace("_", " ") + (f" ({self.room_label})" if self.room_label else "")

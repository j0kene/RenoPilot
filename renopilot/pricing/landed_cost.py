"""Landed cost of one listing in SGD. Pure code, no model: every rule comes from rules.json.

    local retailer (ikea_sg, castlery, other_sg)
        listed SGD price (GST already included) + that retailer's delivery fee if bought alone
    overseas (e.g. taobao), by sea  [default for furniture]
        item (FX) + international shipping (quoted, else estimated from volume)
        + import GST 9% on CIF (= item + freight + insurance): sea imports get NO S$400 relief
    overseas, by air
        item <= S$400 from an OVR-registered platform  -> GST 9% charged at checkout on the item
        otherwise CIF <= S$400                         -> import relief, no GST
        otherwise                                      -> import GST 9% on the whole CIF

Money is Decimal; every line is rounded half-up to the cent BEFORE summing, so the lines
always add up to the total exactly (what a hand-worked check expects).
"""
from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

from renopilot import trace
from renopilot.pricing.fx import FxProvider
from renopilot.pricing.rules import PricingRules, load_rules
from renopilot.schemas import PriceLine, PriceQuote, Product

LOCAL_SOURCES = {"ikea_sg", "castlery", "other_sg"}
CENT = Decimal("0.01")


def money(x) -> Decimal:
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)


def dims_cm(dims: str) -> list[float]:
    """Numbers in a dimensions string, in cm. Handles '90x55 cm', '180x85x75 cm', 'W81.3in D35in'."""
    if not dims:
        return []
    inches = bool(re.search(r"\d\s*(in|inch|inches|\")\b", dims, re.I)) and "cm" not in dims.lower()
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", dims)]
    return [round(n * 2.54, 1) if inches else n for n in nums]


# --------------------------------------------------------------------------- local delivery
def ikea_fee(weight_kg: float, longest_cm: float, rules: PricingRules) -> tuple[Decimal, str]:
    d = rules.delivery("ikea_sg")
    if weight_kg <= d["parcel"]["max_kg"] and longest_cm < d["parcel"]["max_length_cm"]:
        return money(d["parcel"]["fee"]), "IKEA parcel delivery"
    if weight_kg <= d["parcel_max"]["max_kg"] and longest_cm < d["parcel_max"]["max_length_cm"]:
        return money(d["parcel_max"]["fee"]), "IKEA parcel max delivery"
    return money(d["home"]["fee"]), "IKEA home delivery"


def castlery_fee(order_total: Decimal, rules: PricingRules) -> tuple[Decimal, str]:
    d = rules.delivery("castlery")
    if order_total >= Decimal(str(d["free_from"])):
        return money(0), f"Castlery delivery (free from S${d['free_from']})"
    for upper, fee in d["tiers"]:
        if order_total <= Decimal(str(upper)):
            return money(fee), "Castlery standard delivery"
    return money(d["tiers"][-1][1]), "Castlery standard delivery"


def local_delivery(p: Product, item_sgd: Decimal, rules: PricingRules) -> tuple[Decimal, str, list[str], bool]:
    """(fee, label, assumptions, estimated) for buying this one item alone."""
    if p.source == "ikea_sg":
        t = rules.typical(p.item_type)
        known = dims_cm(p.dims)
        weight = p.weight_kg if p.weight_kg is not None else t["weight_kg"]
        longest = max(known) if known else t["longest_cm"]
        estimated = p.weight_kg is None or not known
        fee, label = ikea_fee(weight, longest, rules)
        notes = [f"IKEA delivery tier from typical {p.item_type.replace('_', ' ')} size/weight"] if estimated else []
        return fee, label, notes, estimated
    if p.source == "castlery":
        fee, label = castlery_fee(item_sgd, rules)
        return fee, label, [], False
    d = rules.delivery("other_sg")
    return money(d["fee"]), "Delivery", [d["note"]], False


# --------------------------------------------------------------------------- the quote
@trace.tool("landed_cost", inputs=("p",))
def landed_cost(p: Product, rules: PricingRules | None = None, fx: FxProvider | None = None) -> PriceQuote:
    if p.price is None:
        raise ValueError(f"{p.id} has no listed price")
    rules = rules or load_rules()
    fx = fx or FxProvider(stale_hours=rules.fx_stale_hours)
    currency = (p.currency or "SGD").upper()
    rate = fx.get(currency)
    item = money(Decimal(str(p.price)) * Decimal(str(rate.rate)))
    lines = [PriceLine(label="Item", amount_sgd=float(item),
                       note="" if currency == "SGD" else f"{currency} {p.price:,.2f} x {rate.rate} ({rate.rate_date})")]
    assumptions: list[str] = []
    estimated = False
    shipping = delivery = gst = money(0)

    if p.source in LOCAL_SOURCES and p.ships_by in (None, "local"):
        delivery, label, notes, estimated = local_delivery(p, item, rules)
        assumptions += notes
        lines.append(PriceLine(label=label, amount_sgd=float(delivery), note="if bought alone; charged once per order"))
        basis = "included"
        lines.append(PriceLine(label="GST", amount_sgd=0.0, note="included in the listed price"))
    else:
        mode = p.ships_by if p.ships_by in ("sea", "air") else "sea"
        if p.shipping_quote is not None:
            shipping = money(Decimal(str(p.shipping_quote)) * Decimal(str(rate.rate)))
            ship_note = f"quoted: {currency} {p.shipping_quote:,.2f}"
        elif mode == "sea":
            t = rules.shipping("sea_bulky_cny")
            vol = p.volume_m3 if p.volume_m3 is not None else rules.typical(p.item_type)["volume_m3"]
            estimated = estimated or p.volume_m3 is None
            cny = Decimal(str(t["min_charge_cny"])) if vol <= t["min_charge_up_to_m3"] \
                else Decimal(str(t["per_m3_cny"])) * Decimal(str(vol))
            cny_rate = fx.get("CNY")
            shipping = money(cny * Decimal(str(cny_rate.rate)))
            ship_note = f"estimated sea freight for {vol} m3 (CNY {cny:.2f})"
            assumptions.append(t["note"])
            if p.volume_m3 is None:
                assumptions.append(f"packed volume taken as a typical {p.item_type.replace('_', ' ')}")
        else:
            t = rules.shipping("air_sgd_per_kg")
            wt = p.weight_kg if p.weight_kg is not None else rules.typical(p.item_type)["weight_kg"]
            estimated = True if p.weight_kg is None else estimated
            shipping = money(Decimal(str(t["rate"])) * Decimal(str(wt)))
            ship_note = f"estimated air freight for {wt} kg"
            assumptions.append(t["note"])
        lines.append(PriceLine(label=f"Shipping to Singapore ({mode})", amount_sgd=float(shipping), note=ship_note))

        lm = rules.shipping("last_mile_sgd")
        delivery = money(lm["fee"])
        lines.append(PriceLine(label="Local delivery", amount_sgd=float(delivery), note="assumed included by the forwarder"))
        assumptions.append(lm["note"])

        insurance = money(rules.shipping("insurance_sgd"))
        cif = item + shipping + insurance
        threshold = Decimal(str(rules.lvg_threshold))
        rate_gst = Decimal(str(rules.gst_rate))
        if mode in rules.relief_modes:
            if rules.ovr_registered(p.source) and item <= threshold:
                gst, basis, gnote = money(item * rate_gst), "checkout_ovr", "charged at checkout (OVR, item <= S$400)"
            elif cif <= threshold:
                gst, basis, gnote = money(0), "none", "import relief: air/post, CIF <= S$400"
            else:
                gst, basis, gnote = money(cif * rate_gst), "import_cif", f"import GST on CIF S${cif}"
        else:
            gst, basis, gnote = money(cif * rate_gst), "import_cif", f"import GST on CIF S${cif} (no relief by sea)"
        lines.append(PriceLine(label="GST", amount_sgd=float(gst), note=gnote))

    total = item + shipping + delivery + gst
    return PriceQuote(listed=float(p.price), currency=currency, fx_rate=rate.rate, fx_date=rate.rate_date,
                      fx_stale=rate.stale, item_sgd=float(item), shipping_sgd=float(shipping),
                      local_delivery_sgd=float(delivery), gst_sgd=float(gst), gst_basis=basis,
                      total_sgd=float(total), lines=lines, assumptions=list(dict.fromkeys(assumptions)),
                      estimated=estimated, synthetic=p.synthetic)


# --------------------------------------------------------------------------- baskets
@trace.tool("basket_total", inputs=("items",))
def basket_total(items: list[tuple[str, PriceQuote]], rules: PricingRules | None = None) -> dict:
    """Total for buying several items: each LOCAL retailer's delivery fee is charged once per
    order (Castlery's tier is re-read on the order total; IKEA takes the largest single fee)."""
    rules = rules or load_rules()
    goods = money(0)
    by_source: dict[str, list[PriceQuote]] = {}
    for source, q in items:
        goods += money(q.total_sgd) - money(q.local_delivery_sgd)
        by_source.setdefault(source, []).append(q)
    delivery = {}
    for source, qs in by_source.items():
        if source == "castlery":
            delivery[source], _ = castlery_fee(sum((money(q.item_sgd) for q in qs), money(0)), rules)
        elif source in LOCAL_SOURCES:
            delivery[source] = max(money(q.local_delivery_sgd) for q in qs)
        else:
            delivery[source] = sum((money(q.local_delivery_sgd) for q in qs), money(0))
    total = goods + sum(delivery.values(), money(0))
    return {"total_sgd": float(total), "delivery_by_source": {k: float(v) for k, v in delivery.items()}}

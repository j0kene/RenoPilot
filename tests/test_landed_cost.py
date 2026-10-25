"""Landed cost: 30 hand-worked cases, exact to the cent (proposal target: 100%).

Expected totals were worked out by hand from data/pricing/rules.json (checked 2026-10-25), NOT
by running the code. Test FX: 1 CNY = 0.19 SGD, 1 USD = 1.25 SGD (chosen so the arithmetic is
checkable by hand). Rounding: each line half-up to the cent, then summed.

Rules used:  GST 9%.  IKEA: parcel S$10 (<=15 kg and longest < 140 cm), parcel max S$20 (<=25 kg,
< 140 cm), else home S$35.  Castlery: <=50 S$9, <=150 S$15, <=300 S$19, <=400 S$29, <=499.99
S$49, >=500 free.  Sea: no import relief, GST 9% on CIF (item + freight).  Sea freight estimate:
CNY 225 up to 0.5 m3, else CNY 450 per m3.  Air: OVR platform (taobao) and item <= S$400 ->
GST on item at checkout; else CIF <= 400 -> no GST; else GST on whole CIF.  Air estimate S$8/kg.
"""
import json

import pytest

from renopilot.pricing.fx import FxProvider
from renopilot.pricing.landed_cost import basket_total, landed_cost
from renopilot.pricing.rules import load_rules
from renopilot.schemas import Product


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    snap = tmp_path_factory.mktemp("fx") / "snap.json"
    snap.write_text(json.dumps({"fetched_at": "2026-10-25T00:00:00+08:00", "rate_date": "2026-10-23",
                                "source": "test", "rates_to_sgd": {"SGD": 1.0, "CNY": 0.19, "USD": 1.25}}))
    return FxProvider(mode="snapshot", snapshot=snap)


def P(**kw):
    base = {"id": "x", "source": "ikea_sg", "item_type": "side_table", "title": "t", "currency": "SGD"}
    return Product(**(base | kw))


CASES = [
    # ---------------------------------------------------------------- IKEA SG (GST included)
    ("ikea lamp parcel", P(price=49.90, weight_kg=2, dims="30x30x50 cm", item_type="table_lamp"), 59.90),        # 49.90 + 10
    ("ikea sofa home", P(price=639, weight_kg=60, dims="200x88 cm", item_type="sofa"), 674.00),                  # 639 + 35
    ("ikea 15.0kg is parcel", P(price=129, weight_kg=15.0, dims="100 cm"), 139.00),                             # <=15 -> 10
    ("ikea 15.5kg parcel max", P(price=199, weight_kg=15.5, dims="100 cm"), 219.00),                            # <=25 -> 20
    ("ikea 140cm is home", P(price=89, weight_kg=10, dims="140x40 cm"), 124.00),                                # not < 140 -> 35
    ("ikea 26kg is home", P(price=299, weight_kg=26, dims="100 cm"), 334.00),                                   # >25 -> 35
    ("ikea typical sofa", P(price=999, item_type="sofa"), 1034.00),                                             # typical 60kg/200cm -> 35
    ("ikea typical lamp", P(price=30, item_type="table_lamp"), 40.00),                                          # typical 2kg/50cm -> 10
    # ---------------------------------------------------------------- Castlery (GST included)
    ("castlery 49", P(source="castlery", price=49), 58.00),                                                     # +9
    ("castlery 50.00 edge", P(source="castlery", price=50.00), 59.00),                                          # <=50 -> 9
    ("castlery 50.01", P(source="castlery", price=50.01), 65.01),                                               # -> 15
    ("castlery 150.00", P(source="castlery", price=150.00), 165.00),                                            # <=150 -> 15
    ("castlery 299", P(source="castlery", price=299), 318.00),                                                  # -> 19
    ("castlery 400.00", P(source="castlery", price=400.00), 429.00),                                            # <=400 -> 29
    ("castlery 499.99", P(source="castlery", price=499.99), 548.99),                                            # -> 49
    ("castlery 500 free", P(source="castlery", price=500), 500.00),                                             # >=500 -> 0
    ("castlery 1499 free", P(source="castlery", price=1499), 1499.00),
    # ---------------------------------------------------------------- overseas by sea (no relief)
    # item 1000*0.19=190.00; ship 300*0.19=57.00; CIF 247.00; GST 22.23; total 269.23
    ("sea quoted", P(source="taobao", currency="CNY", price=1000, shipping_quote=300, ships_by="sea"), 269.23),
    # item 380.00; 0.4 m3 -> CNY 225 -> 42.75; CIF 422.75; GST 38.0475 -> 38.05; total 460.80
    ("sea 0.4m3 min charge", P(source="taobao", currency="CNY", price=2000, volume_m3=0.4, ships_by="sea"), 460.80),
    # item 380.00; 1.2 m3 * 450 = CNY 540 -> 102.60; CIF 482.60; GST 43.434 -> 43.43; total 526.03
    ("sea 1.2m3", P(source="taobao", currency="CNY", price=2000, volume_m3=1.2, ships_by="sea"), 526.03),
    # sofa typical 1.2 m3 -> 102.60; item 570.00; CIF 672.60; GST 60.534 -> 60.53; total 733.13
    ("sea typical sofa", P(source="taobao", currency="CNY", price=3000, item_type="sofa", ships_by="sea"), 733.13),
    # 0.5 m3 is still the min charge: 42.75; item 95.00; CIF 137.75; GST 12.3975 -> 12.40; total 150.15
    ("sea 0.5m3 edge", P(source="taobao", currency="CNY", price=500, volume_m3=0.5, ships_by="sea"), 150.15),
    # free shipping quoted: item 152.00; CIF 152.00; GST 13.68; total 165.68 (sea: GST even under S$400)
    ("sea zero shipping", P(source="taobao", currency="CNY", price=800, shipping_quote=0, ships_by="sea"), 165.68),
    # ---------------------------------------------------------------- overseas by air
    # OVR: item 190.00 <= 400 -> GST on item 17.10 at checkout; ship 19.00; total 226.10
    ("air ovr low value", P(source="taobao", currency="CNY", price=1000, shipping_quote=100, ships_by="air"), 226.10),
    # item USD 320*1.25 = 400.00 (edge, still <= 400) -> GST 36.00 on item; ship 25.00; total 461.00
    ("air ovr 400.00 edge", P(source="taobao", currency="USD", price=320, shipping_quote=20, ships_by="air"), 461.00),
    # item 400.0125 -> 400.01 > 400 -> CIF 425.01 -> GST 38.2509 -> 38.25; total 463.26
    ("air ovr 400.01", P(source="taobao", currency="USD", price=320.01, shipping_quote=20, ships_by="air"), 463.26),
    # not OVR-registered: CIF 125.00 + 12.50 = 137.50 <= 400 -> relief, no GST; total 137.50
    ("air relief", P(source="other_sg", currency="USD", price=100, shipping_quote=10, ships_by="air"), 137.50),
    # not OVR: item 375.00 + ship 37.50 = CIF 412.50 > 400 -> GST 37.125 -> 37.13 on whole CIF; total 449.63
    ("air import gst", P(source="other_sg", currency="USD", price=300, shipping_quote=30, ships_by="air"), 449.63),
    # air by weight: 3 kg * S$8 = 24.00; item 95.00 -> GST 8.55 at checkout; total 127.55
    ("air by weight", P(source="taobao", currency="CNY", price=500, weight_kg=3, ships_by="air"), 127.55),
    # rounding: item 1.5*0.19 = 0.285 -> 0.29; CIF 0.29; GST 0.0261 -> 0.03; total 0.32
    ("rounding half up", P(source="taobao", currency="CNY", price=1.5, shipping_quote=0, ships_by="sea"), 0.32),
]


@pytest.mark.parametrize("name,product,expected", CASES, ids=[c[0] for c in CASES])
def test_landed_cost_exact(name, product, expected, fx):
    q = landed_cost(product, load_rules(), fx)
    assert q.total_sgd == pytest.approx(expected, abs=1e-9), q.lines
    assert round(sum(line.amount_sgd for line in q.lines), 2) == pytest.approx(q.total_sgd)   # lines add up


def test_there_are_30_cases():
    assert len(CASES) == 30


def test_gst_basis_and_flags(fx):
    r = load_rules()
    assert landed_cost(P(price=10), r, fx).gst_basis == "included"
    assert landed_cost(P(source="taobao", currency="CNY", price=100, shipping_quote=0, ships_by="sea"), r, fx).gst_basis == "import_cif"
    assert landed_cost(P(source="taobao", currency="CNY", price=100, shipping_quote=0, ships_by="air"), r, fx).gst_basis == "checkout_ovr"
    q = landed_cost(P(source="taobao", currency="CNY", price=3000, item_type="sofa"), r, fx)
    assert q.estimated and any("typical" in a for a in q.assumptions)              # estimates are disclosed
    assert landed_cost(P(source="taobao", currency="CNY", price=1, synthetic=True, shipping_quote=0), r, fx).synthetic


def test_missing_price_raises(fx):
    with pytest.raises(ValueError):
        landed_cost(P(price=None), load_rules(), fx)


def test_basket_charges_each_retailers_delivery_once(fx):
    r = load_rules()
    a = landed_cost(P(source="castlery", price=300), r, fx)       # alone: 300 + 19
    b = landed_cost(P(source="castlery", price=250), r, fx)       # alone: 250 + 19
    c = landed_cost(P(price=49.90, weight_kg=2, dims="50 cm"), r, fx)          # ikea parcel 10
    d = landed_cost(P(price=639, weight_kg=60, dims="200 cm", item_type="sofa"), r, fx)   # ikea home 35
    out = basket_total([("castlery", a), ("castlery", b), ("ikea_sg", c), ("ikea_sg", d)], r)
    # castlery order 550 -> free; ikea one home delivery 35: 550 + 49.90 + 639 + 35
    assert out["total_sgd"] == pytest.approx(1273.90)
    assert out["delivery_by_source"] == {"castlery": 0.0, "ikea_sg": 35.0}

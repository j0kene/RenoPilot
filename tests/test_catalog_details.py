import csv
import sys
from pathlib import Path

from renopilot.pricing.comparables import judge_comparables
from renopilot.schemas import Product
from tests.test_pricing_node import FakeJudge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "scripts"))
from collect_sg_catalog import apply_overrides, product_details  # noqa: E402


def test_product_jsonld_reads_only_explicit_fields():
    html = '''<script type="application/ld+json">{
      "@context":"https://schema.org","@graph":[{"@type":"Product",
      "material":"Solid oak","additionalProperty":[
      {"name":"Dimensions","value":"120 x 60 x 42 cm"}] }]}</script>'''
    assert product_details(html) == {"material": "Solid oak", "dims": "120 x 60 x 42 cm"}
    assert product_details('<script type="application/ld+json">{"@type":"Product"}</script>') == {
        "material": "", "dims": ""}


def test_related_product_facts_are_not_misattributed_and_expected_url_wins():
    html = '''<script type="application/ld+json">{"@graph":[
      {"@type":"Product","url":"https://shop.example/related","material":"Silk",
       "width":{"value":999,"unitCode":"CMT"}},
      {"@type":"Product","url":"https://shop.example/main/","material":["Oak","Steel"],
       "width":{"value":80,"unitText":"centimeters"},"depth":{"value":500,"unitCode":"MMT"},
       "height":"28 inches"}]}</script>'''
    expected = "https://shop.example/main"
    assert product_details(html, expected) == {
        "material": "Oak, Steel", "dims": "W 80 cm D 50 cm H 71.12 cm"}
    assert product_details(html, "https://shop.example/missing") == {"material": "", "dims": ""}
    assert product_details(html) == {"material": "", "dims": ""}


def test_override_requires_source_and_iso_checked_date(tmp_path):
    path = tmp_path / "overrides.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "dims", "material", "source_url", "checked_at"])
        writer.writeheader()
        writer.writerow({"id": "cast:one", "dims": "120x60 cm", "material": "Oak",
                         "source_url": "https://shop.example/item", "checked_at": "2026-10-30"})
        writer.writerow({"id": "cast:bad", "dims": "made up", "source_url": "https://shop.example/x",
                         "checked_at": "yesterday"})
    rows = [{"id": "cast:one", "dims": "", "material": ""}, {"id": "cast:bad", "dims": ""}]
    assert apply_overrides(rows, path) == 2
    assert rows == [{"id": "cast:one", "dims": "120x60 cm", "material": "Oak"},
                    {"id": "cast:bad", "dims": ""}]


def test_unknown_size_is_explicit_in_comparable_reason():
    p = Product(id="ikea:one", source="ikea_sg", item_type="sofa", title="linen sofa")
    c = Product(id="cast:one", source="castlery", item_type="sofa", title="linen sofa")
    judge = FakeJudge([{"shortlisted_id": p.id, "candidate_id": c.id, "comparable": True, "why": "similar style"}])
    out = judge_comparables([(p, [c])], judge)
    assert "size unknown (not physically verified)" in out[p.id][0][1]
    assert "material unknown" in out[p.id][0][1]

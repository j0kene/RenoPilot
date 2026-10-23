"""Title typing: every case here was a real mistype found while compiling the catalog."""
import pytest

from renopilot.catalog.taxonomy import EXCLUDE, classify


@pytest.mark.parametrize("title,expected", [
    ("LACK TV bench", "tv_console"),                                   # was typed bench
    ("RULLERUM 2-seat sofa with electric recliner", "sofa"),          # was typed armchair
    ("Cedric Table Lamp", "table_lamp"),                               # Castlery lists it under floor lamps
    ("KNARREVIK Bedside table", "bedside_table"),                      # IKEA returns it for "side table"
    ("Iris Floor Lamp", "floor_lamp"),
    ("Owen II 3 Seater Sofa with Ottoman", "sofa"),
    ("POANG Armchair", "armchair"),
    ("MALM Chest of 4 drawers", "dresser"),
    ("VALTORP Bench", "bench"),
    ("Round oak coffee table", "coffee_table"),
])
def test_classify(title, expected):
    assert classify(title) == expected


@pytest.mark.parametrize("title", ["Stone & Beam Stone Brown Swatch, 25020039-01", "Sofa slipcover, grey",
                                   "Replacement legs for coffee table"])
def test_accessories_are_excluded(title):
    assert EXCLUDE.search(title)

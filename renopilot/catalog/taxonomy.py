"""Title -> item type, shared by every catalog source.

Retailer categories are leaky (a "floor lamps" page lists table lamps; a "side table"
search returns bedside tables), so every product is typed from its own title by ordered
keyword rules. The first matching rule wins: specific phrases come before generic ones.
"""
from __future__ import annotations

import re

EXCLUDE = re.compile(r"\b(swatch|cover|slipcover|replacement|parts?|pad|sheets?|kit|hardware|"
                     r"cushion|pillow|bulb|shade only|cable|adapter|sticker|decal|mat for|"
                     r"protector|leg[s]? only|caster|glides?|organi[sz]er|basket|curtain|"
                     r"outdoor|patio|garden|kids?|baby|toddler|dollhouse|miniature)\b", re.I)

# Ordered: the first matching rule wins, so specific phrases come before generic ones.
RULES: list[tuple[str, str]] = [
    (r"\bcoffee table|cocktail table", "coffee_table"),
    (r"\b(night ?stand|bedside table|bedside cabinet)", "bedside_table"),
    (r"\b(side|end|accent) table|\bc-table|nesting tables?", "side_table"),
    (r"\b(tv|media|entertainment) (stand|console|unit|cabinet|center|centre|bench|storage)", "tv_console"),
    (r"\bdining (table|set)|kitchen table|\bdinette", "dining_table"),
    (r"\b(dining|side|kitchen) chairs?\b", "dining_chair"),
    (r"\b(office|desk|task|computer|gaming|executive) chair", "office_chair"),
    (r"\b(bar|counter)(-| )?(height )?stools?\b", "bar_stool"),
    (r"\b(sofa|couch|loveseat|sectional|settee|futon|chesterfield)\b", "sofa"),   # before armchair: "sofa with recliner"
    (r"\b(accent|arm|lounge|club|wingback|barrel|slipper|recliner|reclining) ?chair|\barmchair|\brecliner", "armchair"),
    (r"\b(writing|computer|standing|office) desk|\bdesk\b", "desk"),
    (r"\b(platform bed|bed frame|bedframe|upholstered bed|daybed|\bbed\b)", "bed_frame"),
    (r"\b(dresser|chest of (\d+ )?drawers|tallboy)", "dresser"),
    (r"\b(wardrobe|armoire|clothes rack|garment rack)", "wardrobe"),
    (r"\b(bookcase|bookshelf|book shelf|shelving unit|etagere|ladder shelf)", "bookshelf"),
    (r"\bfloor lamp|standing lamp|arc lamp|torchiere", "floor_lamp"),
    (r"\b(table|desk|bedside) lamp", "table_lamp"),
    (r"\bpendant|chandelier|ceiling light", "pendant_light"),
    (r"\b(area )?rug\b|\brunner rug", "rug"),
    (r"\bottoman|pouf|footstool", "ottoman"),
    (r"\bbench\b", "bench"),
    (r"\bmirror\b", "mirror"),
    (r"\blamp\b", "table_lamp"),
]
RULES_RE = [(re.compile(p, re.I), t) for p, t in RULES]


def classify(title: str) -> str | None:
    for rx, item_type in RULES_RE:
        if rx.search(title):
            return item_type
    return None

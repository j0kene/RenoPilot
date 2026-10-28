"""Fixed vocabularies. The VLM may only answer in these words, which does two jobs:
attribute F1 against the human labels is computable, and the model cannot invent a
style or an item type the catalog and the rule table have never heard of.

Changing a list here changes the label sheet, so agree changes with the team.
"""
from __future__ import annotations

STYLE_TAGS = (
    "japandi", "scandinavian", "muji", "minimalist", "modern_contemporary", "modern_luxe",
    "industrial", "mid_century_modern", "wabi_sabi", "bohemian", "coastal", "tropical_resort",
    "vintage_retro", "classic_traditional",
)

COLOURS = (
    "white", "off_white_cream", "beige", "greige", "light_grey", "charcoal", "black",
    "light_wood", "mid_wood", "dark_wood", "tan_camel", "brown", "terracotta_rust",
    "mustard_yellow", "sage_green", "olive_green", "forest_green", "navy", "dusty_blue",
    "blush_pink", "burgundy", "brass_gold",
)

MATERIALS = (
    "light_wood", "mid_wood", "dark_wood", "rattan_cane", "bamboo", "linen_cotton", "boucle",
    "velvet", "leather", "wool", "jute_seagrass", "black_metal", "brass_gold_metal",
    "chrome_steel", "glass", "marble", "terrazzo", "sintered_stone", "concrete_microcement",
    "ceramic", "plastic", "paper_fabric_shade",
)

# Every item type has a work type. The compliance guard keys off the work type:
# loose furniture passes straight through; finishes and built-ins are checked.
WORK_TYPE_OF = {
    # loose furniture
    "sofa": "loose_furniture", "armchair": "loose_furniture", "coffee_table": "loose_furniture",
    "side_table": "loose_furniture", "tv_console": "loose_furniture",
    "dining_table": "loose_furniture", "dining_chair": "loose_furniture",
    "bar_stool": "loose_furniture", "bench": "loose_furniture", "ottoman": "loose_furniture",
    "bed_frame": "loose_furniture", "bedside_table": "loose_furniture",
    "dresser": "loose_furniture", "wardrobe": "loose_furniture", "bookshelf": "loose_furniture",
    "desk": "loose_furniture", "office_chair": "loose_furniture", "rug": "loose_furniture",
    "floor_lamp": "loose_furniture", "table_lamp": "loose_furniture",
    "pendant_light": "loose_furniture", "mirror": "loose_furniture",
    # finishes (HDB rules apply)
    "floor_tiles": "finish", "wall_tiles": "finish", "vinyl_flooring": "finish",
    "timber_flooring": "finish", "wall_paint": "finish", "bathroom_finishes": "finish",
    # built-ins and works (HDB rules apply)
    "built_in_wardrobe": "built_in", "kitchen_cabinets": "built_in",
    "carpentry_feature_wall": "built_in", "false_ceiling": "built_in",
    "partition_wall_change": "built_in",
}
ITEM_TYPES = tuple(WORK_TYPE_OF)
LOOSE_FURNITURE = tuple(t for t, w in WORK_TYPE_OF.items() if w == "loose_furniture")

ROOMS = ("living", "dining", "bedroom", "kitchen", "bathroom", "study", "household_shelter", "mixed")

FAILURE_TAGS = (
    "style_misread", "retrieval_miss", "over_budget", "wrong_tool", "tool_error",
    "unsupported_claim", "rule_violation", "injection_followed", "over_escalation",
    "under_escalation", "format_error",
)

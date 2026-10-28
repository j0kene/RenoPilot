# Labelling guide: style eval boards

## Why we label
The Style agent reads a mood board and returns a style profile (style tags, palette, materials, item types).
Your labels are the **ground truth** it is scored against (attribute F1, target 0.7). Two people label every
board **independently**; how much you agree with each other is reported as the ceiling (the agent can't be
expected to agree with one person more than two people agree with each other).

## The rules (these protect the eval)
1. **Label independently.** Don't discuss boards or compare sheets until both are finished.
2. **Never look at the agent's output for a board before labelling it.** That would be grading the agent
   against itself (the course calls this contamination).
3. **Label what you see, not the folder name.** `japandi_01` is the Unsplash search query, not the answer.
   If a "japandi" board looks Scandinavian to you, write `scandinavian`.
4. **Use only the words below** (copy them exactly). The eval script warns about anything outside the list.

## How
- Labeller A fills `style_labels_A.csv`, labeller B fills `style_labels_B.csv`. One row per board, already listed.
- Open the board folder `data/boards/<board_id>/` and look at **all** its photos together (VS Code's
  explorer preview or Windows Photos is fine). Label the look the photos **share**, the way you would
  describe the board to a friend who wants to recreate it.
- Separate several values with `;` and no spaces needed, e.g. `japandi;minimalist`.
- Roughly 2-3 minutes per board: 27 boards is about 1-1.5 hours per labeller.

| column | what to write | how many | tips |
|---|---|---|---|
| `style_tags` | the overall look, most dominant first | 1-3 | if you hesitate between two, write both (dominant first) |
| `palette` | colours that define the board | 2-6 | dominant first; include wood tone (`light_wood`) when wood is a main colour |
| `materials` | materials that recur or stand out | 0-6 | only what you can actually see (e.g. `boucle` only if clearly textured) |
| `item_types` | distinct items worth shopping for, clearly visible | 1-12 | count an item if it appears in any photo and is clear enough to shop for; skip tiny decor |
| `notes` | anything odd (optional) | | e.g. "photo 3 is an exterior", "mixed styles" |

### Made-up example (not one of the real boards)
| board_id | style_tags | palette | materials | item_types | notes |
|---|---|---|---|---|---|
| example_99 | `scandinavian;minimalist` | `white;light_wood;light_grey` | `light_wood;linen_cotton;wool` | `sofa;coffee_table;rug;floor_lamp;pendant_light` | |

## Allowed words

**style_tags:** `japandi`, `scandinavian`, `muji`, `minimalist`, `modern_contemporary`, `modern_luxe`, `industrial`, `mid_century_modern`, `wabi_sabi`, `bohemian`, `coastal`, `tropical_resort`, `vintage_retro`, `classic_traditional`

**palette:** `white`, `off_white_cream`, `beige`, `greige`, `light_grey`, `charcoal`, `black`, `light_wood`, `mid_wood`, `dark_wood`, `tan_camel`, `brown`, `terracotta_rust`, `mustard_yellow`, `sage_green`, `olive_green`, `forest_green`, `navy`, `dusty_blue`, `blush_pink`, `burgundy`, `brass_gold`

**materials:** `light_wood`, `mid_wood`, `dark_wood`, `rattan_cane`, `bamboo`, `linen_cotton`, `boucle`, `velvet`, `leather`, `wool`, `jute_seagrass`, `black_metal`, `brass_gold_metal`, `chrome_steel`, `glass`, `marble`, `terrazzo`, `sintered_stone`, `concrete_microcement`, `ceramic`, `plastic`, `paper_fabric_shade`

**item_types, loose furniture:** `sofa`, `armchair`, `coffee_table`, `side_table`, `tv_console`, `dining_table`, `dining_chair`, `bar_stool`, `bench`, `ottoman`, `bed_frame`, `bedside_table`, `dresser`, `wardrobe`, `bookshelf`, `desk`, `office_chair`, `rug`, `floor_lamp`, `table_lamp`, `pendant_light`, `mirror`

**item_types, finishes / built-ins** (include when prominent, e.g. a patterned tile floor or a feature wall,
because these are what the compliance guard checks): `floor_tiles`, `wall_tiles`, `vinyl_flooring`, `timber_flooring`, `wall_paint`, `bathroom_finishes`, `built_in_wardrobe`, `kitchen_cabinets`, `carpentry_feature_wall`, `false_ceiling`, `partition_wall_change`

## After both sheets are done
Run `python -m uv run python evals/run_style.py --n-runs 5 --baseline`. It prints your inter-rater agreement first,
then the agent's F1 against both of you, plus the no-schema baseline. Then (and only then) compare sheets and
discuss disagreements; don't edit labels after seeing the agent's results.

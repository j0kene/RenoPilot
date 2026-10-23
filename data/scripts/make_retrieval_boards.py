"""Build the retrieval eval: boards made of KNOWN catalog products, so ground truth needs no
human labels.

    uv run python data/scripts/make_retrieval_boards.py [--n 50]

Each case is a "room" of 3-4 ABO products of different item types. The board shows each
product's ALTERNATE image; the index holds the MAIN image. So the system can never win by
finding the identical picture: it has to recognise the product from another angle or shot.

Writes evals/cases/retrieval.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from renopilot import config  # noqa: E402

ROOMS = {
    "living": ["sofa", "coffee_table", "armchair", "rug", "side_table", "floor_lamp", "tv_console", "ottoman"],
    "bedroom": ["bed_frame", "bedside_table", "dresser", "table_lamp", "mirror", "wardrobe", "bench"],
    "dining": ["dining_table", "dining_chair", "pendant_light", "bar_stool", "bench", "rug"],
    "study": ["desk", "office_chair", "bookshelf", "table_lamp", "floor_lamp"],
}
ROOM_SHARE = {"living": 0.4, "bedroom": 0.25, "dining": 0.2, "study": 0.15}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    rng = random.Random(621)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for line in (config.INDEX / "abo_products.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        if p.get("alt_image_paths"):
            by_type[p["item_type"]].append(p)
    for items in by_type.values():
        rng.shuffle(items)
    used: set[str] = set()

    def take(t: str) -> dict | None:
        for p in by_type.get(t, []):
            if p["id"] not in used:
                used.add(p["id"])
                return p
        return None

    cases = []
    for room, share in ROOM_SHARE.items():
        for _ in range(round(args.n * share)):
            types = [t for t in ROOMS[room] if by_type.get(t)]
            pick = rng.sample(types, k=min(len(types), rng.choice([3, 4])))
            prods = [p for p in (take(t) for t in pick) if p]
            if len(prods) < 3:
                continue
            cases.append({
                "id": f"r{len(cases) + 1:02d}", "input": {"board": [p["alt_image_paths"][0] for p in prods],
                                                         "room": room},
                "expected": [{"product_id": p["id"], "item_type": p["item_type"]} for p in prods],
                "tags": [room, f"n{len(prods)}"],
            })
    out = config.ROOT / "evals" / "cases" / "retrieval.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases ({sum(len(c['expected']) for c in cases)} targets) to {out}")


if __name__ == "__main__":
    main()

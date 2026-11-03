"""Paths, model-per-role settings and the numeric limits the graph enforces in code."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"              # downloads, never committed
CACHE = DATA / "cache"          # regenerable (embeddings, VLM dev cache)
IMAGES = DATA / "images"        # product images, cached locally for embedding only
INDEX = DATA / "index"          # SQLite catalog + .npy embeddings
BOARDS = DATA / "boards"
RUNS = ROOT / "runs"            # JSONL traces

# --------------------------------------------------------------------------- models
# One backend for every role while building and evaluating; swap at the end.
BACKEND = os.environ.get("RENOPILOT_BACKEND", "claude_cli")      # claude_cli | openai
EFFORT = os.environ.get("RENOPILOT_EFFORT", "")   # claude_cli: low|medium|high|... ; "" = model default

MODELS = {
    "claude_cli": {
        "vision":  os.environ.get("RENOPILOT_MODEL_VISION", "sonnet"),
        "planner": os.environ.get("RENOPILOT_MODEL_PLANNER", "sonnet"),
        "rerank":  os.environ.get("RENOPILOT_MODEL_RERANK", "haiku"),
        "shopper": os.environ.get("RENOPILOT_MODEL_SHOPPER", "sonnet"),
        "intake":  os.environ.get("RENOPILOT_MODEL_INTAKE", "haiku"),
    },
    "openai": {        # placeholder names: confirm current models and prices before switching
        "vision":  os.environ.get("RENOPILOT_MODEL_VISION", "gpt-4.1-mini"),
        "planner": os.environ.get("RENOPILOT_MODEL_PLANNER", "gpt-4.1-mini"),
        "rerank":  os.environ.get("RENOPILOT_MODEL_RERANK", "gpt-4.1-mini"),
        "shopper": os.environ.get("RENOPILOT_MODEL_SHOPPER", "gpt-4.1-mini"),
        "intake":  os.environ.get("RENOPILOT_MODEL_INTAKE", "gpt-4.1-mini"),
    },
}

CLI_TIMEOUT_S = int(os.environ.get("RENOPILOT_CLI_TIMEOUT", "480"))   # a no-tools plan took 164 s and timed out at 240 s

# --------------------------------------------------------------------------- retrieval
EMBED_MODEL = os.environ.get("RENOPILOT_EMBED_MODEL", "ViT-B-16-SigLIP:webli")
WIDE_K = 50            # candidates kept after fusion, before the rerank
TOP_K = 5              # shortlist size per item type
MIN_IN_BUDGET = 3      # fewer in-budget matches than this -> ask the user (autonomy boundary)
RRF_K = 60             # reciprocal-rank-fusion constant
# How the shortlist picks from the catalog: "react" (a shopper subagent per item type, calling the
# catalog tools itself: nodes/shopper.py) or "rerank" (one fixed search, one model call picks).
PICKER = os.environ.get("RENOPILOT_PICKER", "react")

# Which catalog the graph shortlists from. ABO has no prices, so the budget-aware demo uses
# the SG retailers. RENOPILOT_SOURCES: "sg" (default) | "all" | comma-separated sources.
SG_SOURCES = ("ikea_sg", "castlery", "taobao", "other_sg")
_SRC = os.environ.get("RENOPILOT_SOURCES", "sg")
SHORTLIST_SOURCES = SG_SOURCES if _SRC == "sg" else None if _SRC == "all" else tuple(_SRC.split(","))

# --------------------------------------------------------------------------- graph limits
MAX_PLANNER_STEPS = 8
MIN_BOARD_IMAGES = 3
MAX_BOARD_IMAGES = 12
VLM_IMAGE_MAX_SIDE = 768    # downscale before sending (cost)

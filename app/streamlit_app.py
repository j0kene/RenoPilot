"""Temporary demo UI: mood board -> priced shortlist, with the master agent, the Style agent's
shopper subagents, Pricing, the HDB Compliance guard and the two human gates (question + approval).

    python -m uv run --group ui streamlit run app/streamlit_app.py

Two modes. Live: the real models and catalog (needs the catalog index and the model backend).
Demo: runs offline with no setup (renopilot/demo.py): the same graph, gates, tools, pricing and
tracing over the real catalog snapshot, with the models played by code.
"""
from __future__ import annotations

import io
import shutil
import sys
import time
from datetime import date
from pathlib import Path
from uuid import uuid4

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from renopilot import config, trace  # noqa: E402
from renopilot.basket import offers_for, selection, summary, work_totals  # noqa: E402
from renopilot.catalog.embed import slug  # noqa: E402
from renopilot.compliance import PLAIN  # noqa: E402
from renopilot.demo import DEMO_BOARD, demo_nodes  # noqa: E402
from renopilot.graph import build_graph  # noqa: E402
from renopilot.nodes.guard import blocked, request_from  # noqa: E402
from renopilot.nodes.approval import apply_routes  # noqa: E402
from renopilot.nodes.respond import shopping_list_csv  # noqa: E402
from renopilot.schemas import ProjectProfile, ItemSpec, WorkEstimate  # noqa: E402
from renopilot.tools.budget import fmt_sgd  # noqa: E402
from renopilot.vocab import LOOSE_FURNITURE, WORK_TYPE_OF, ITEM_TYPES, ROOMS  # noqa: E402

st.set_page_config(page_title="RenoPilot demo", page_icon="🛋️", layout="wide")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
NODE_LABEL = {"intake": "Checking the board", "planner": "Planning the next step",
              "style_extract": "Reading the mood board", "shortlist": "Searching the catalog",
              "compliance_guard": "Checking HDB rules", "ask_user": "Question for you",
              "approval": "Waiting for your approval", "respond": "Writing the reply"}
DEMO, LIVE = "Demo (offline)", "Live"

# The graph as a diagram (renopilot/graph.py): node -> (x, y, title, role). Lit up live by run().
BOX = {"ask_user": (10, 16, "Ask you", "human gate"), "style_extract": (160, 16, "Style extract", "Style agent"),
       "shortlist": (310, 16, "Shortlist", "shopper subagents"), "pricing": (460, 16, "Pricing", "landed SGD"),
       "intake": (10, 140, "Intake", "checks the board"), "planner": (190, 140, "Planner", "master agent"),
       "compliance_guard": (420, 140, "Compliance", "HDB rules"), "approval": (580, 140, "Approval", "human gate"),
       "respond": (740, 140, "Respond", "final reply")}
W, H = 126, 50
COLOR = {"running": "#3b82f6", "done": "#16a34a", "waiting": "#d97706", "error": "#dc2626"}
FLOW_CSS = """<style>
.flow{color:#31333f}@media (prefers-color-scheme:dark){.flow{color:#fafafa}}
.flow text{fill:currentColor;font:600 13px sans-serif;text-anchor:middle}.flow text.r{font:400 11px sans-serif;opacity:.7}
.flow .e{fill:none;stroke:currentColor;stroke-opacity:.25;stroke-width:1.5}.flow .e.on{stroke-opacity:.8}
.flow .running{animation:pulse 1.2s ease-in-out infinite}@keyframes pulse{50%{stroke-width:4;fill-opacity:.25}}
</style>"""


def flow_svg(flow: dict) -> str:
    """The graph as an SVG: idle grey, running blue (pulsing), done green, waiting on you amber."""
    def at(n, side):                                     # "t48" = top edge, 48px in
        x, y = BOX[n][:2]
        dx = float(side[1:] or W / 2)
        return {"l": (x, y + H / 2), "r": (x + W, y + H / 2), "t": (x + dx, y), "b": (x + dx, y + H)}[side[0]]

    edges = [("intake", "r", "planner", "l"), ("planner", "r", "compliance_guard", "l"),
             ("compliance_guard", "r", "approval", "l"), ("approval", "r", "respond", "l"),
             ("planner", "t20", "ask_user", "b"), ("planner", "t48", "style_extract", "b"),
             ("planner", "t78", "shortlist", "b"), ("shortlist", "r", "pricing", "l"), ("pricing", "b", "planner", "t106")]
    svg = []
    for a, sa, b, sb in edges:
        (x1, y1), (x2, y2) = at(a, sa), at(b, sb)
        back = 'marker-start="url(#h)"' if b in ("ask_user", "style_extract") else ""   # these return to the planner
        on = " on" if a in flow and b in flow else ""
        svg.append(f'<path class="e{on}" d="M{x1},{y1} L{x2},{y2}" marker-end="url(#h)" {back}/>')
    on = " on" if flow.get("approval", {}).get("n", 0) > 1 else ""
    svg.append(f'<path class="e{on}" d="M643,190 C643,232 253,232 253,190" marker-end="url(#h)" stroke-dasharray="4 3"/>'
               '<text class="r" x="448" y="226">rejections</text>')
    for n, (x, y, title, role) in BOX.items():
        f = flow.get(n, {})
        s, c = f.get("state", "idle"), COLOR.get(f.get("state"), "currentColor")
        note = {"running": f.get("note") or "running…", "waiting": "waiting for you", "error": "failed, retrying"}.get(s)
        if s == "done":
            note = f"done{' ×' + str(f['n']) if f['n'] > 1 else ''} · {f['secs']:.1f}s"
        op = ".12" if s != "idle" else ".04"
        svg.append(f'<rect class="{s}" x="{x}" y="{y}" width="{W}" height="{H}" rx="10" fill="{c}" fill-opacity="{op}" '
                   f'stroke="{c}" stroke-opacity="{1 if s != "idle" else .3}" stroke-width="{2 if s != "idle" else 1}"/>'
                   f'<text x="{x + W / 2}" y="{y + 21}">{title}</text><text class="r" x="{x + W / 2}" y="{y + 38}">{note or role}</text>')
    # st.html strips <svg> (DOMPurify html profile), so this goes to st.image with its CSS inside
    return ('<svg xmlns="http://www.w3.org/2000/svg" class="flow" viewBox="0 0 880 240" width="880" height="240">'
            + FLOW_CSS + '<defs><marker id="h" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="currentColor" fill-opacity=".6"/></marker></defs>' + "".join(svg) + "</svg>")


def live_ready() -> str | None:
    """Why live mode cannot run here, or None."""
    if not (config.INDEX / slug(config.EMBED_MODEL) / "ids.txt").exists():
        return "the catalog index is not built (see the README's Data steps)"
    if config.BACKEND == "claude_cli" and not shutil.which("claude"):
        return "the `claude` CLI is not on PATH"
    return None


@st.cache_data
def no_photo() -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (400, 300), (236, 234, 230))
    ImageDraw.Draw(img).text((170, 142), "no photo", fill=(130, 130, 130))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def md(text: str) -> str:
    """Escape $ so prices like S$159 ... S$149 are not rendered as LaTeX."""
    return text.replace("$", r"\$")


def picture(image_path: str | None) -> None:
    """The product photo (cached locally by the data scripts), or a neutral tile without one."""
    path = config.ROOT / image_path if image_path else None
    st.image(str(path) if path and path.exists() else no_photo(), width="stretch")


def hdb_routes(c: dict) -> None:
    """A finish's HDB routes (the guard's verdict per way of doing it): the owner picks one;
    blocked routes are shown, never offered."""
    by_id = {r["rule_id"]: r for r in c["routes"]}
    ok = [r["rule_id"] for r in c["routes"] if r["decision"] in ("allow", "allow_with_permit")]
    if ok and c["work_method"] == "unspecified":
        route_key = f"route_{c['key']}"
        if route_key in st.session_state and st.session_state[route_key] not in ok:
            del st.session_state[route_key]
        st.radio("How will this be done?", ok, key=f"route_{c['key']}",
                 index=ok.index(c["rule_id"]) if c["rule_id"] in ok else None,
                 format_func=lambda i: f"{by_id[i]['route']} ({PLAIN[by_id[i]['decision']]})",
                 captions=[md(by_id[i]["reason"]) for i in ok])
    if c["work_method"] != "unspecified" or c["verdict"] == "escalate":
        st.warning(md(f"{PLAIN[c['verdict']]}: {c['verdict_reason']}"))
    for r in c["routes"]:
        if r["decision"] == "block":
            st.caption(md(f":red[Not allowed] · {r['route']}: {r['reason']}"))
    with st.expander("HDB conditions and sources"):
        for r in c["routes"]:
            st.markdown(md(f"**{r['route']}** · {PLAIN[r['decision']]} · ") + f"[{r['rule_id']}]({r['citation']})")
            for cond in r["conditions"]:
                st.caption(md(f"- {cond}"))


def show_blocked(items: list[dict]) -> None:
    """What the HDB guard blocked: shown with the reason and the legal alternative, never dropped."""
    for b in items:
        alt = f" Instead: {b['alternative']}" if b["alternative"] else ""
        st.error(md(f"**Not allowed under HDB rules: {b['title'] or b['item_type'].replace('_', ' ')}.** "
                    f"{b['reason']}{alt}") + f" [{b['rule_id']}]({b['citation']})")


def work_cost_summary(project, suggestions, furniture_total):
    costs = work_totals(project, suggestions)
    st.metric("Priced works subtotal", fmt_sgd(costs["subtotal_sgd"]))
    if costs["complete"] and costs["remaining_sgd"] is not None:
        st.caption(md(f"Works budget remaining: {fmt_sgd(costs['remaining_sgd'])}"))
    label = "Combined project estimate" if costs["complete"] else "Combined priced subtotal · incomplete"
    st.metric(label, fmt_sgd(furniture_total + costs["subtotal_sgd"]))
    if not costs["complete"]:
        st.warning("Work estimates are incomplete: " + "; ".join(costs["missing_requests"] + costs["unpriced_rows"]) +
                   ". Enter a rate, source and quote date for each cost line.")


def edit_requests(profile):
    with st.expander("Edit requested items and work facts"):
        specs = {i.request_id or request_from(i).key: i for i in profile.items}
        key = st.selectbox("Request to edit", list(specs),
            format_func=lambda k: f"{specs[k].room_label or specs[k].room or 'Room unknown'} · {specs[k].description}")
        spec = specs[key]
        with st.form(f"edit_request_{key}"):
            description = st.text_input("Request description", spec.description)
            room = st.selectbox("Room", [None, *ROOMS], index=[None, *ROOMS].index(spec.room),
                                format_func=lambda r: "Unknown" if r is None else r.replace("_", " "))
            room_label = st.text_input("Room name", spec.room_label, placeholder="e.g. Bedroom 2")
            quantity = st.number_input("Quantity", min_value=1, max_value=100, value=spec.quantity, step=1)
            patch = {"description": description, "room": room, "room_label": room_label, "quantity": quantity}
            if WORK_TYPE_OF[spec.item_type] == "loose_furniture":
                for axis in ("width", "depth", "height"):
                    field = f"max_{axis}_cm"
                    patch[field] = st.number_input(f"Maximum {axis} (cm, optional)", min_value=0.1,
                                                  value=getattr(spec, field), step=1.0)
            else:
                methods = ["unspecified", "overlay", "replace", "raise", "demolish", "erect"]
                patch["work_method"] = st.selectbox("Requested method", methods, index=methods.index(spec.work_method))
                if spec.item_type == "partition_wall_change":
                    patch["structural"] = st.selectbox("Structural wall?", [None, True, False],
                        index=[None, True, False].index(spec.structural),
                        format_func=lambda x: "Unknown" if x is None else "Yes" if x else "Confirmed non-structural")
                facts = spec.work_details.model_dump()
                if spec.item_type in ("floor_tiles", "wall_tiles", "bathroom_finishes", "timber_flooring"):
                    facts["overlay_thickness_mm"] = st.number_input("New overlay including adhesive (mm)",
                        min_value=0.0, value=facts["overlay_thickness_mm"], step=1.0)
                    if spec.item_type != "wall_tiles":
                        facts["existing_floor_layers"] = st.number_input("Existing floor-finish layers", min_value=0,
                            value=facts["existing_floor_layers"], step=1)
                if spec.item_type == "false_ceiling":
                    facts["ceiling_clearance_mm"] = st.number_input("Clearance above finished floor (mm)",
                        min_value=0.0, value=facts["ceiling_clearance_mm"], step=10.0)
                    for field, label in (("non_combustible", "Non-combustible material?"),
                                         ("gas_pipe_enclosed", "Will a gas pipe be enclosed?")):
                        facts[field] = st.selectbox(label, [None, True, False], index=[None, True, False].index(facts[field]),
                            format_func=lambda x: "Unknown" if x is None else "Yes" if x else "No")
                if spec.item_type == "partition_wall_change":
                    materials = [None, "hollow_block", "glass_block", "gypsum", "plywood", "plastic", "asbestos", "other"]
                    facts["partition_material"] = st.selectbox("Partition material", materials,
                        index=materials.index(facts["partition_material"]),
                        format_func=lambda m: "Unknown" if m is None else m.replace("_", " "))
                patch["work_details"] = facts
            if st.form_submit_button("Save request and recheck"):
                try:
                    ItemSpec.model_validate(spec.model_dump() | patch)
                except ValueError as e:
                    st.error(str(e))
                else:
                    run(Command(resume={"work_updates": {key: patch}}))
                    st.rerun()
    with st.expander("Add a missing request"):
        with st.form("add_request"):
            item_type = st.selectbox("Item or work type", ITEM_TYPES)
            description = st.text_input("What is needed?")
            room = st.selectbox("Room for the new request", [None, *ROOMS],
                format_func=lambda r: "Unknown" if r is None else r.replace("_", " "))
            if st.form_submit_button("Add request"):
                if description.strip():
                    run(Command(resume={"add_requests": [{"item_type": item_type,
                        "description": description.strip(), "room": room}]}))
                    st.rerun()
                else:
                    st.error("Describe the item or work you need.")


def edit_work_costs(project, suggestions):
    with st.expander("Works budget and estimates"):
        st.caption("Enter owner or contractor estimates, including labour, installation and contingency as separate lines. "
                   "Each quoted work needs a source and date. Unpriced work remains incomplete.")
        rows = [r.model_dump() for r in work_totals(project, suggestions)["rows"]]
        for s in suggestions:
            if not s.product_id and s.status != "rejected" and not any(r["request_id"] == s.key for r in rows):
                rows.append(WorkEstimate(request_id=s.key, description=s.title).model_dump())
        if not rows:
            rows = [WorkEstimate(description="").model_dump()]
        with st.form("work_costs"):
            budget = st.number_input("Works budget (SGD, optional)", min_value=0.0, value=project.works_budget, step=100.0)
            edited = st.data_editor(rows, num_rows="dynamic", hide_index=True, disabled=["request_id"], key="work_estimate_rows",
                column_config={"unit_rate_sgd": st.column_config.NumberColumn("Rate SGD", min_value=0),
                    "quantity": st.column_config.NumberColumn("Quantity", min_value=0.01),
                    "checked_at": st.column_config.DateColumn("Quote date"),
                    "request_id": st.column_config.TextColumn("Work reference"),
                    "source": st.column_config.TextColumn("Quote source")})
            if st.form_submit_button("Save work estimates"):
                try:
                    records = edited.to_dict("records") if hasattr(edited, "to_dict") else edited
                    estimates = [WorkEstimate.model_validate({k: v for k, v in r.items() if v is not None}).model_dump(mode="json") for r in records
                                 if str(r.get("description") or "").strip()]
                except (ValueError, TypeError) as e:
                    st.error(str(e))
                else:
                    run(Command(resume={"works_budget": budget, "work_estimates": estimates}))
                    st.rerun()


# --------------------------------------------------------------------------- session
def reset(planner: str, mode: str) -> None:
    st.session_state.planner, st.session_state.mode = planner, mode
    st.session_state.graph = build_graph(planner=planner, nodes=demo_nodes() if mode == DEMO else None)
    st.session_state.cfg = trace.run_config("renopilot-ui", recursion_limit=80,
                                            tags=["ui", "demo" if mode == DEMO else f"planner:{planner}"])
    st.session_state.started = False
    st.session_state.log = []
    st.session_state.flow = {}
    st.session_state.usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


if "graph" not in st.session_state:
    reset("llm", DEMO)
st.session_state.setdefault("flow", {})                  # sessions started before the diagram existed


def run(payload) -> None:
    """Stream the graph until it finishes or parks at an interrupt, lighting up each node as it runs."""
    g, cfg, flow = st.session_state.graph, st.session_state.cfg, st.session_state.flow
    if isinstance(payload, Command) and isinstance(payload.resume, dict) and payload.resume.keys() & {"board_images", "notes", "brief"}:
        for key in [k for k in st.session_state if k.startswith(("pick_", "rej_", "why_", "route_"))]:
            del st.session_state[key]
    with st.status("RenoPilot is working...", expanded=True) as status, trace.tracing() as t:
        t0 = time.perf_counter()
        for ns, kind, chunk in g.stream(payload, cfg, stream_mode=["tasks", "updates"], subgraphs=True):
            if ns:                                       # inside a shopper subagent: count its tool calls
                if kind == "updates" and "tools" in chunk:
                    f = flow.setdefault("shortlist", {"n": 0, "secs": 0.0})
                    f["calls"] = f.get("calls", 0) + 1
                    f["note"] = f"{f['calls']} tool calls · {chunk['tools']['messages'][-1].name}"
                    diagram.image(flow_svg(flow))
                continue
            if kind == "tasks":
                f = flow.setdefault(chunk["name"], {"n": 0, "secs": 0.0})
                if "input" in chunk:                     # task started
                    f.update(state="running", t0=time.perf_counter(), note="", calls=0)
                    status.update(label=f"{NODE_LABEL.get(chunk['name'], chunk['name'])}...")
                else:
                    f["state"] = "waiting" if chunk["interrupts"] else "error" if chunk["error"] else "done"
                    f["n"] += f["state"] == "done"
                    f["secs"] += time.perf_counter() - f.get("t0", time.perf_counter())
                diagram.image(flow_svg(flow))
                continue
            for node, delta in chunk.items():
                if node == "__interrupt__":
                    continue
                why = next((d.get("why", "") for d in (delta or {}).get("trace", [])), "") if isinstance(delta, dict) else ""
                line = f"**{NODE_LABEL.get(node, node)}**" + (f": {why}" if why else "")
                st.session_state.log.append(line)
                st.write(line)
        status.update(label=f"Done in {time.perf_counter() - t0:.0f}s", state="complete", expanded=False)
        for k, v in t.totals().items():
            st.session_state.usage[k] += v
        st.session_state.last_trace = str(t.path)


def pending_interrupt():
    snap = st.session_state.graph.get_state(st.session_state.cfg)
    return snap.interrupts[0].value if snap.interrupts else None


# --------------------------------------------------------------------------- sidebar: inputs
with st.sidebar:
    st.header("Demo settings")
    not_live = live_ready()
    mode = st.radio("Mode", [DEMO, LIVE], horizontal=True, disabled=st.session_state.started,
                    index=[DEMO, LIVE].index(st.session_state.mode),
                    help="Offline: one explicitly scripted Japandi example. Live: read your board with the real models."
                         + (f" Live is unavailable: {not_live}." if not_live else ""))
    planner = st.session_state.planner
    if mode == LIVE:
        planner = st.radio("Planner", ["llm", "fixed"], horizontal=True, disabled=st.session_state.started)
        if not_live:
            st.warning(f"Live mode will fail: {not_live}.")
    st.header("1. Mood board")
    boards = sorted(p.name for p in config.BOARDS.iterdir() if p.is_dir()) if config.BOARDS.exists() else []
    if mode == DEMO:
        boards = [DEMO_BOARD.name]
    source = st.radio("Board source", ["Example board"] if mode == DEMO else ["Example board", "Upload images"], horizontal=True,
                      disabled=st.session_state.started)
    paths: list[str] = []
    if source == "Example board" and boards:
        pick = st.selectbox("Board", boards, disabled=st.session_state.started,
                            index=boards.index("japandi_01") if "japandi_01" in boards else 0)
        paths = sorted(str(p) for p in (config.BOARDS / pick).iterdir() if p.suffix.lower() in IMG_EXT)
    elif source == "Upload images":
        files = st.file_uploader("3-12 images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
                                 disabled=st.session_state.started)
        if files:
            up = config.CACHE / "uploads" / st.session_state.cfg["configurable"]["thread_id"]
            up.mkdir(parents=True, exist_ok=True)
            for f in files:
                dest = up / f"{uuid4().hex}_{Path(f.name).name}"
                dest.write_bytes(f.getbuffer())
                paths.append(str(dest))

    st.header("2. Your flat & what you'd like")
    FLATS = ["not sure", "2-room", "3-room", "4-room", "5-room", "executive", "other"]
    flat_type = st.selectbox("Flat type", FLATS, index=0, disabled=st.session_state.started)
    done = st.date_input("Confirmed block completion date", value=None, format="YYYY-MM-DD",
                         min_value=date(1970, 1, 1), max_value=date(date.today().year + 6, 12, 31),
                         disabled=st.session_state.started,
                         help="Use the actual block completion date, which may differ from key collection. "
                              "Leave empty if unknown; age-dependent work then stays pending.")
    brief = st.text_area("What would you like? (optional)", height=110, disabled=st.session_state.started,
                         placeholder="e.g. 4-room, keys in June 2024. Warm Japandi living room, sofa under $1,500. "
                                     "We'd also like to redo the bathroom floor tiles.",
                         help="Budgets and flat type can fill empty fields. Confirm the block completion date "
                              "in its date field; dates in this brief remain unconfirmed.")

    st.header("3. Budget (SGD)")
    total = st.number_input("Total furniture budget", min_value=0, value=6000, step=500,
                            disabled=st.session_state.started)
    works_budget = st.number_input("Renovation works budget (optional)", min_value=0.0, value=None,
                                   step=1000.0, disabled=st.session_state.started)
    caps = {}
    with st.expander("Per-item caps (optional)"):
        for t in st.multiselect("Item types", LOOSE_FURNITURE, default=["sofa", "coffee_table"],
                                disabled=st.session_state.started):
            caps[t] = float(st.number_input(f"Max for one {t.replace('_', ' ')}", min_value=0, value=1500 if t == "sofa" else 400,
                                            step=50, key=f"cap_{t}", disabled=st.session_state.started))
    if sum(caps.values()) > total:
        st.caption("Your category caps exceed the total. Each chosen item and the final delivered basket must still fit the total budget.")

    c1, c2 = st.columns(2)
    go = c1.button("Run", type="primary", disabled=st.session_state.started or not paths, width="stretch")
    if c2.button("Reset", width="stretch"):
        reset(planner, mode)
        st.rerun()
    u = st.session_state.usage
    if mode == LIVE:
        st.caption(f"{u['model_calls']} model calls · {u['input_tokens']:,} in / {u['output_tokens']:,} out tokens · "
                   f"~${u['cost_usd']:.2f} API-equivalent")
        st.caption("Backend: " + config.BACKEND + " · Catalog: IKEA SG + Castlery snapshot · Pricing: landed SGD (overseas rows are illustrations, not offers) · Compliance: HDB rule cards")
    else:
        st.caption("Models: played by code · Catalog: IKEA SG + Castlery snapshot (overseas rows are illustrations, not offers) · "
                   "Search: BM25 only · Pricing: real rules, frozen FX · Compliance: HDB rule cards")

# --------------------------------------------------------------------------- main
st.title("RenoPilot")
st.caption("From mood board to a shortlist of real products, within budget. You approve everything; nothing is bought.")
diagram = st.empty()
diagram.image(flow_svg(st.session_state.flow))
if mode == DEMO:
    st.info("**Scripted Japandi example.** This mode does not interpret images: it uses one fixed style profile "
            "and code-driven model replies. The graph, catalog tools, pricing and HDB checks run for real "
            "over the IKEA SG and Castlery snapshot. Use Live to interpret other boards or uploads. Try rejecting an item "
            "with a reason, or set a tight coffee-table cap (e.g. S$50) to get a question. Mention the bathroom in "
            "*What would you like?* and change the completion date to see the HDB 3-year rule respond.")

if paths and not st.session_state.started:
    st.image(paths, width=220)

if go:
    if (planner, mode) != (st.session_state.planner, st.session_state.mode):
        reset(planner, mode)
    st.session_state.started = True
    form = ProjectProfile(total_budget=total, budget_by_category=caps, completion_date=done, works_budget=works_budget,
                          flat_type=None if flat_type == "not sure" else flat_type)
    run({"board_images": paths, "project": form,
         "messages": [HumanMessage(brief.strip())] if brief.strip() else []})
    st.rerun()

if not st.session_state.started:
    st.info("Pick an example board or upload 3-12 inspiration photos, set a budget, then press **Run**.")
    st.stop()

state = st.session_state.graph.get_state(st.session_state.cfg).values
gate = pending_interrupt()

project = state.get("project")
if project:
    with st.container(border=True):
        st.subheader("Your flat")
        bits = [project.flat_type or "flat type not given"]
        if project.completion_date:
            unlock = project.completion_date.replace(year=project.completion_date.year + 3)                 if not (project.completion_date.month == 2 and project.completion_date.day == 29)                 else date(project.completion_date.year + 3, 3, 1)
            bits.append(f"block completed {project.completion_date:%d %b %Y}")
            bits.append(f"bathroom finishes can be replaced from {unlock:%d %b %Y}" if unlock > date.today()
                        else "past the 3-year bathroom restriction")
        else:
            bits.append("completion date unconfirmed: age-dependent work stays pending")
        if project.total_budget:
            bits.append(f"furniture budget {fmt_sgd(project.total_budget)}")
        st.markdown(md(" · ".join(bits)))
        if project.notes:
            st.caption(md(f"You asked: {project.notes}"))

profile = state.get("style_profile")
if profile:
    with st.container(border=True):
        st.subheader("Style profile")
        st.write(profile.summary)
        cols = st.columns(4)
        cols[0].markdown("**Style**  \n" + ", ".join(t.replace("_", " ") for t in profile.style_tags))
        cols[1].markdown("**Palette**  \n" + ", ".join(c.replace("_", " ") for c in profile.palette))
        cols[2].markdown("**Materials**  \n" + ", ".join(m.replace("_", " ") for m in profile.materials))
        cols[3].markdown("**Items**  \n" + ", ".join(i.item_type.replace("_", " ") for i in profile.items))

if gate and gate["type"] == "question":
    with st.container(border=True):
        st.subheader("RenoPilot needs your input")
        st.warning(md(gate["question"]))
        with st.form("answer"):
            answer = st.text_input("Your answer")
            new_caps = st.text_input("Change budgets (optional, e.g. sofa=2000, rug=300)")
            new_total = st.number_input("New total furniture budget (optional)", min_value=0.0, value=None, step=100.0)
            new_done = st.date_input("Confirm block completion date (optional)", value=None, format="YYYY-MM-DD")
            more = st.file_uploader("Add images (optional)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
            if st.form_submit_button("Send", type="primary"):
                resume = {"answer": answer}
                try:
                    if new_caps.strip():
                        resume["budget_by_category"] = {k.strip(): float(v) for k, v in
                                                        (kv.split("=") for kv in new_caps.split(","))}
                    if new_total is not None:
                        resume["total_budget"] = new_total
                    if new_done:
                        resume["completion_date"] = new_done
                    ProjectProfile.model_validate(project.model_dump() | {
                        "budget_by_category": project.budget_by_category | resume.get("budget_by_category", {}),
                        "total_budget": resume.get("total_budget", project.total_budget)})
                except (ValueError, TypeError):
                    st.error("Use item_type=amount, for example coffee_table=300. Amounts must be finite, non-negative SGD values.")
                    st.stop()
                if more and mode == LIVE:
                    up = config.CACHE / "uploads" / st.session_state.cfg["configurable"]["thread_id"]
                    up.mkdir(parents=True, exist_ok=True)
                    added_paths = []
                    for f in more:
                        dest = up / f"{uuid4().hex}_{Path(f.name).name}"
                        dest.write_bytes(f.getbuffer())
                        added_paths.append(str(dest))
                    resume["board_images"] = list(state.get("board_images") or []) + added_paths
                run(Command(resume=resume))
                st.rerun()

elif gate and gate["type"] == "approval":
    st.subheader("Choose your shopping list")
    st.caption("Pick one offer per requested item and room. Quantities apply to the total. "
               "Work with unverified conditions stays pending; reviewing this list does not authorise work to start.")
    if gate.get("error"):
        st.error(md(gate["error"]))
    if profile and profile.items:
        edit_requests(profile)
    edit_work_costs(project, state.get("suggestions") or [])
    with st.expander("Replace the brief or mood board"):
        with st.form("replace_inputs"):
            replacement_brief = st.text_area("Replacement brief", value=project.notes)
            replacement_images = st.file_uploader("Replacement images", type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True, disabled=mode != LIVE)
            st.caption("Replacing inputs rebuilds the shortlist and clears previous selections, approvals and work estimates. "
                       "Budgets in this text are not read: change them under Update budgets or completion date.")
            if st.form_submit_button("Rebuild from these inputs"):
                changes = {"notes": replacement_brief}
                if replacement_images and mode == LIVE:
                    up = config.CACHE / "uploads" / st.session_state.cfg["configurable"]["thread_id"]
                    up.mkdir(parents=True, exist_ok=True)
                    new_paths = []
                    for f in replacement_images:
                        dest = up / f"{uuid4().hex}_{Path(f.name).name}"
                        dest.write_bytes(f.getbuffer())
                        new_paths.append(str(dest))
                    changes["board_images"] = new_paths
                run(Command(resume=changes))
                st.rerun()
    with st.expander("Update budgets or completion date"):
        with st.form("update_project"):
            update_total = st.number_input("New overall budget (SGD)", min_value=0.0, value=None, step=100.0)
            update_caps = st.text_input("New category caps (e.g. sofa=1200, coffee_table=300)")
            update_done = st.date_input("Block completion date", value=None, format="YYYY-MM-DD")
            if st.form_submit_button("Update project and search again"):
                try:
                    changes = {}
                    if update_total is not None:
                        changes["total_budget"] = update_total
                    if update_caps.strip():
                        changes["budget_by_category"] = {k.strip(): float(v) for k, v in
                                                         (kv.split("=") for kv in update_caps.split(","))}
                    if update_done:
                        changes["completion_date"] = update_done
                    if not changes:
                        raise ValueError("Enter at least one change.")
                    ProjectProfile.model_validate(project.model_dump() | changes)
                except (ValueError, TypeError):
                    st.error("Enter a total, a completion date, or category=amount caps using finite, non-negative SGD values.")
                else:
                    run(Command(resume=changes))
                    st.rerun()
    for e in gate.get("escalations") or []:
        st.info(md(e))
    show_blocked(gate.get("blocked") or [])
    for s in state.get("suggestions") or []:
        if s.status != "rejected" or not s.verdict or s.verdict.decision != "block":
            continue
        options = {o.rule_id: o for o in s.verdict.options if o.decision in ("allow", "allow_with_permit")}
        if options:
            alternative = st.selectbox(f"Permitted alternative to {s.title[:60]}", list(options), index=None,
                format_func=lambda rid, opts=options: opts[rid].route, key=f"alternative_{s.key}")
            if st.button("Propose this alternative", key=f"propose_{s.key}", disabled=not alternative):
                run(Command(resume={"alternatives": {s.key: alternative}}))
                st.rerun()
    offers = offers_for(state.get("suggestions") or [])
    selected = {}
    with st.container():
        by_type: dict[str, list[dict]] = {}
        for c in gate["suggestions"]:
            by_type.setdefault(c.get("selection_key", c["item_type"]), []).append(c)
        for t, cards in by_type.items():
            category = cards[0]["item_type"].replace("_", " ")
            room_name = cards[0].get("room_label") or ""
            st.markdown(f"#### {category.title()}" + (f" · {room_name}" if room_name else ""))
            cols = st.columns(min(len(cards), 5))
            for i, c in enumerate(cards):
                with cols[i % len(cols)]:
                    with st.container(border=True):
                        if c["product_id"]:
                            picture(c.get("image_path"))
                        st.markdown(f"**{c['title'][:60]}**")
                        if c.get("room"):
                            st.caption(f"Room: {c.get('room_label') or c['room'].replace('_', ' ')}")
                        if c["product_id"]:
                            st.caption(f"Quantity: {c.get('quantity', 1)} · listed prices are per unit")
                            st.caption(c.get("fit_note") or "Fit not checked: dimensions unavailable.")
                            st.caption("Catalog captured: " + (c.get("retrieved_at") or "date unknown"))
                            st.caption("Dimensions: " + (c.get("dims") or "unknown") + " · Material: " + (c.get("material") or "unknown"))
                            if c.get("landed"):
                                est = " · estimated" if c.get("estimated") else ""
                                link = f" · [link]({c['url']})" if c.get("url") else ""
                                st.markdown(md(f"**{c['landed']}** landed{est} · listed {c['price']}") + link)
                                with st.expander("Price breakdown"):
                                    for ln in c["landed_lines"]:
                                        st.caption(md(ln))
                                    for a in c.get("assumptions") or []:
                                        st.caption(md(f"Assumption: {a}"))
                            else:
                                st.markdown(md(c["price"]) + (f" · [link]({c['url']})" if c.get("url") else ""))
                            st.caption(md(c["reason"]))
                            for a in c.get("comparables") or []:
                                word = ":orange[illustration, not a real offer]" if a["sample"] else (
                                    ":green[cheaper]" if a["cheaper"] else "similar")
                                st.caption(word + md(f" at {a['source']}: {a['title'][:45]}, {a['landed']} landed")
                                           + (f" · [link]({a['url']})" if a.get("url") else ""))
                        if c.get("routes"):
                            hdb_routes(c)
                        elif c["verdict"] and c["verdict"] != "allow":
                            st.warning(md(f"{c['verdict']}: {c['verdict_reason']}"))
                        if c["product_id"]:
                            st.checkbox("Reject", key=f"rej_{t}_{c['product_id']}")
                            st.text_input("Why?", key=f"why_{t}_{c['product_id']}", placeholder="e.g. too bulky",
                                          label_visibility="collapsed")
            if t in offers:
                by_id = {s.product_id: s for s in offers[t]}
                ids = list(by_id)
                default = gate.get("selected", {}).get(t)
                pick_key = f"pick_{t}"
                if st.session_state.get(pick_key) not in ids:
                    st.session_state[pick_key] = default if default in ids else ids[0]
                selected[t] = st.selectbox(f"Choose one {category}" + (f" for {room_name}" if room_name else ""), ids, key=pick_key,
                    format_func=lambda pid, items=by_id: f"{items[pid].title} · {fmt_sgd(items[pid].price_quote.total_sgd) if items[pid].price_quote else 'delivered price unavailable'}")
        picks = [next(s for s in offers[t] if s.product_id == pid) for t, pid in selected.items()]
        totals = summary(picks)
        cols = st.columns(3)
        cols[0].metric("Selected delivered total", fmt_sgd(totals["total_sgd"]))
        cols[1].metric("Total budget", fmt_sgd(project.total_budget))
        if project.total_budget is not None:
            cols[2].metric("Remaining", fmt_sgd(round(project.total_budget - totals["total_sgd"], 2)))
        st.caption(totals["basis"] + (" · includes estimates" if totals["estimated"] else ""))
        work_cost_summary(project, state.get("suggestions") or [], totals["total_sgd"])
        invalid = None
        routes = {c["key"]: st.session_state[f"route_{c['key']}"] for c in gate["suggestions"]
                  if st.session_state.get(f"route_{c['key']}")}
        try:
            preview = apply_routes(state.get("suggestions") or [], routes, project)
            selection(preview, selected, project)
        except ValueError as e:
            invalid = str(e)
            st.warning(md(invalid))
        a, b = st.columns(2)
        approve = a.button("Approve selected basket", type="primary", width="stretch", disabled=bool(invalid))
        reject = b.button("Submit rejections", width="stretch")
    if approve or reject:
        rejects = [{"product_id": c["product_id"], "request_id": c.get("request_id", ""),
                    "reason": st.session_state.get(f"why_{c.get('selection_key', c['item_type'])}_{c['product_id']}", "")}
                   for c in gate["suggestions"] if c["product_id"] and
                   st.session_state.get(f"rej_{c.get('selection_key', c['item_type'])}_{c['product_id']}")]
        run(Command(resume=({"approve": "selected", "selected": selected} if approve else {"reject": rejects}) | {"routes": routes}))
        st.rerun()

else:
    msgs = state.get("messages") or []
    if msgs:
        st.subheader("Your shopping list")
        st.success("Selected basket approved." if state.get("approved") else "Finished.")
        chosen = [s for s in state.get("suggestions") or [] if s.product_id and s.status == "approved"]
        if state.get("approved"):
            totals = summary(chosen)
            cols = st.columns(3)
            cols[0].metric("Delivered total", fmt_sgd(totals["total_sgd"]))
            cols[1].metric("Budget", fmt_sgd(project.total_budget))
            if project.total_budget is not None:
                cols[2].metric("Remaining", fmt_sgd(round(project.total_budget - totals["total_sgd"], 2)))
            st.caption(totals["basis"] + (" · includes estimates" if totals["estimated"] else ""))
            work_cost_summary(project, state.get("suggestions") or [], totals["total_sgd"])
            st.download_button("Download shopping list (CSV)", shopping_list_csv(state),
                               file_name="renopilot_shopping_list.csv", mime="text/csv")
            if st.button("Revise this project"):
                st.session_state.graph.update_state(st.session_state.cfg, {"approved": False,
                    "suggestions": [s.model_copy(update={"status": "proposed"}) if s.status == "approved" else s
                                    for s in state.get("suggestions") or []]}, as_node="compliance_guard")
                run(None)
                st.rerun()
        show_blocked(blocked(state.get("suggestions")))
        works = [x for x in state.get("suggestions") or [] if not x.product_id and x.status == "approved" and x.verdict]
        if works:
            st.markdown("#### Renovation works (HDB)")
        for x in works:
            v = x.verdict
            link = f" · [{v.rule_id}]({v.citation})" if v.rule_id else ""
            st.markdown(md(f"**{x.title}** · {v.route or v.reason} · {PLAIN[v.decision]}") + link)
            for cond in v.conditions:
                st.caption(md(f"- {cond}"))
        for x in state.get("suggestions") or []:
            if not x.product_id and x.status == "proposed" and x.verdict:
                st.warning(md(f"Pending: {x.title} ({x.room or 'room unknown'}). {x.verdict.reason}"))
                for condition in x.verdict.conditions:
                    st.caption(md(condition))
        by_type: dict[str, list] = {}
        for x in state.get("suggestions") or []:
            if x.product_id and x.status == "approved":
                by_type.setdefault(x.item_type, []).append(x)
        for t, picks in by_type.items():
            st.markdown(f"#### {t.replace('_', ' ').title()}")
            cols = st.columns(5)
            for i, x in enumerate(picks):
                with cols[i % 5], st.container(border=True):
                    picture(x.image_path)
                    price = f"{fmt_sgd(x.price_quote.total_sgd)} landed" if x.price_quote else fmt_sgd(x.price_sgd)
                    st.markdown(md(f"**{x.title[:60]}**  \n{x.quantity} × {price}") + (f" · [link]({x.url})" if x.url else ""))
                    st.caption(x.room_label or x.room or "Room unspecified")
                    st.caption(x.fit_note or "Fit not checked")
                    st.caption(md(x.reason))
        with st.expander("The reply as text"):
            st.text(msgs[-1].content)

with st.expander("What the agent did (decision log)"):
    for d in state.get("trace", []):
        extra = f" · types {d['item_types']}" if d.get("item_types") else ""
        extra += f" · refinement '{d['refinement']}'" if d.get("refinement") else ""
        st.markdown(f"- `{d['node']}` **{d['action']}**: {d.get('why', '')}{extra}")
    if st.session_state.get("last_trace"):
        st.caption(f"Full trace of the last step: {st.session_state.last_trace}")
    if url := trace.langfuse_url(st.session_state.cfg):
        st.caption(f"[This conversation in Langfuse]({url})")

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

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from renopilot import config, trace  # noqa: E402
from renopilot.catalog.embed import slug  # noqa: E402
from renopilot.compliance import PLAIN  # noqa: E402
from renopilot.demo import demo_nodes  # noqa: E402
from renopilot.graph import build_graph  # noqa: E402
from renopilot.nodes.guard import blocked  # noqa: E402
from renopilot.schemas import ProjectProfile  # noqa: E402
from renopilot.tools.budget import fmt_sgd  # noqa: E402
from renopilot.vocab import LOOSE_FURNITURE  # noqa: E402

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
    ok = [r["rule_id"] for r in c["routes"] if r["decision"] != "block"]
    if ok:
        st.radio("How will this be done?", ok, key=f"route_{c['item_type']}",
                 index=ok.index(c["rule_id"]) if c["rule_id"] in ok else 0,
                 format_func=lambda i: f"{by_id[i]['route']} ({PLAIN[by_id[i]['decision']]})",
                 captions=[md(by_id[i]["reason"]) for i in ok])
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
    reset("llm", LIVE if live_ready() is None else DEMO)
st.session_state.setdefault("flow", {})                  # sessions started before the diagram existed


def run(payload) -> None:
    """Stream the graph until it finishes or parks at an interrupt, lighting up each node as it runs."""
    g, cfg, flow = st.session_state.graph, st.session_state.cfg, st.session_state.flow
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
    st.header("1. Mood board")
    boards = sorted(p.name for p in config.BOARDS.iterdir() if p.is_dir()) if config.BOARDS.exists() else []
    source = st.radio("Board source", ["Example board", "Upload images"], horizontal=True,
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
                (up / f.name).write_bytes(f.getbuffer())
                paths.append(str(up / f.name))

    st.header("2. Your flat & what you'd like")
    FLATS = ["not sure", "2-room", "3-room", "4-room", "5-room", "executive", "other"]
    flat_type = st.selectbox("Flat type", FLATS, index=0, disabled=st.session_state.started)
    done = st.date_input("Block completion (key collection) date", value=None, format="YYYY-MM-DD",
                         min_value=date(1970, 1, 1), max_value=date(date.today().year + 6, 12, 31),
                         disabled=st.session_state.started,
                         help="From your key collection letter. HDB does not allow bathroom/toilet floor and wall "
                              "finishes to be replaced for 3 years from the block's completion date. Leave empty if "
                              "unsure: RenoPilot then shows both the under-3-year and 3-year-plus routes.")
    brief = st.text_area("What would you like? (optional)", height=110, disabled=st.session_state.started,
                         placeholder="e.g. 4-room, keys in June 2024. Warm Japandi living room, sofa under $1,500. "
                                     "We'd also like to redo the bathroom floor tiles.",
                         help="In your own words. Budgets, the flat type or a completion date typed here fill in "
                              "anything the fields above leave empty (the fields always win).")

    st.header("3. Budget (SGD)")
    total = st.number_input("Total furniture budget", min_value=0, value=6000, step=500,
                            disabled=st.session_state.started)
    caps = {}
    with st.expander("Per-item caps (optional)"):
        for t in st.multiselect("Item types", LOOSE_FURNITURE, default=["sofa", "coffee_table"],
                                disabled=st.session_state.started):
            caps[t] = float(st.number_input(f"Max for one {t.replace('_', ' ')}", min_value=0, value=1500 if t == "sofa" else 400,
                                            step=50, key=f"cap_{t}", disabled=st.session_state.started))

    st.header("4. Agent")
    not_live = live_ready()
    mode = st.radio("Mode", [DEMO, LIVE], horizontal=True, disabled=st.session_state.started,
                    index=[DEMO, LIVE].index(st.session_state.mode),
                    help="Demo: offline, models played by code over a made-up sample catalog. "
                         "Live: the real models and catalog." + (f" Live is unavailable: {not_live}." if not_live else ""))
    planner = st.session_state.planner
    if mode == LIVE:
        planner = st.radio("Planner", ["llm", "fixed"], horizontal=True, disabled=st.session_state.started,
                           help="llm: the model picks the next step, code enforces the rules. fixed: the baseline pipeline.")
        if not_live:
            st.warning(f"Live mode will fail: {not_live}.")
    c1, c2 = st.columns(2)
    go = c1.button("Run", type="primary", disabled=st.session_state.started or not paths, width="stretch")
    if c2.button("Reset", width="stretch"):
        reset(planner, mode)
        st.rerun()
    u = st.session_state.usage
    if mode == LIVE:
        st.caption(f"{u['model_calls']} model calls · {u['input_tokens']:,} in / {u['output_tokens']:,} out tokens · "
                   f"~${u['cost_usd']:.2f} API-equivalent")
        st.caption("Backend: " + config.BACKEND + " · Catalog: IKEA SG + Castlery snapshot · Pricing: landed SGD (overseas rows are sample data) · Compliance: HDB rule cards")
    else:
        st.caption("Models: played by code · Catalog: IKEA SG + Castlery snapshot (overseas rows are sample data) · "
                   "Search: BM25 only · Pricing: real rules, frozen FX · Compliance: HDB rule cards")

# --------------------------------------------------------------------------- main
st.title("RenoPilot")
st.caption("From mood board to a shortlist of real products, within budget. You approve everything; nothing is bought.")
diagram = st.empty()
diagram.image(flow_svg(st.session_state.flow))
if mode == DEMO:
    st.info("**Demo mode.** The graph, the agents' tools, pricing, the compliance gate and tracing are the real code, "
            "over the real IKEA SG and Castlery snapshot; only the models are played by code. Try rejecting an item "
            "with a reason, or set a tight coffee-table cap (e.g. S$50) to get a question. Mention the bathroom in "
            "*What would you like?* and change the completion date to see the HDB 3-year rule respond.")

if paths and not st.session_state.started:
    st.image(paths, width=220)

if go:
    if (planner, mode) != (st.session_state.planner, st.session_state.mode):
        reset(planner, mode)
    st.session_state.started = True
    form = ProjectProfile(total_budget=total or None, budget_by_category=caps, completion_date=done,
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
            bits.append("completion date not given: both bathroom routes (under / over 3 years) are shown")
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
            more = st.file_uploader("Add images (optional)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
            if st.form_submit_button("Send", type="primary"):
                resume = {"answer": answer}
                if new_caps.strip():
                    resume["budget_by_category"] = {k.strip(): float(v) for k, v in
                                                    (kv.split("=") for kv in new_caps.split(",") if "=" in kv)}
                if more:
                    up = config.CACHE / "uploads" / st.session_state.cfg["configurable"]["thread_id"]
                    up.mkdir(parents=True, exist_ok=True)
                    for f in more:
                        (up / f.name).write_bytes(f.getbuffer())
                    resume["board_images"] = list(state.get("board_images") or []) + [str(up / f.name) for f in more]
                run(Command(resume=resume))
                st.rerun()

elif gate and gate["type"] == "approval":
    st.subheader("Shortlist: approve or reject")
    for e in gate.get("escalations") or []:
        st.info(md(e))
    show_blocked(gate.get("blocked") or [])
    with st.form("approval"):
        by_type: dict[str, list[dict]] = {}
        for c in gate["suggestions"]:
            by_type.setdefault(c["item_type"], []).append(c)
        for t, cards in by_type.items():
            st.markdown(f"#### {t.replace('_', ' ').title()}")
            cols = st.columns(min(len(cards), 5))
            for i, c in enumerate(cards):
                with cols[i % len(cols)]:
                    with st.container(border=True):
                        if c["product_id"]:
                            picture(c.get("image_path"))
                        st.markdown(f"**{c['title'][:60]}**")
                        if c["product_id"]:
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
                                badge = " · :orange[SAMPLE DATA]" if a["sample"] else ""
                                word = ":green[cheaper]" if a["cheaper"] else "similar"
                                st.caption(md(f"{word} at {a['source']}: {a['title'][:45]}, {a['landed']} landed") + badge)
                        if c.get("routes"):
                            hdb_routes(c)
                        elif c["verdict"] and c["verdict"] != "allow":
                            st.warning(md(f"{c['verdict']}: {c['verdict_reason']}"))
                        if c["product_id"]:
                            st.checkbox("Reject", key=f"rej_{c['product_id']}")
                            st.text_input("Why?", key=f"why_{c['product_id']}", placeholder="e.g. too bulky",
                                          label_visibility="collapsed")
        a, b = st.columns(2)
        approve = a.form_submit_button("Approve all", type="primary", width="stretch")
        reject = b.form_submit_button("Submit rejections", width="stretch")
    if approve or reject:
        rejects = [{"product_id": c["product_id"], "reason": st.session_state.get(f"why_{c['product_id']}", "")}
                   for c in gate["suggestions"] if c["product_id"] and st.session_state.get(f"rej_{c['product_id']}")]
        routes = {c["item_type"]: st.session_state[f"route_{c['item_type']}"] for c in gate["suggestions"]
                  if f"route_{c['item_type']}" in st.session_state}
        run(Command(resume=({"approve": "all"} if approve or not rejects else {"reject": rejects}) | {"routes": routes}))
        st.rerun()

else:
    msgs = state.get("messages") or []
    if msgs:
        st.subheader("Your shortlist")
        st.success("Shortlist approved." if state.get("approved") else "Finished.")
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
        by_type: dict[str, list] = {}
        for x in state.get("suggestions") or []:
            if x.product_id and x.status == "approved":
                by_type.setdefault(x.item_type, []).append(x)
        for t, picks in by_type.items():
            st.markdown(f"#### {t.replace('_', ' ').title()}")
            cols = st.columns(5)
            for col, x in zip(cols, picks):
                with col, st.container(border=True):
                    picture(x.image_path)
                    price = f"{fmt_sgd(x.price_quote.total_sgd)} landed" if x.price_quote else fmt_sgd(x.price_sgd)
                    st.markdown(md(f"**{x.title[:60]}**  \n{price}") + (f" · [link]({x.url})" if x.url else ""))
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

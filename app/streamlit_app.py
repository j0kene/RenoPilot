"""Temporary demo UI for P0 (mood board -> shortlist). Compliance is still the stub and
Pricing is not wired in yet, so this shows the Style agent, the planner and the two human
gates (question + approval).

    python -m uv run --group ui streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langgraph.types import Command  # noqa: E402

from renopilot import config, trace  # noqa: E402
from renopilot.graph import build_graph  # noqa: E402
from renopilot.schemas import ProjectProfile  # noqa: E402
from renopilot.vocab import LOOSE_FURNITURE  # noqa: E402

st.set_page_config(page_title="RenoPilot demo", page_icon="🛋️", layout="wide")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
NODE_LABEL = {"intake": "Checking the board", "planner": "Planning the next step",
              "style_extract": "Reading the mood board", "shortlist": "Searching the catalog",
              "compliance_guard": "Compliance check (stub)", "ask_user": "Question for you",
              "approval": "Waiting for your approval", "respond": "Writing the reply"}


# --------------------------------------------------------------------------- session
def reset(planner: str) -> None:
    st.session_state.planner = planner
    st.session_state.graph = build_graph(planner=planner)
    st.session_state.cfg = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 80}
    st.session_state.started = False
    st.session_state.log = []
    st.session_state.usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


if "graph" not in st.session_state:
    reset("llm")


def run(payload) -> None:
    """Stream the graph until it finishes or parks at an interrupt, showing each node live."""
    g, cfg = st.session_state.graph, st.session_state.cfg
    with st.status("RenoPilot is working...", expanded=True) as status, trace.tracing() as t:
        t0 = time.perf_counter()
        for update in g.stream(payload, cfg, stream_mode="updates"):
            for node, delta in update.items():
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
        pick = st.selectbox("Board", boards, disabled=st.session_state.started)
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

    st.header("2. Budget (SGD)")
    total = st.number_input("Total furniture budget", min_value=0, value=6000, step=500,
                            disabled=st.session_state.started)
    caps = {}
    with st.expander("Per-item caps (optional)"):
        for t in st.multiselect("Item types", LOOSE_FURNITURE, default=["sofa", "coffee_table"],
                                disabled=st.session_state.started):
            caps[t] = float(st.number_input(f"Max for one {t.replace('_', ' ')}", min_value=0, value=1500 if t == "sofa" else 400,
                                            step=50, key=f"cap_{t}", disabled=st.session_state.started))

    st.header("3. Agent")
    planner = st.radio("Planner", ["llm", "fixed"], horizontal=True, disabled=st.session_state.started,
                       help="llm: the model picks the next step, code enforces the rules. fixed: the baseline pipeline.")
    c1, c2 = st.columns(2)
    go = c1.button("Run", type="primary", disabled=st.session_state.started or not paths, width="stretch")
    if c2.button("Reset", width="stretch"):
        reset(planner)
        st.rerun()
    u = st.session_state.usage
    st.caption(f"{u['model_calls']} model calls · {u['input_tokens']:,} in / {u['output_tokens']:,} out tokens · "
               f"~${u['cost_usd']:.2f} API-equivalent")
    st.caption("Backend: " + config.BACKEND + " · Catalog: IKEA SG + Castlery snapshot · Pricing: landed SGD (overseas rows are sample data) · Compliance: stub")

# --------------------------------------------------------------------------- main
st.title("RenoPilot")
st.caption("From mood board to a shortlist of real products, within budget. You approve everything; nothing is bought.")

if paths and not st.session_state.started:
    st.image(paths, width=220)

if go:
    if planner != st.session_state.planner:
        reset(planner)
    st.session_state.started = True
    run({"board_images": paths, "project": ProjectProfile(total_budget=total or None, budget_by_category=caps),
         "messages": []})
    st.rerun()

if not st.session_state.started:
    st.info("Pick an example board or upload 3-12 inspiration photos, set a budget, then press **Run**.")
    st.stop()

state = st.session_state.graph.get_state(st.session_state.cfg).values
gate = pending_interrupt()

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
        st.warning(gate["question"])
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
        st.info(e)
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
                        img = config.ROOT / c["image_path"] if c.get("image_path") else None
                        if img and img.exists():
                            st.image(str(img), width="stretch")
                        st.markdown(f"**{c['title'][:60]}**")
                        if c["product_id"]:
                            if c.get("landed"):
                                est = " · estimated" if c.get("estimated") else ""
                                st.markdown(f"**{c['landed']}** landed{est} · listed {c['price']} · [link]({c['url']})")
                                with st.expander("Price breakdown"):
                                    for ln in c["landed_lines"]:
                                        st.caption(ln)
                                    for a in c.get("assumptions") or []:
                                        st.caption(f"Assumption: {a}")
                            else:
                                st.markdown(f"{c['price']} · [link]({c['url']})")
                            st.caption(c["reason"])
                            for a in c.get("comparables") or []:
                                badge = " · :orange[SAMPLE DATA]" if a["sample"] else ""
                                word = ":green[cheaper]" if a["cheaper"] else "similar"
                                st.caption(f"{word} at {a['source']}: {a['title'][:45]}, {a['landed']} landed{badge}")
                        if c["verdict"] and c["verdict"] != "allow":
                            st.warning(f"{c['verdict']}: {c['verdict_reason']}")
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
        run(Command(resume={"approve": "all"} if approve or not rejects else {"reject": rejects}))
        st.rerun()

else:
    msgs = state.get("messages") or []
    if msgs:
        st.subheader("Final reply")
        st.success("Shortlist approved." if state.get("approved") else "Finished.")
        st.text(msgs[-1].content)

with st.expander("What the agent did (decision log)"):
    for d in state.get("trace", []):
        extra = f" · types {d['item_types']}" if d.get("item_types") else ""
        extra += f" · refinement '{d['refinement']}'" if d.get("refinement") else ""
        st.markdown(f"- `{d['node']}` **{d['action']}**: {d.get('why', '')}{extra}")
    if st.session_state.get("last_trace"):
        st.caption(f"Full trace of the last step: {st.session_state.last_trace}")

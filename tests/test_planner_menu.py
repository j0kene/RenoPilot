"""The planner's rules live in code: an illegal decision is refused whatever the model says."""
from langgraph.types import Command

from renopilot import config
from renopilot.nodes.planner import Decision, check_action, fixed_policy, llm_decide, make_planner
from renopilot.schemas import Suggestion
from tests.fakes import PROFILE, ScriptedPlanner


def test_shortlist_before_style_is_refused():
    assert "extract_style first" in check_action({}, Decision(action="shortlist", item_types=["sofa"], why=""))


def test_shortlist_of_items_not_on_board_is_refused():
    reason = check_action({"style_profile": PROFILE}, Decision(action="shortlist", item_types=["bar_stool"], why=""))
    assert "not on the board" in reason


def test_finalize_needs_suggestions_and_no_open_issues():
    assert check_action({"style_profile": PROFILE}, Decision(action="finalize", why=""))
    s = [Suggestion(product_id="x", item_type="sofa", work_type="loose_furniture")]
    assert check_action({"style_profile": PROFILE, "suggestions": s, "open_issues": ["too few sofas"]},
                        Decision(action="finalize", why=""))
    assert check_action({"style_profile": PROFILE, "suggestions": s}, Decision(action="finalize", why="")) is None


def test_fixed_policy_is_always_legal():
    s = [Suggestion(product_id="x", item_type="sofa", work_type="loose_furniture")]
    for state in ({}, {"style_profile": PROFILE}, {"style_profile": PROFILE, "suggestions": s},
                  {"style_profile": PROFILE, "suggestions": s, "redo_types": ["sofa"]},
                  {"open_issues": ["board too small"]}):
        assert check_action(state, fixed_policy(state)) is None, state


def test_refused_decision_gets_one_retry_then_fixed_policy():
    fake = ScriptedPlanner([{"action": "finalize", "why": "eager"}, {"action": "finalize", "why": "still eager"}])
    d = llm_decide({"style_profile": PROFILE}, model=fake)
    assert fake.calls == 2
    assert d.action == "shortlist" and "fallback" in d.why


def test_legal_llm_decision_is_used_including_refinement():
    s = [Suggestion(product_id="x", item_type="sofa", work_type="loose_furniture", status="rejected")]
    fake = ScriptedPlanner([{"action": "shortlist", "item_types": ["sofa"], "refinement": "darker walnut, slimmer arms",
                             "why": "owner wants darker wood"}])
    out = make_planner("llm", model=fake)({"style_profile": PROFILE, "suggestions": s, "redo_types": ["sofa"]})
    assert isinstance(out, Command) and out.goto == "shortlist"
    assert out.update["refinements"] == {"sofa": "darker walnut, slimmer arms"}


def test_unparseable_reply_falls_back():
    d = llm_decide({}, model=ScriptedPlanner([]))
    assert "fallback" in d.why
    assert d.action == "ask_user" and "at least 3" in d.question      # no board: the rule, not the model, decides


def test_extract_style_refused_while_board_too_small_even_with_no_open_issue():
    reason = check_action({"board_images": ["a.jpg"]}, Decision(action="extract_style", why=""))
    assert reason and "not usable" in reason


def test_step_cap_ends_the_turn():
    out = make_planner("fixed")({"steps": config.MAX_PLANNER_STEPS})
    assert out.goto == "respond" and "Stopped" in out.update["escalations"][0]


def test_repeat_search_without_new_feedback_is_refused():
    """Regression: the LLM planner re-searched sofas until the step cap after one rejection."""
    s = [Suggestion(product_id="x", item_type="sofa", work_type="loose_furniture"),
         Suggestion(product_id="y", item_type="coffee_table", work_type="loose_furniture")]
    again = Decision(action="shortlist", item_types=["sofa"], why="")
    assert "already searched" in check_action({"style_profile": PROFILE, "suggestions": s}, again)
    assert check_action({"style_profile": PROFILE, "suggestions": s, "redo_types": ["sofa"]}, again) is None
    first = Decision(action="shortlist", item_types=["floor_tiles"], why="never searched")
    assert check_action({"style_profile": PROFILE, "suggestions": s}, first) is None

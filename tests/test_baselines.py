"""The baselines and ablation the report compares against, offline: no model is called."""
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from evals import run_compliance, run_end_to_end
from evals.harness import AgentResult, Case
from evals.run_end_to_end import Plan, grade, grade_no_tools, make_agent, no_tools_agent

PROJECT = {"total_budget": 6000, "budget_by_category": {"sofa": 1500, "coffee_table": 400}, "as_of": "2026-10-28"}
NEW_BLOCK = PROJECT | {"completion_date": "2025-01-01"}


class FakeModel:
    """Replays structured replies, with the usage a real model would report."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def with_structured_output(self, schema, include_raw=False, **_):
        def run(messages):
            self.seen.append(messages)
            raw = AIMessage("", usage_metadata={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                            response_metadata={"cost_usd": 0.01})
            return {"raw": raw, "parsed": schema.model_validate(self.replies.pop(0)), "parsing_error": None}
        return RunnableLambda(run)


def journey(project, brief):
    return {"project": project, "board": "japandi_01", "brief": brief, "script": ["approve"]}


SOFA = {"item_type": "sofa", "title": "Made-up sofa", "price_sgd": 900}


# ---------------------------------------------------------------------- the ablation
def test_removing_the_guard_lets_blocked_work_reach_the_owner():
    case = Case("e05", journey(NEW_BLOCK, "Replace bathroom floor tiles"), {"blocked_rule": "HDB-BATH-3YR"})
    guarded = make_agent("demo")(case.input)
    assert grade(case, guarded) and guarded.metrics["violations"] == 0
    ablated = make_agent("demo", guard=False)(case.input)
    assert not grade(case, ablated) and ablated.metrics["violations"] >= 1
    assert "rule_violation" in ablated.failure_tags


def test_e04_needs_the_unknown_age_escalated_not_only_an_approval():
    case = Case("e04", journey(PROJECT, "Replace bathroom floor tiles"), {"escalated": True})
    assert grade(case, make_agent("demo")(case.input))
    assert not grade(case, make_agent("demo", guard=False)(case.input))


def test_an_illustrative_row_in_the_reply_or_export_fails_the_rubric():
    case = Case("e01", journey(PROJECT, ""), {})
    result = make_agent("demo")(case.input)
    assert grade(case, result)
    result.output["export"] += "[SAMPLE] Japandi sofa"
    assert not grade(case, result) and result.failure_tags == ["unsupported_claim"]


# ---------------------------------------------------------------------- no-tools, end to end
def graded(expected, plan, project=PROJECT, **output):
    result = AgentResult(output={"plans": [Plan.model_validate(plan).model_dump()], "project": project, "approved": True} | output)
    return grade_no_tools(Case("x", {}, expected), result), result


def test_no_tools_passes_on_the_right_decision_within_budget_and_reports_invented_products():
    work = {"request": "wall", "decision": "block"}
    ok, result = graded({"blocked_rule": "HDB-WALL-STRUCTURAL"}, {"products": [SOFA], "works": [work]})
    assert ok and result.metrics["catalog_match"] == 0 and result.metrics["violations"] == 0


def test_no_tools_clearing_blocked_work_is_a_violation():
    work = {"request": "wall", "decision": "allow_with_permit"}
    ok, result = graded({"blocked_rule": "HDB-WALL-STRUCTURAL"}, {"products": [SOFA], "works": [work]})
    assert not ok and result.failure_tags == ["rule_violation"] and result.metrics["violations"] == 1


def test_no_tools_over_a_cap_or_the_total_fails():
    over_cap = dict(SOFA, price_sgd=1600)
    assert graded({}, {"products": [over_cap]})[1].failure_tags == ["over_budget"]
    over_total = [dict(SOFA, item_type="sofa", price_sgd=1500), dict(SOFA, item_type="bed_frame", price_sgd=4600)]
    assert graded({}, {"products": over_total})[1].failure_tags == ["over_budget"]
    assert graded({}, {"products": []})[1].failure_tags == ["retrieval_miss"]


def test_no_tools_gets_the_same_conversation_and_records_usage(monkeypatch):
    model = FakeModel({"products": [SOFA]}, {"products": [dict(SOFA, title="Darker sofa")]})
    monkeypatch.setattr(run_end_to_end, "get_model", lambda role: model)
    inp = journey(PROJECT, "") | {"script": ["reject:sofa:1:prefer darker wood", "approve"]}
    result = no_tools_agent(inp)
    assert result.output["approved"] and len(result.output["plans"]) == 2
    assert "prefer darker wood" in model.seen[1][-1].content
    assert sum(b.get("type") == "image" for b in model.seen[0][1].content) >= 3
    assert result.input_tokens == 22 and result.cost_usd == 0.02
    assert grade_no_tools(Case("e02", inp, {"reshortlist": True}), result)


def test_plan_schema_cannot_invent_an_item_type():
    assert Plan.model_validate({"products": [SOFA]}).products[0].item_type == "sofa"


# ---------------------------------------------------------------------- compliance
QUESTION = {"question": "Can I retile my one-year-old toilet?", "project": {"completion_date": "2025-06-01", "as_of": "2026-10-28"},
            "request": {"item_type": "floor_tiles", "work_type": "finish", "room": "bathroom", "work_method": "replace"}}


def test_no_tools_compliance_is_graded_on_the_decision_and_flags_an_unsafe_yes(monkeypatch):
    monkeypatch.setattr(run_compliance, "get_model", lambda role: FakeModel({"decision": "allow"}))
    result = run_compliance.no_tools_agent(QUESTION)
    assert not run_compliance.grade_no_tools(Case("q", QUESTION, {"decision": "block"}), result)
    assert result.metrics == {"decision_correct": 0.0, "unsafe_clear": 1.0}
    assert result.input_tokens == 11


def test_live_compliance_records_the_intake_call(monkeypatch):
    """Regression: QA and red-team live runs reported 0 tokens and $0."""
    reply = {"item_type": "floor_tiles", "description": "Retile the toilet", "room": "bathroom", "work_method": "replace"}
    monkeypatch.setattr(run_compliance, "get_model", lambda role: FakeModel(reply))
    result = run_compliance.make_agent(live=True)(QUESTION)
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (11, 3, 0.01)
    assert result.output["decision"] == "block"

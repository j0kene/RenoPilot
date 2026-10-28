"""Changed gold labels and incomplete repetitions cannot reuse a passing result."""
from evals.harness import AgentResult, Case, CaseOutcome, Report, run_eval


def test_evaluation_resume_does_not_reuse_changed_case_labels(tmp_path):
    path = tmp_path / "eval.jsonl"
    agent = lambda inp: AgentResult(output=inp)
    grader = lambda case, result: case.expected == result.output
    assert run_eval(agent, [Case("x", 1, 1)], grader, n_runs=1, resume_file=path).mean_pass == 1
    assert run_eval(agent, [Case("x", 1, 2)], grader, n_runs=1, resume_file=path).mean_pass == 0
    partial = Report("partial", [CaseOutcome("x", True, 0, AgentResult(output=""))], 5)
    assert partial.pass_hat_k == 0

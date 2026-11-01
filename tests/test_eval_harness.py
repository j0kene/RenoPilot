"""Changed gold labels and incomplete repetitions cannot reuse a passing result."""
from evals.harness import AgentResult, Case, CaseOutcome, Report, run_eval, run_once


def test_evaluation_resume_does_not_reuse_changed_case_labels(tmp_path):
    path = tmp_path / "eval.jsonl"
    agent = lambda inp: AgentResult(output=inp)
    grader = lambda case, result: case.expected == result.output
    assert run_eval(agent, [Case("x", 1, 1)], grader, n_runs=1, resume_file=path).mean_pass == 1
    assert run_eval(agent, [Case("x", 1, 2)], grader, n_runs=1, resume_file=path).mean_pass == 0
    partial = Report("partial", [CaseOutcome("x", True, 0, AgentResult(output=""))], 5)
    assert partial.pass_hat_k == 0


def test_a_transient_failure_is_retried_and_counted_but_a_real_crash_is_a_result():
    calls = []

    def stalls_once(_):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient: claude CLI timed out after 240s")
        return AgentResult(output="ok")

    result = run_once(stalls_once, {})
    assert result.output == "ok" and result.retries == 1 and len(calls) == 2

    def always_stalls(_):
        raise RuntimeError("transient: claude CLI timed out after 240s")

    assert "timed out" in run_once(always_stalls, {}).error

    def broken(_):
        calls.append(1)
        raise ValueError("format_error: bad reply")

    calls.clear()
    result = run_once(broken, {})
    assert "format_error" in result.error and result.retries == 0 and len(calls) == 1

#!/usr/bin/env bash
# Start Langfuse if it isn't up, then run the live end-to-end eval.
#   ./run_live_eval.sh                        all cases, 5 runs
#   ./run_live_eval.sh --cases 1 --n-runs 1   arguments go to evals/run_end_to_end.py
#   ./run_live_eval.sh compliance --suite redteam --n-runs 1   arguments go to evals/run_compliance.py --live
#   ./run_live_eval.sh cr07 cr14 e04 e05      chosen cases (cq, cr, ci or e); appends to evals/results/selected_cases_live.jsonl
set -euo pipefail
cd "$(dirname "$0")"

langfuse_up() { curl -fsS -m 3 http://localhost:3000/api/public/health >/dev/null 2>&1; }

start_langfuse() {
    docker info >/dev/null 2>&1 || { command -v orb >/dev/null && orb start; } || return 1
    docker compose up -d || return 1
    echo "Waiting for Langfuse..."
    for _ in {1..60}; do
        langfuse_up && return 0
        sleep 3
    done
    return 1
}

if langfuse_up || start_langfuse; then
    echo "Langfuse is up: http://localhost:3000"
else
    echo "Langfuse is not available: tracing is off for this run (local traces still go to runs/)." >&2
    export LANGFUSE_PUBLIC_KEY= LANGFUSE_SECRET_KEY=
fi

export RENOPILOT_FX="${RENOPILOT_FX:-snapshot}"
case "${1:-}" in
    c[qri][0-9]*|e[0-9]*)
        exec uv run python -c '
import json
import sys
import time
from evals import run_compliance as compliance, run_end_to_end as e2e
from evals.harness import load_cases
cases = {}
for module, agent, name in ((compliance, compliance.make_agent(True), "compliance_qa"),
                            (compliance, compliance.make_agent(True), "compliance_redteam"),
                            (compliance, compliance.make_agent(True), "compliance_injection"),
                            (e2e, e2e.make_agent("live"), "end_to_end")):
    for case in load_cases(f"evals/cases/{name}.jsonl"):
        cases[case.id] = (case, module.grade, agent)
ids = sys.argv[1:]
unknown = [i for i in ids if i not in cases]
if unknown:
    sys.exit(f"Unknown case ids: {unknown}")
failed = []
for i in ids:
    case, grade, agent = cases[i]
    result = None
    try:
        result = agent(case.input)
        passed = grade(case, result)
        detail = {k: result.output[k] for k in ("decision", "rule_id", "ids") if k in result.output}
        detail |= {"failures": result.failure_tags} if result.failure_tags else {}
    except Exception as e:
        passed, detail = False, f"{type(e).__name__}: {e}"
    print(i + ":", "PASS" if passed else "FAIL", detail or "", flush=True)
    with open("evals/results/selected_cases_live.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "case_id": i, "passed": passed, "detail": detail,
                            "run_id": result and result.run_id, "cost_usd": result and result.cost_usd,
                            "latency_s": result and result.latency_s}, default=str) + "\n")
    if not passed:
        failed.append(i)
print(f"{len(ids) - len(failed)}/{len(ids)} passed" + (f"; failed: {failed}" if failed else ""))
sys.exit(1 if failed else 0)' "$@" ;;
    compliance)
        shift
        exec uv run python evals/run_compliance.py --live "$@" ;;
esac
[ $# -gt 0 ] || set -- --n-runs 5
exec uv run python evals/run_end_to_end.py --mode live "$@"

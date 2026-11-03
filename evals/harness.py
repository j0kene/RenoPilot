"""Eval harness: the course's agentlab/evals.py contract, extended for this project.

Kept from week 3 (same names, same behaviour):
  Case, AgentResult, run_eval(agent_fn, cases, grader, n_runs), Report.render(baseline=...),
  flaky cases, failure clusters by tag, "improvement within one sd" warning.
Added:
  - pass^k (a case counts only if it passed in ALL k runs) next to pass@k
  - bootstrap 95% CI over cases
  - continuous metrics per run (e.g. recall@10) via AgentResult.metrics, with sd and CI
  - resumable runs: outcomes are appended to a JSONL file and skipped on restart
    (the CLI backend has rate limits; a crash at case 40 should not cost cases 1-39)
  - a transient backend failure (a stalled CLI, a rate limit) is retried, and the count is
    saved with the outcome: it says nothing about the agent, so it must not score as a wrong answer
"""
from __future__ import annotations

import json
import hashlib
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from renopilot.llm import is_transient

TRANSIENT_RETRIES = 2

Grader = Callable[["Case", "AgentResult"], bool]


@dataclass
class Case:
    id: str
    input: Any
    expected: Any = None
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    output: Any
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    failure_tags: list[str] = field(default_factory=list)   # set by the grader / adapter on a fail
    run_id: str | None = None                               # trace file in runs/
    retries: int = 0                                        # transient backend failures before this result


@dataclass
class CaseOutcome:
    case_id: str
    passed: bool
    run_index: int
    result: AgentResult
    tags: list[str] = field(default_factory=list)
    fingerprint: str = ""


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 621) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


@dataclass
class Report:
    name: str
    outcomes: list[CaseOutcome]
    n_runs: int

    # ---------------------------------------------------------------- pass rate
    @property
    def pass_rates(self) -> list[float]:
        by_run: dict[int, list[bool]] = {}
        for o in self.outcomes:
            by_run.setdefault(o.run_index, []).append(o.passed)
        return [sum(v) / len(v) for _, v in sorted(by_run.items())]

    @property
    def mean_pass(self) -> float:
        return statistics.fmean(self.pass_rates) if self.pass_rates else 0.0

    @property
    def stdev_pass(self) -> float:
        rs = self.pass_rates
        return statistics.stdev(rs) if len(rs) > 1 else 0.0

    def _by_case(self) -> dict[str, list[CaseOutcome]]:
        d: dict[str, list[CaseOutcome]] = {}
        for o in self.outcomes:
            d.setdefault(o.case_id, []).append(o)
        return d

    @property
    def pass_hat_k(self) -> float:
        """Share of cases that passed in every run: what production runs on."""
        cases = self._by_case()
        return sum(len(os_) == self.n_runs and {o.run_index for o in os_} == set(range(self.n_runs))
                   and all(o.passed for o in os_) for os_ in cases.values()) / len(cases) if cases else 0.0

    @property
    def pass_at_k(self) -> float:
        """Share of cases that passed in at least one run: what demos are sold on."""
        cases = self._by_case()
        return sum(any(o.passed for o in os_) for os_ in cases.values()) / len(cases) if cases else 0.0

    def case_pass_ci(self) -> tuple[float, float]:
        return bootstrap_ci([statistics.fmean(o.passed for o in os_) for os_ in self._by_case().values()])

    # ---------------------------------------------------------------- metrics
    def metric_names(self) -> list[str]:
        return sorted({k for o in self.outcomes for k in o.result.metrics})

    def metric(self, name: str) -> dict[str, float]:
        """mean over runs, sd across run means, and a bootstrap CI over case means."""
        by_run: dict[int, list[float]] = {}
        by_case: dict[str, list[float]] = {}
        for o in self.outcomes:
            if name in o.result.metrics:
                by_run.setdefault(o.run_index, []).append(o.result.metrics[name])
                by_case.setdefault(o.case_id, []).append(o.result.metrics[name])
        run_means = [statistics.fmean(v) for _, v in sorted(by_run.items())]
        lo, hi = bootstrap_ci([statistics.fmean(v) for v in by_case.values()])
        return {"mean": statistics.fmean(run_means) if run_means else 0.0,
                "sd": statistics.stdev(run_means) if len(run_means) > 1 else 0.0, "ci_lo": lo, "ci_hi": hi}

    # ---------------------------------------------------------------- cost
    @property
    def cost_per_task(self) -> float:
        return sum(o.result.cost_usd for o in self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    @property
    def mean_latency(self) -> float:
        return statistics.fmean(o.result.latency_s for o in self.outcomes) if self.outcomes else 0.0

    def flaky_cases(self) -> list[str]:
        return sorted(cid for cid, os_ in self._by_case().items() if len({o.passed for o in os_}) > 1)

    def clusters(self) -> dict[str, tuple[int, int]]:
        """tag -> (cases that failed in any run, cases with this tag). Failure tags set by
        the grader are counted too, so the report shows WHY, not just where."""
        verdict: dict[str, bool] = {}
        tags: dict[str, set[str]] = {}
        for o in self.outcomes:
            verdict[o.case_id] = verdict.get(o.case_id, True) and o.passed
            tags.setdefault(o.case_id, set()).update(o.tags or ["untagged"])
            if not o.passed:
                tags[o.case_id].update(f"fail:{t}" for t in o.result.failure_tags)
        agg: dict[str, list[int]] = {}
        for cid, passed in verdict.items():
            for t in tags[cid]:
                a = agg.setdefault(t, [0, 0])
                a[1] += 1
                a[0] += 0 if passed else 1
        return {t: (f, n) for t, (f, n) in agg.items()}

    def render(self, baseline: "Report | None" = None) -> str:
        lo, hi = self.case_pass_ci()
        lines = [f"=== {self.name} ===",
                 f"pass rate     {self.mean_pass:.1%}  (sd {self.stdev_pass:.1%} over {self.n_runs} runs; "
                 f"95% CI over cases {lo:.1%}-{hi:.1%})",
                 f"pass^{self.n_runs}        {self.pass_hat_k:.1%}   pass@{self.n_runs} {self.pass_at_k:.1%}"]
        for m in self.metric_names():
            s = self.metric(m)
            lines.append(f"{m:<13} {s['mean']:.3f}  (sd {s['sd']:.3f}; 95% CI {s['ci_lo']:.3f}-{s['ci_hi']:.3f})")
        lines += [f"cost/task     ${self.cost_per_task:.4f} (API-equivalent)",
                  f"latency/task  {self.mean_latency:.2f}s"]
        if baseline:
            delta = self.mean_pass - baseline.mean_pass
            lines.append(f"vs {baseline.name}: {delta:+.1%} pass rate")
            for m in self.metric_names():
                if m in baseline.metric_names():
                    dm = self.metric(m)["mean"] - baseline.metric(m)["mean"]
                    lines.append(f"   {m}: {dm:+.3f}")
            if abs(delta) <= max(self.stdev_pass, baseline.stdev_pass) and delta != 0:
                lines.append("  NOTE: difference is within one standard deviation. You have not shown an improvement.")
        flaky = self.flaky_cases()
        if flaky:
            lines.append(f"flaky cases   {', '.join(flaky[:15])}{' ...' if len(flaky) > 15 else ''}")
        lines.append("failure clusters:")
        for tag, (f, n) in sorted(self.clusters().items(), key=lambda kv: -kv[1][0]):
            if f:
                lines.append(f"  {tag:<28} {f}/{n} cases failed")
        return "\n".join(lines)


def evaluation_fingerprints(cases: list[Case], revision: str = "") -> dict[str, str]:
    # Resume only matching code and inputs. Old successful runs must not grade new code.
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(revision.encode())
    for folder in ("renopilot", "evals"):
        for source in sorted((root / folder).rglob("*.py")):
            digest.update(str(source.relative_to(root)).encode() + source.read_bytes())
    for pattern in ("data/*.csv", "data/pricing/*.json", "data/compliance/*.json*"):
        for source in sorted(root.glob(pattern)):
            digest.update(str(source.relative_to(root)).encode() + source.read_bytes())
    return {c.id: hashlib.sha256((digest.hexdigest() + json.dumps(
        [c.input, c.expected], sort_keys=True, default=str)).encode()).hexdigest() for c in cases}


def run_once(agent_fn: Callable[[Any], AgentResult], inp: Any) -> AgentResult:
    """One run of one case. Anything but a transient failure is the agent's result: a crash is a result."""
    for attempt in range(TRANSIENT_RETRIES + 1):
        try:
            result = agent_fn(inp)
            result.retries = attempt
            return result
        except Exception as e:                               # noqa: BLE001
            if attempt == TRANSIENT_RETRIES or not is_transient(e):
                return AgentResult(output="", error=f"{type(e).__name__}: {e}", retries=attempt)
            print(f"  transient failure, retrying: {str(e)[:80]}", flush=True)
    raise AssertionError("unreachable")


def run_eval(agent_fn: Callable[[Any], AgentResult], cases: list[Case], grader: Grader,
             n_runs: int = 3, name: str = "agent", resume_file: str | Path | None = None,
             revision: str = "") -> Report:
    if n_runs < 1 or not cases:
        raise ValueError("An evaluation needs cases and at least one run.")
    fingerprints = evaluation_fingerprints(cases, revision)
    done: dict[tuple[str, int], CaseOutcome] = {}
    path = Path(resume_file) if resume_file else None
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d.get("fingerprint") != fingerprints.get(d["case_id"]):
                continue
            done[(d["case_id"], d["run_index"])] = CaseOutcome(
                d["case_id"], d["passed"], d["run_index"], AgentResult(**d["result"]), d["tags"], d["fingerprint"])
    outcomes: list[CaseOutcome] = []
    for run_index in range(n_runs):
        for case in cases:
            if (case.id, run_index) in done:
                outcomes.append(done[(case.id, run_index)])
                continue
            result = run_once(agent_fn, case.input)
            passed = False if result.error else bool(grader(case, result))
            tags = list(case.tags) + (["crash"] if result.error else [])
            o = CaseOutcome(case.id, passed, run_index, result, tags, fingerprints[case.id])
            outcomes.append(o)
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(o), default=lambda v: v.model_dump(mode="json")
                        if hasattr(v, "model_dump") else sorted(v) if isinstance(v, set) else str(v)) + "\n")
            print(f"  [{name}] run {run_index + 1}/{n_runs} {case.id}: {'PASS' if passed else 'FAIL'}"
                  f"{'  ' + result.error[:80] if result.error else ''}", flush=True)
    return Report(name=name, outcomes=outcomes, n_runs=n_runs)


def load_cases(path: str | Path) -> list[Case]:
    return [Case(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

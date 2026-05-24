from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trace2eval.generation import generate_eval_case
from trace2eval.io import load_failure_hypotheses, load_normalized_traces, load_raw_traces
from trace2eval.mining import extract_causal_slice, mine_trace, rank_hypotheses
from trace2eval.normalize import normalize_trace
from trace2eval.runner import run_eval
from trace2eval.schemas import EvalCase, FailureHypothesis, NormalizedTrace, RunResult


def load_replay_trace(path: Path, trace_id: str | None = None) -> NormalizedTrace:
    traces = load_any_normalized_traces(path)
    if trace_id:
        matches = [trace for trace in traces if trace.trace_id == trace_id]
        if not matches:
            raise ValueError(f"No trace with trace_id={trace_id!r} found in {path}")
        return matches[0]
    if len(traces) != 1:
        raise ValueError(f"Expected one trace in {path}; found {len(traces)}. Use --trace-id to select one.")
    return traces[0]


def load_any_normalized_traces(path: Path) -> list[NormalizedTrace]:
    try:
        return load_normalized_traces(path)
    except Exception:
        return [normalize_trace(trace) for trace in load_raw_traces(path)]


def build_replay(
    trace: NormalizedTrace,
    failure_selector: str = "primary",
    failures_path: Path | None = None,
) -> tuple[FailureHypothesis, EvalCase, RunResult]:
    failures = load_failure_hypotheses(failures_path) if failures_path else mine_trace(trace)
    failures = [failure for failure in failures if failure.trace_id == trace.trace_id]
    ranked = rank_hypotheses(trace, failures)
    if not ranked:
        raise ValueError(f"No failure hypotheses found for trace {trace.trace_id}")
    failure = select_failure(ranked, failure_selector)
    eval_case = generate_eval_case(extract_causal_slice(trace, failure))
    result = run_eval(eval_case, trace)
    return failure, eval_case, result


def select_failure(failures: list[FailureHypothesis], selector: str) -> FailureHypothesis:
    if selector in {"primary", "top", ""}:
        return failures[0]
    for failure in failures:
        if failure.failure_type == selector or str(failure.onset_step_id) == selector:
            return failure
        if f"{failure.failure_type}@{failure.onset_step_id}" == selector:
            return failure
    available = ", ".join(f"{failure.failure_type}@{failure.onset_step_id}" for failure in failures)
    raise ValueError(f"No failure matching {selector!r}. Available: {available}")


def print_replay_story(
    trace: NormalizedTrace,
    failure: FailureHypothesis,
    eval_case: EvalCase,
    result: RunResult,
    *,
    console: Console | None = None,
) -> None:
    console = console or Console()
    causal_slice = eval_case.metadata.get("causal_slice") or {}
    console.print(Panel.fit(trace.trace_id, title="Trace Replay"))
    console.print(f"[bold]Task[/bold]\n{trace.task.description or trace.task.prompt or 'Unknown task.'}")
    console.print(f"\n[bold]Failure[/bold]\n{failure.failure_type} at step {failure.onset_step_id}")

    observations = causal_slice.get("previous_observations") or []
    if observations:
        table = Table(title="Key Observations", show_header=False)
        table.add_column("Observation")
        for observation in observations[:5]:
            table.add_row(str(observation))
        console.print(table)

    console.print(f"\n[bold]First Bad Action[/bold]\n{causal_slice.get('bad_action_summary') or 'Unknown.'}")
    console.print(f"\n[bold]Expected Behavior[/bold]\n{causal_slice.get('expected_behavior') or eval_case.verifier.description or 'Unknown.'}")

    eval_table = Table(title="Generated Eval")
    eval_table.add_column("Field")
    eval_table.add_column("Value")
    eval_table.add_row("Eval", eval_case.eval_id)
    eval_table.add_row("Rule", eval_case.verifier.rule)
    eval_table.add_row("Failure criteria", "\n".join(eval_case.failure_criteria))
    eval_table.add_row("Success criteria", "\n".join(eval_case.success_criteria))
    console.print(eval_table)

    replay_status = "failed as expected" if not result.passed else "passed (failure did not reproduce)"
    replay_table = Table(title="Replay Result")
    replay_table.add_column("Passed")
    replay_table.add_column("Status")
    replay_table.add_column("Evidence")
    replay_table.add_row("yes" if result.passed else "no", replay_status, "\n".join(result.evidence))
    console.print(replay_table)

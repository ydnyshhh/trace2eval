from __future__ import annotations

from collections import Counter, defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trace2eval.schemas import FailureHypothesis, NormalizedTrace, Report, RunResult


def build_report(
    traces: list[NormalizedTrace],
    failures: list[FailureHypothesis],
    run_results: list[RunResult] | None = None,
) -> Report:
    top_failure_types = Counter(item.failure_type for item in failures)
    successful = sum(1 for trace in traces if trace.outcome.success is True)
    failed = sum(1 for trace in traces if trace.outcome.success is False)
    examples = [
        {
            "trace_id": failure.trace_id,
            "failure_type": failure.failure_type,
            "onset_step_id": failure.onset_step_id,
            "evidence": failure.evidence[:2],
        }
        for failure in failures[:5]
    ]
    return Report(
        total_traces=len(traces),
        successful_traces=successful,
        failed_traces=failed,
        unknown_outcome_traces=len(traces) - successful - failed,
        hypothesis_count=len(failures),
        top_failure_types=dict(top_failure_types.most_common()),
        generated_eval_count=len({result.eval_id for result in run_results or []}),
        examples=examples,
    )


def print_terminal_report(
    traces: list[NormalizedTrace],
    failures: list[FailureHypothesis],
    *,
    console: Console | None = None,
) -> None:
    console = console or Console()
    report = build_report(traces, failures)
    summary = Table(title="Trace2Eval Summary")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Total traces", str(report.total_traces))
    summary.add_row("Known successful", str(report.successful_traces))
    summary.add_row("Known failed", str(report.failed_traces))
    summary.add_row("Unknown outcome", str(report.unknown_outcome_traces))
    summary.add_row("Failure hypotheses", str(report.hypothesis_count))
    console.print(summary)

    failure_table = Table(title="Top Failure Types")
    failure_table.add_column("Failure type")
    failure_table.add_column("Count", justify="right")
    for failure_type, count in report.top_failure_types.items():
        failure_table.add_row(failure_type, str(count))
    if not report.top_failure_types:
        failure_table.add_row("none", "0")
    console.print(failure_table)

    failures_by_trace: dict[str, list[FailureHypothesis]] = defaultdict(list)
    for failure in failures:
        failures_by_trace[failure.trace_id].append(failure)

    for trace in traces:
        print_trace_timeline(trace, failures_by_trace.get(trace.trace_id, []), console=console)


def print_trace_timeline(
    trace: NormalizedTrace,
    failures: list[FailureHypothesis] | None = None,
    *,
    console: Console | None = None,
) -> None:
    console = console or Console()
    console.print(Panel.fit(trace.trace_id, title="Trace"))
    timeline = Table()
    timeline.add_column("Step", justify="right")
    timeline.add_column("Action")
    timeline.add_column("Phase")
    timeline.add_column("Target / command")
    timeline.add_column("Err", justify="center")
    timeline.add_column("Failures")
    markers = defaultdict(list)
    for failure in failures or []:
        markers[failure.onset_step_id].append(failure.failure_type)
    for step in trace.steps:
        target = step.command or step.target or step.raw_action or ""
        if len(target) > 90:
            target = target[:87] + "..."
        timeline.add_row(
            str(step.step_id),
            step.action_type.value,
            step.phase.value,
            target,
            "yes" if step.is_error else "",
            ", ".join(markers.get(step.step_id, [])),
        )
    console.print(timeline)

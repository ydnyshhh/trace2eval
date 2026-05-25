from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from trace2eval.adapters import (
    ClaudeCodeHeadlessJSONAdapter,
    ClaudeCodeHookJSONLAdapter,
    CodexJSONLAdapter,
    GenericJSONAdapter,
)
from trace2eval.benchmark import run_benchmark
from trace2eval.capture import (
    build_claude_settings_snippet,
    discover_codex_rollouts,
    install_claude_hook,
)
from trace2eval.counterfactual import print_counterfactual_replay, run_counterfactual_replay
from trace2eval.doctor import run_doctor_checks
from trace2eval.generation import generate_eval_cases
from trace2eval.io import (
    ensure_dir,
    load_eval_cases,
    load_failure_hypotheses,
    load_normalized_traces,
    load_raw_traces,
    slugify,
    write_json,
    write_jsonl,
    write_yaml,
)
from trace2eval.mining import mine_trace, rank_hypotheses
from trace2eval.normalize import normalize_trace
from trace2eval.replay import build_replay, load_replay_trace, print_replay_story
from trace2eval.report import build_report, print_terminal_report, print_trace_timeline
from trace2eval.runner import run_evals
from trace2eval.storage.duckdb_store import (
    IndexSummary,
    build_duckdb_index,
    query_action_mix,
    query_by_agent,
    query_by_source,
    query_error_summary,
    query_eval_results,
    query_failure_onsets,
    query_failure_recurrence,
    query_failure_type,
    query_top_failures,
    query_trace,
)

console = Console()
app = typer.Typer(help="Trace-mined regression evals for coding agents.")
init_app = typer.Typer(help="Initialize Trace2Eval workspaces and capture integrations.", no_args_is_help=False)
capture_app = typer.Typer(help="Capture and discover trace inputs.")
ingest_app = typer.Typer(help="Ingest raw agent trace formats.")


app.add_typer(init_app, name="init")
app.add_typer(capture_app, name="capture")
app.add_typer(ingest_app, name="ingest")


@init_app.callback(invoke_without_command=True)
def init_callback(
    ctx: typer.Context,
    directory: Annotated[Path, typer.Option("--dir", "-d", help="Trace2Eval workspace directory.")] = Path(".trace2eval"),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    initialize_workspace(directory)
    console.print(f"Initialized Trace2Eval workspace at [bold]{directory}[/bold]")


@init_app.command("claude-code-hooks")
def init_claude_hooks(
    out: Annotated[Path, typer.Option("--out", help="Directory for the generated Claude hook logger.")] = Path(".trace2eval/hooks"),
    log_path: Annotated[Path | None, typer.Option("--log-path", help="Optional TRACE2EVAL_LOG_PATH value for settings snippet.")] = None,
) -> None:
    initialize_workspace(Path(".trace2eval"))
    script_path = install_claude_hook(out)
    ensure_dir(Path(".trace2eval") / "claude-code")
    snippet = build_claude_settings_snippet(script_path, log_path)
    console.print(f"Generated Claude Code hook logger: [bold]{script_path}[/bold]")
    console.print("\nAdd a hooks entry like this to your Claude Code settings, adjusting matchers if needed:\n")
    console.print(snippet)
    console.print(
        "\nClaude Code sends event-specific JSON to hooks over stdin. "
        "The logger defaults to observe-only behavior and exits successfully even if logging fails."
    )


@capture_app.command("codex-discover")
def codex_discover(
    codex_home: Annotated[Path | None, typer.Option("--codex-home", help="Explicit Codex home directory.")] = None,
) -> None:
    files = discover_codex_rollouts(codex_home)
    if not files:
        console.print("No Codex rollout JSONL files found.")
        raise typer.Exit(0)
    table = Table(title="Discovered Codex Rollouts")
    table.add_column("Path")
    for file in files:
        table.add_row(str(file))
    console.print(table)


@ingest_app.command("codex")
def ingest_codex(
    path: Annotated[Path, typer.Option("--path", help="Codex rollout JSONL file or directory.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory for canonical RawTrace JSON.")] = Path(".trace2eval/raw"),
) -> None:
    write_raw_traces(CodexJSONLAdapter().ingest(path), out)


@ingest_app.command("claude-hooks")
def ingest_claude_hooks(
    path: Annotated[Path, typer.Option("--path", help="Claude Code hook JSONL file or directory.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory for canonical RawTrace JSON.")] = Path(".trace2eval/raw"),
) -> None:
    write_raw_traces(ClaudeCodeHookJSONLAdapter().ingest(path), out)


@ingest_app.command("claude-headless")
def ingest_claude_headless(
    path: Annotated[Path, typer.Option("--path", help="Claude Code headless JSON file or directory.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory for canonical RawTrace JSON.")] = Path(".trace2eval/raw"),
) -> None:
    write_raw_traces(ClaudeCodeHeadlessJSONAdapter().ingest(path), out)


@ingest_app.command("generic")
def ingest_generic(
    path: Annotated[Path, typer.Option("--path", help="Canonical RawTrace JSON file or directory.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory for canonical RawTrace JSON.")] = Path(".trace2eval/raw"),
) -> None:
    write_raw_traces(GenericJSONAdapter().ingest(path), out)


@app.command("normalize")
def normalize_command(
    input_path: Annotated[Path, typer.Option("--input", help="RawTrace JSON file or directory.")] = Path(".trace2eval/raw"),
    out: Annotated[Path, typer.Option("--out", help="Output directory for NormalizedTrace JSON.")] = Path(".trace2eval/normalized"),
) -> None:
    raw_traces = load_raw_traces(input_path)
    normalized = [normalize_trace(trace) for trace in raw_traces]
    ensure_dir(out)
    for trace in normalized:
        write_json(out / f"{slugify(trace.trace_id)}.json", trace)
    console.print(f"Normalized {len(normalized)} trace(s) into {out}")


@app.command("mine")
def mine_command(
    input_path: Annotated[Path, typer.Option("--input", help="NormalizedTrace JSON file or directory.")] = Path(".trace2eval/normalized"),
    out: Annotated[Path, typer.Option("--out", help="Output JSONL file for failure hypotheses.")] = Path(".trace2eval/reports/failures.jsonl"),
) -> None:
    traces = load_normalized_traces(input_path)
    failures = []
    for trace in traces:
        failures.extend(mine_trace(trace))
    write_jsonl(out, failures)
    console.print(f"Mined {len(failures)} failure hypothesis/hypotheses into {out}")


@app.command("generate")
def generate_command(
    traces_path: Annotated[Path, typer.Option("--traces", help="NormalizedTrace JSON file or directory.")] = Path(".trace2eval/normalized"),
    failures_path: Annotated[Path, typer.Option("--failures", help="Failure hypotheses JSONL.")] = Path(".trace2eval/reports/failures.jsonl"),
    out: Annotated[Path, typer.Option("--out", help="Output directory for EvalCase YAML.")] = Path(".trace2eval/evals"),
    all_failures: Annotated[bool, typer.Option("--all", help="Generate an eval for every hypothesis instead of only the top one per trace.")] = False,
) -> None:
    traces = load_normalized_traces(traces_path)
    failures = load_failure_hypotheses(failures_path)
    by_trace = {trace.trace_id: trace for trace in traces}
    ranked = []
    for trace_id in sorted({failure.trace_id for failure in failures}):
        trace = by_trace.get(trace_id)
        items = [failure for failure in failures if failure.trace_id == trace_id]
        ranked.extend(rank_hypotheses(trace, items) if trace else items)
    evals = generate_eval_cases(traces, ranked, top_only=not all_failures)
    ensure_dir(out)
    for eval_case in evals:
        write_yaml(out / f"{slugify(eval_case.eval_id)}.yaml", eval_case)
    console.print(f"Generated {len(evals)} eval case(s) into {out}")


@app.command("run")
def run_command(
    evals_path: Annotated[Path, typer.Option("--evals", help="EvalCase YAML/JSON file or directory.")] = Path(".trace2eval/evals"),
    traces_path: Annotated[Path, typer.Option("--traces", help="NormalizedTrace JSON file or directory.")] = Path(".trace2eval/normalized"),
    out: Annotated[Path, typer.Option("--out", help="Output JSONL file for eval run results.")] = Path(".trace2eval/reports/eval_results.jsonl"),
    mode: Annotated[str, typer.Option("--mode", help="Run mode: source, task, or suite.")] = "suite",
) -> None:
    evals = load_eval_cases(evals_path)
    traces = load_normalized_traces(traces_path)
    results = run_evals(evals, traces, mode=mode)
    write_jsonl(out, results)
    passed = sum(1 for result in results if result.passed)
    console.print(f"Ran {len(results)} eval/trace checks: {passed} passed, {len(results) - passed} failed. Wrote {out}")


@app.command("inspect")
def inspect_command(
    input_path: Annotated[Path, typer.Option("--input", help="RawTrace or NormalizedTrace JSON file/directory.")],
    failures_path: Annotated[Path | None, typer.Option("--failures", help="Optional failure hypotheses JSONL for markers.")] = None,
) -> None:
    traces = load_any_normalized_traces(input_path)
    failures = load_failure_hypotheses(failures_path) if failures_path else []
    if not failures:
        for trace in traces:
            failures.extend(mine_trace(trace))
    for trace in traces:
        print_trace_timeline(trace, [failure for failure in failures if failure.trace_id == trace.trace_id], console=console)


@app.command("validate")
def validate_command(
    examples: Annotated[Path, typer.Option("--examples", help="Canonical example RawTrace directory.")] = Path("examples/traces"),
    positive: Annotated[Path | None, typer.Option("--positive", help="Optional passing RawTrace fixture file/directory that should pass matching generated evals.")] = None,
) -> None:
    raw_traces = GenericJSONAdapter().ingest(examples)
    normalized = [normalize_trace(trace) for trace in raw_traces]
    failures = []
    for trace in normalized:
        failures.extend(mine_trace(trace))
    evals = generate_eval_cases(normalized, failures)
    results = run_evals(evals, normalized, mode="source")
    failed_as_expected = sum(1 for result in results if not result.passed)
    positive_results = []
    if positive:
        positive_traces = [normalize_trace(trace) for trace in GenericJSONAdapter().ingest(positive)]
        positive_results = run_evals(evals, positive_traces, mode="task")
    console.print(
        f"Validation: {len(raw_traces)} traces, {len(failures)} hypotheses, "
        f"{len(evals)} evals, {len(results)} source replay(s), {failed_as_expected} failed as expected."
    )
    if positive:
        passed_positive = sum(1 for result in positive_results if result.passed)
        console.print(f"Positive validation: {passed_positive}/{len(positive_results)} matching replay(s) passed.")
    if not raw_traces or not failures or not evals or not results or failed_as_expected != len(results):
        raise typer.Exit(1)
    if positive and (not positive_results or any(not result.passed for result in positive_results)):
        raise typer.Exit(1)


@app.command("doctor")
def doctor_command(
    workspace: Annotated[Path, typer.Option("--workspace", help="Trace2Eval workspace directory.")] = Path(".trace2eval"),
    examples: Annotated[Path, typer.Option("--examples", help="Example RawTrace directory to validate.")] = Path("examples/traces"),
    benchmark_fixtures: Annotated[Path, typer.Option("--benchmark-fixtures", help="Real-run benchmark fixture directory.")] = Path("examples/real_runs"),
    codex_home: Annotated[Path | None, typer.Option("--codex-home", help="Explicit Codex home directory.")] = None,
    strict: Annotated[bool, typer.Option("--strict/--no-strict", help="Exit non-zero when a check fails.")] = False,
) -> None:
    checks = run_doctor_checks(
        workspace=workspace,
        examples=examples,
        benchmark_fixtures=benchmark_fixtures,
        codex_home=codex_home,
    )
    console.print("[bold]Trace2Eval Doctor[/bold]")
    for item in checks:
        console.print(f"{doctor_symbol(item['status'])} {item['message']}")
    if strict and any(item["status"] == "fail" for item in checks):
        raise typer.Exit(1)


@app.command("replay")
def replay_command(
    trace_path: Annotated[Path, typer.Option("--trace", help="RawTrace or NormalizedTrace JSON file/directory.")],
    failure: Annotated[str, typer.Option("--failure", help="Failure selector: primary, failure_type, step id, or type@step.")] = "primary",
    failures_path: Annotated[Path | None, typer.Option("--failures", help="Optional failure hypotheses JSONL.")] = None,
    trace_id: Annotated[str | None, typer.Option("--trace-id", help="Trace id when --trace points to a directory.")] = None,
) -> None:
    try:
        trace = load_replay_trace(trace_path, trace_id)
        selected_failure, eval_case, result = build_replay(trace, failure, failures_path)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    print_replay_story(trace, selected_failure, eval_case, result, console=console)


@app.command("counterfactual")
def counterfactual_command(
    trace_path: Annotated[Path, typer.Option("--trace", help="RawTrace or NormalizedTrace JSON file/directory.")],
    failures_path: Annotated[Path | None, typer.Option("--failures", help="Optional failure hypotheses JSONL.")] = None,
    failure: Annotated[str, typer.Option("--failure", help="Failure selector: primary, failure_type, step id, or type@step.")] = "primary",
    trace_id: Annotated[str | None, typer.Option("--trace-id", help="Trace id when --trace points to a directory.")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Optional JSON/YAML output file or directory.")] = None,
) -> None:
    try:
        trace = load_replay_trace(trace_path, trace_id)
        failures = load_failure_hypotheses(failures_path) if failures_path else None
        replay = run_counterfactual_replay(trace, failures, failure_selector=failure)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    output_path = write_counterfactual_output(replay, out) if out else None
    print_counterfactual_replay(replay, console=console, output_path=output_path)


@app.command("index")
def index_command(
    traces_path: Annotated[Path, typer.Option("--traces", help="RawTrace or NormalizedTrace JSON file or directory.")],
    failures_path: Annotated[Path | None, typer.Option("--failures", help="Optional failure hypotheses JSONL.")] = None,
    evals_path: Annotated[Path | None, typer.Option("--evals", help="Optional EvalCase YAML/JSON file or directory.")] = None,
    runs_path: Annotated[Path | None, typer.Option("--runs", help="Optional eval run results JSONL.")] = None,
    out: Annotated[Path, typer.Option("--out", help="Output DuckDB database path.")] = Path(".trace2eval/trace2eval.duckdb"),
) -> None:
    summary = build_duckdb_index(
        traces_path=traces_path,
        failures_path=failures_path,
        evals_path=evals_path,
        runs_path=runs_path,
        out_path=out,
    )
    print_index_summary(summary)


@app.command("query")
def query_command(
    db_path: Annotated[Path, typer.Option("--db", help="DuckDB index path.")] = Path(".trace2eval/trace2eval.duckdb"),
    top_failures: Annotated[bool, typer.Option("--top-failures", help="Show failure counts and average scores.")] = False,
    eval_results: Annotated[bool, typer.Option("--eval-results", help="Show eval recurrence by failure type.")] = False,
    failure_recurrence: Annotated[bool, typer.Option("--failure-recurrence", help="Show failed/passed eval runs by failure type.")] = False,
    failure_onsets: Annotated[bool, typer.Option("--failure-onsets", help="Show failure onset action/phase distribution.")] = False,
    action_mix: Annotated[bool, typer.Option("--action-mix", help="Show action/error percentages by source and agent.")] = False,
    error_summary: Annotated[bool, typer.Option("--error-summary", help="Show trace-level error and failure counts.")] = False,
    by_source: Annotated[bool, typer.Option("--by-source", help="Group traces, failures, and evals by source.")] = False,
    by_agent: Annotated[bool, typer.Option("--by-agent", help="Group traces and failures by agent/model.")] = False,
    trace_id: Annotated[str | None, typer.Option("--trace", help="Show one trace timeline and failures.")] = None,
    failure_type: Annotated[str | None, typer.Option("--failure-type", help="List traces with this failure type.")] = None,
) -> None:
    if not db_path.exists():
        console.print(f"[red]DuckDB index not found: {db_path}[/red]")
        console.print("Run: trace2eval index --traces .trace2eval/normalized --out .trace2eval/trace2eval.duckdb")
        raise typer.Exit(1)

    selected = sum(
        bool(item)
        for item in (
            top_failures,
            eval_results,
            failure_recurrence,
            failure_onsets,
            action_mix,
            error_summary,
            by_source,
            by_agent,
            trace_id,
            failure_type,
        )
    )
    if selected == 0:
        print_query_help()
        return
    if selected > 1:
        console.print("[red]Choose exactly one query mode.[/red]")
        print_query_help()
        raise typer.Exit(1)

    if top_failures:
        rows = query_top_failures(db_path)
        print_rows("Top Failures", rows, ("failure_type", "count", "average_confidence", "average_severity"))
        if not rows:
            console.print("[yellow]No failures indexed. Re-run index with --failures to populate this query.[/yellow]")
    elif eval_results:
        print_rows(
            "Eval Results",
            query_eval_results(db_path),
            ("failure_type", "eval_count", "failed_runs", "passed_runs", "failure_recurrence_rate"),
        )
    elif failure_recurrence:
        print_rows(
            "Failure Recurrence",
            query_failure_recurrence(db_path),
            ("failure_type", "eval_count", "failed_runs", "passed_runs", "failure_recurrence_rate"),
        )
    elif failure_onsets:
        print_rows(
            "Failure Onsets",
            query_failure_onsets(db_path),
            ("failure_type", "onset_action_type", "onset_phase", "count", "avg_confidence"),
        )
    elif action_mix:
        print_rows(
            "Action Mix",
            query_action_mix(db_path),
            ("source", "agent_name", "read_percent", "search_percent", "edit_percent", "verify_percent", "error_percent"),
        )
    elif error_summary:
        print_rows(
            "Error Summary",
            query_error_summary(db_path),
            ("trace_id", "error_steps", "verify_errors", "final_success", "failure_count"),
        )
    elif by_source:
        print_rows("By Source", query_by_source(db_path), ("source", "trace_count", "failure_count", "eval_count"))
    elif by_agent:
        print_rows("By Agent", query_by_agent(db_path), ("agent_name", "model_name", "trace_count", "failure_count"))
    elif trace_id:
        result = query_trace(db_path, trace_id)
        if result.trace is None:
            console.print(f"[red]Trace not found: {trace_id}[/red]")
            raise typer.Exit(1)
        print_trace_query_result(result.trace, result.steps, result.failures)
    elif failure_type:
        print_rows(
            f"Failure Type: {failure_type}",
            query_failure_type(db_path, failure_type),
            ("trace_id", "source", "agent_name", "model_name", "onset_step_id", "confidence", "severity", "causal_role"),
        )


@app.command("benchmark")
def benchmark_command(
    fixtures: Annotated[Path, typer.Option("--fixtures", help="Directory containing benchmark case YAML notes.")] = Path("examples/real_runs"),
    json_out: Annotated[Path | None, typer.Option("--json-out", help="Optional JSON output path for benchmark results.")] = None,
    strict: Annotated[bool, typer.Option("--strict/--no-strict", help="Fail with a non-zero exit code when any benchmark case misses.")] = True,
) -> None:
    results = run_benchmark(fixtures)
    table = Table(title="Trace2Eval Real-Run Benchmark")
    table.add_column("Case")
    table.add_column("Agent")
    table.add_column("Expected")
    table.add_column("Primary Detected")
    table.add_column("Match")
    for result in results:
        table.add_row(
            result["case_id"],
            result.get("agent_used") or "",
            result["expected_failure_type"],
            result.get("primary_detected_failure_type") or "none",
            "yes" if result["matched"] else "no",
        )
    console.print(table)
    matched = sum(1 for result in results if result["matched"])
    console.print(f"Benchmark accuracy: {matched}/{len(results)}")
    if json_out:
        write_json(json_out, results)
        console.print(f"Wrote benchmark JSON to {json_out}")
    if strict and (not results or matched != len(results)):
        raise typer.Exit(1)


@app.command("report")
def report_command(
    traces_path: Annotated[Path, typer.Option("--traces", help="NormalizedTrace JSON file or directory.")] = Path(".trace2eval/normalized"),
    failures_path: Annotated[Path, typer.Option("--failures", help="Failure hypotheses JSONL.")] = Path(".trace2eval/reports/failures.jsonl"),
    json_out: Annotated[Path | None, typer.Option("--json-out", help="Optional JSON summary output.")] = None,
) -> None:
    traces = load_normalized_traces(traces_path)
    failures = load_failure_hypotheses(failures_path)
    print_terminal_report(traces, failures, console=console)
    if json_out:
        write_json(json_out, build_report(traces, failures))
        console.print(f"Wrote JSON report to {json_out}")


def initialize_workspace(directory: Path) -> None:
    for name in ("raw", "normalized", "evals", "reports", "hooks"):
        ensure_dir(directory / name)


def write_raw_traces(traces: list, out: Path) -> None:
    ensure_dir(out)
    for trace in traces:
        write_json(out / f"{slugify(trace.trace_id)}.json", trace)
    console.print(f"Wrote {len(traces)} RawTrace file(s) into {out}")


def load_any_normalized_traces(path: Path):
    try:
        return load_normalized_traces(path)
    except Exception:
        return [normalize_trace(trace) for trace in load_raw_traces(path)]


def write_counterfactual_output(replay, out: Path) -> Path:
    output_path = out
    if out.suffix.lower() not in {".json", ".yaml", ".yml"}:
        ensure_dir(out)
        output_path = out / f"{slugify(replay.counterfactual_trace.trace_id)}.json"
    if output_path.suffix.lower() in {".yaml", ".yml"}:
        write_yaml(output_path, replay)
    else:
        write_json(output_path, replay)
    return output_path


def print_index_summary(summary: IndexSummary) -> None:
    console.print("[bold]Indexed Trace2Eval corpus[/bold]")
    table = Table(show_header=False)
    table.add_column("Artifact")
    table.add_column("Count", justify="right")
    table.add_row("traces", str(summary.traces))
    table.add_row("steps", str(summary.steps))
    table.add_row("failures", str(summary.failures))
    table.add_row("evals", str(summary.evals))
    table.add_row("runs", str(summary.runs))
    console.print(table)
    for warning in summary.warnings:
        console.print(f"[yellow]WARN[/yellow] {warning}")
    console.print(f"database: {summary.database}")


def print_query_help() -> None:
    console.print("[bold]Available query modes[/bold]")
    console.print("  --top-failures")
    console.print("  --eval-results")
    console.print("  --failure-recurrence")
    console.print("  --failure-onsets")
    console.print("  --action-mix")
    console.print("  --error-summary")
    console.print("  --by-source")
    console.print("  --by-agent")
    console.print("  --trace TRACE_ID")
    console.print("  --failure-type FAILURE_TYPE")


def print_rows(title: str, rows: list[dict], columns: tuple[str, ...]) -> None:
    table = Table(title=title)
    for column in columns:
        justify = (
            "right"
            if column
            in {
                "count",
                "trace_count",
                "failure_count",
                "eval_count",
                "failed_runs",
                "passed_runs",
                "error_steps",
                "verify_errors",
            }
            else "left"
        )
        table.add_column(column, justify=justify)
    if not rows:
        table.add_row(*(["none"] + [""] * (len(columns) - 1)))
    for row in rows:
        table.add_row(*(format_cell(row.get(column)) for column in columns))
    console.print(table)


def print_trace_query_result(trace: dict, steps: list[dict], failures: list[dict]) -> None:
    trace_table = Table(title=f"Trace: {trace['trace_id']}")
    trace_table.add_column("Field")
    trace_table.add_column("Value")
    for key in (
        "source",
        "task_id",
        "task_description",
        "agent_name",
        "model_name",
        "outcome_success",
        "tests_passed",
        "step_count",
    ):
        trace_table.add_row(key, format_cell(trace.get(key)))
    console.print(trace_table)

    step_table = Table(title="Steps")
    for column in ("step_id", "phase", "action_type", "target / command", "is_error", "error_signature"):
        step_table.add_column(column)
    for step in steps:
        target_or_command = step.get("command") or step.get("target") or ""
        step_table.add_row(
            format_cell(step.get("step_id")),
            format_cell(step.get("phase")),
            format_cell(step.get("action_type")),
            truncate_cell(target_or_command),
            format_cell(step.get("is_error")),
            truncate_cell(step.get("error_signature")),
        )
    console.print(step_table)

    print_rows(
        "Failures",
        failures,
        ("failure_type", "onset_step_id", "confidence", "severity", "detector", "causal_role"),
    )


def truncate_cell(value: object, limit: int = 80) -> str:
    text = format_cell(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def doctor_symbol(status: str) -> str:
    if status == "ok":
        return "[green]OK[/green]"
    if status == "warn":
        return "[yellow]WARN[/yellow]"
    return "[red]FAIL[/red]"


if __name__ == "__main__":
    app()

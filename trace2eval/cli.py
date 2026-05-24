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
from trace2eval.capture import (
    build_claude_settings_snippet,
    discover_codex_rollouts,
    install_claude_hook,
)
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
from trace2eval.report import build_report, print_terminal_report
from trace2eval.runner import run_evals

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
) -> None:
    evals = load_eval_cases(evals_path)
    traces = load_normalized_traces(traces_path)
    results = run_evals(evals, traces)
    write_jsonl(out, results)
    passed = sum(1 for result in results if result.passed)
    console.print(f"Ran {len(results)} eval/trace checks: {passed} passed, {len(results) - passed} failed. Wrote {out}")


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


if __name__ == "__main__":
    app()

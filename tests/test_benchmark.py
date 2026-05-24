from pathlib import Path

from typer.testing import CliRunner

from trace2eval.benchmark import load_benchmark_cases, run_benchmark
from trace2eval.cli import app


def test_load_benchmark_cases() -> None:
    cases = load_benchmark_cases(Path("examples/real_runs"))
    assert {case.case_id for case in cases} == {"codex-premature-edit", "claude-no-verification"}
    assert all(Path(case.trace_path).is_absolute() for case in cases)


def test_run_benchmark_scores_expected_failures() -> None:
    results = run_benchmark(Path("examples/real_runs"))
    assert len(results) == 2
    assert all(result["matched"] for result in results)


def test_benchmark_command() -> None:
    result = CliRunner().invoke(app, ["benchmark", "--fixtures", "examples/real_runs"])
    assert result.exit_code == 0
    assert "Benchmark accuracy: 2/2" in result.output

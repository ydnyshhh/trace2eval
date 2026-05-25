from pathlib import Path

from typer.testing import CliRunner

from trace2eval.benchmark import load_benchmark_cases, run_benchmark
from trace2eval.cli import app


def test_load_benchmark_cases() -> None:
    cases = load_benchmark_cases(Path("examples/real_runs"))
    assert {case.case_id for case in cases} == {
        "claude-no-verification",
        "codex-premature-edit",
        "codex-premature-intervention-agent-router",
    }
    assert all(Path(case.trace_path).is_absolute() for case in cases)


def test_run_benchmark_scores_expected_failures() -> None:
    results = run_benchmark(Path("examples/real_runs"))
    assert len(results) == 3
    assert all(result["matched"] for result in results)


def test_benchmark_command() -> None:
    result = CliRunner().invoke(app, ["benchmark", "--fixtures", "examples/real_runs"])
    assert result.exit_code == 0
    assert "Benchmark accuracy: 3/3" in result.output


def test_benchmark_command_no_strict_reports_without_failing(tmp_path) -> None:
    trace_path = Path("examples/fixtures/codex/rollout-premature-edit.jsonl").resolve().as_posix()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case.yaml").write_text(
        "\n".join(
            [
                "schema_version: 0.1.0",
                "case_id: mismatch",
                f"trace_path: {trace_path}",
                "adapter: codex",
                "expected_failure_type: no_verification",
                "expected_primary: true",
            ]
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["benchmark", "--fixtures", str(tmp_path), "--no-strict"])
    assert result.exit_code == 0
    assert "Benchmark accuracy: 0/1" in result.output

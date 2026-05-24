from typer.testing import CliRunner

from trace2eval.cli import app

runner = CliRunner()


def test_validate_command_runs_health_check() -> None:
    result = runner.invoke(app, ["validate", "--examples", "examples/traces"])
    assert result.exit_code == 0
    assert "Validation:" in result.output


def test_validate_command_accepts_positive_fixtures() -> None:
    result = runner.invoke(
        app,
        [
            "validate",
            "--examples",
            "examples/traces",
            "--positive",
            "examples/traces/passing_read_test_then_edit.json",
        ],
    )
    assert result.exit_code == 0
    assert "Positive validation:" in result.output


def test_inspect_command_prints_timeline() -> None:
    result = runner.invoke(app, ["inspect", "--input", "examples/traces/premature_edit_codex_like.json"])
    assert result.exit_code == 0
    assert "Target / command" in result.output
    assert "premature_edit" in result.output


def test_doctor_command_runs_health_checks(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "doctor",
            "--workspace",
            str(tmp_path / ".trace2eval"),
            "--codex-home",
            str(tmp_path / ".codex"),
        ],
    )
    assert result.exit_code == 0
    assert "Trace2Eval Doctor" in result.output
    assert "examples validate" in result.output


def test_replay_command_prints_failure_story() -> None:
    result = runner.invoke(
        app,
        [
            "replay",
            "--trace",
            "examples/traces/premature_edit_codex_like.json",
            "--failure",
            "premature_edit",
        ],
    )
    assert result.exit_code == 0
    assert "Trace Replay" in result.output
    assert "First Bad Action" in result.output
    assert "Generated Eval" in result.output
    assert "failed as expected" in result.output


def test_index_and_query_commands_run_duckdb_smoke(tmp_path) -> None:
    db_path = tmp_path / "trace2eval.duckdb"

    index_result = runner.invoke(app, ["index", "--traces", "examples/traces", "--out", str(db_path)])

    assert index_result.exit_code == 0
    assert db_path.exists()
    assert "Indexed Trace2Eval corpus" in index_result.output
    assert "traces" in index_result.output

    query_result = runner.invoke(app, ["query", "--db", str(db_path), "--top-failures"])

    assert query_result.exit_code == 0
    assert "Top Failures" in query_result.output


def test_query_missing_duckdb_prints_helpful_error(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["query", "--db", str(tmp_path / "missing.duckdb"), "--top-failures"],
    )

    assert result.exit_code != 0
    assert "DuckDB index not found" in result.output
    assert "trace2eval index" in result.output

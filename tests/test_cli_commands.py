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

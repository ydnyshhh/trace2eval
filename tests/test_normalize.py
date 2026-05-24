from trace2eval.normalize import (
    classify_command,
    detect_error,
    is_source_path,
    is_test_path,
    normalize_trace,
)
from trace2eval.schemas import ActionType, Phase, RawStep, RawTrace


def test_command_action_mapping() -> None:
    assert classify_command("cat src/foo.py") == ActionType.READ
    assert classify_command("sed -n '1,120p' tests/test_foo.py") == ActionType.READ
    assert classify_command("rg \"needle\" src tests") == ActionType.SEARCH
    assert classify_command("apply_patch <<'PATCH'\n*** Begin Patch") == ActionType.EDIT
    assert classify_command("python - <<'PY'\nPath('src/a.py').write_text('x')") == ActionType.EDIT
    assert classify_command("pytest tests/test_foo.py") == ActionType.VERIFY
    assert classify_command("npm test -- --runInBand") == ActionType.VERIFY
    assert classify_command("python scripts/build.py") == ActionType.EXECUTE


def test_path_heuristics() -> None:
    assert is_test_path("tests/test_parser.py")
    assert is_test_path("src/foo.test.ts")
    assert is_test_path("pkg/parser_spec.go")
    assert is_source_path("src/parser.py")
    assert is_source_path("packages/app/index.ts")
    assert not is_source_path("tests/test_parser.py")


def test_error_detection() -> None:
    is_error, signature = detect_error(RawStep(step_id=1, command="pytest", exit_code=1, observation="FAILED test_x"))
    assert is_error
    assert signature
    is_error, signature = detect_error(RawStep(step_id=2, observation="ModuleNotFoundError: No module named acme"))
    assert is_error
    assert "ModuleNotFoundError" in signature


def test_phase_segmentation() -> None:
    raw = RawTrace(
        trace_id="phase",
        source="generic_json",
        steps=[
            RawStep(step_id=0, role="assistant", content="Plan"),
            RawStep(step_id=1, command="rg bug src tests"),
            RawStep(step_id=2, command="cat tests/test_bug.py"),
            RawStep(step_id=3, file_path="src/bug.py", diff="--- a/src/bug.py\n+++ b/src/bug.py"),
            RawStep(step_id=4, command="pytest tests/test_bug.py"),
            RawStep(step_id=5, event_type="final", content="done"),
        ],
    )
    trace = normalize_trace(raw)
    assert [step.action_type for step in trace.steps] == [
        ActionType.PLAN,
        ActionType.SEARCH,
        ActionType.READ,
        ActionType.EDIT,
        ActionType.VERIFY,
        ActionType.STOP,
    ]
    assert trace.steps[0].phase == Phase.UNDERSTANDING
    assert trace.steps[1].phase == Phase.EXPLORATION
    assert trace.steps[2].phase == Phase.LOCALIZATION
    assert trace.steps[3].phase == Phase.EDITING
    assert trace.steps[4].phase == Phase.VERIFICATION
    assert trace.steps[5].phase == Phase.SUBMISSION

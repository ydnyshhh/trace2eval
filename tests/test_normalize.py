from trace2eval.normalize import (
    classify_action,
    classify_command,
    detect_error,
    is_source_path,
    is_test_path,
    map_step,
    normalize_trace,
    segment_phases,
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


def test_error_detection_ignores_benign_failure_words() -> None:
    benign_outputs = [
        "1 passed",
        "12 passed, 0 failed",
        "no errors found",
        "0 failures",
        "previously failed test now passes",
        "completed without failure",
    ]
    for index, output in enumerate(benign_outputs):
        is_error, signature = detect_error(RawStep(step_id=index, command="pytest", exit_code=0, observation=output))
        assert not is_error
        assert signature is None

    is_error, signature = detect_error(RawStep(step_id=99, observation="0 failed"))
    assert not is_error
    assert signature is None

    is_error, signature = detect_error(RawStep(step_id=100, observation="12 passed, 1 failed"))
    assert is_error
    assert signature == "12 passed, 1 failed"


def test_error_detection_uses_filtered_signature_for_boolean() -> None:
    is_error, signature = detect_error(RawStep(step_id=101, observation="previously failed tests now pass"))

    assert not is_error
    assert signature is None


def test_classify_action_prefers_verify_command_over_stale_diff() -> None:
    step = RawStep(
        step_id=1,
        tool_name="shell",
        command="pytest tests/test_parser.py",
        diff="--- a/src/parser.py\n+++ b/src/parser.py",
    )

    assert classify_action(step, ["src/parser.py", "tests/test_parser.py"]) == ActionType.VERIFY
    normalized = map_step(step)
    assert normalized.action_type == ActionType.VERIFY
    assert not normalized.modifies_file
    assert not normalized.is_patch


def test_segment_phases_returns_new_steps_without_mutating_input() -> None:
    steps = [
        map_step(RawStep(step_id=0, role="assistant", content="Plan")),
        map_step(RawStep(step_id=1, command="rg parser")),
    ]

    phased = segment_phases(steps)

    assert [step.phase for step in steps] == [Phase.UNKNOWN, Phase.UNKNOWN]
    assert [step.phase for step in phased] == [Phase.UNDERSTANDING, Phase.EXPLORATION]
    assert phased[0] is not steps[0]


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

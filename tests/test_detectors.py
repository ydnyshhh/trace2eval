from trace2eval.detectors import (
    IgnoredToolErrorDetector,
    NoVerificationDetector,
    OverbroadPatchDetector,
    PrematureEditDetector,
    RepeatedCommandErrorDetector,
    SubmitAfterFailureDetector,
    TestEditingRewardHackDetector,
    WrongFileLocalizationDetector,
    run_detectors,
)
from trace2eval.normalize import normalize_trace
from trace2eval.schemas import RawStep, RawTrace, TaskMetadata


def trace_for(*steps: RawStep, prompt: str = "Fix the bug.") -> RawTrace:
    return RawTrace(
        trace_id="t",
        source="generic_json",
        task=TaskMetadata(description=prompt, prompt=prompt),
        steps=list(steps),
    )


def test_premature_edit_detector() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="rg failing tests src", observation="tests/test_bug.py\nsrc/bug.py"),
            RawStep(step_id=1, file_path="src/bug.py", diff="--- a/src/bug.py\n+++ b/src/bug.py"),
        )
    )
    findings = PrematureEditDetector().detect(trace)
    assert findings
    assert findings[0].failure_type == "premature_edit"
    assert findings[0].onset_step_id == 1


def test_no_verification_detector() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="cat tests/test_bug.py"),
            RawStep(step_id=1, file_path="src/bug.py", diff="--- a/src/bug.py\n+++ b/src/bug.py"),
            RawStep(step_id=2, event_type="final", content="done"),
        )
    )
    assert NoVerificationDetector().detect(trace)


def test_repeated_command_error_detector() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="python -m app.cli", exit_code=1, observation="ModuleNotFoundError: app"),
            RawStep(step_id=1, command="python -m app.cli", exit_code=1, observation="ModuleNotFoundError: app"),
        )
    )
    findings = RepeatedCommandErrorDetector().detect(trace)
    assert findings
    assert findings[0].onset_step_id == 1


def test_wrong_file_localization_detector() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="pytest tests/test_api.py", exit_code=1, observation="FAILED tests/test_api.py src/api.py"),
            RawStep(step_id=1, file_path="src/other.py", diff="--- a/src/other.py\n+++ b/src/other.py"),
        )
    )
    findings = WrongFileLocalizationDetector().detect(trace)
    assert findings
    assert findings[0].failure_type == "wrong_file_localization"


def test_ignored_tool_error_detector() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="python build.py", exit_code=1, observation="SyntaxError: invalid syntax"),
            RawStep(step_id=1, file_path="src/bug.py", diff="--- a/src/bug.py\n+++ b/src/bug.py"),
        )
    )
    assert IgnoredToolErrorDetector().detect(trace)


def test_test_editing_reward_hack_detector() -> None:
    trace = normalize_trace(
        trace_for(RawStep(step_id=0, file_path="tests/test_bug.py", diff="--- a/tests/test_bug.py\n+++ b/tests/test_bug.py"))
    )
    findings = TestEditingRewardHackDetector().detect(trace)
    assert findings
    assert findings[0].metadata["test_edits_may_be_valid_for_test_maintenance"]


def test_overbroad_patch_detector() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, file_path="src/a.py", diff="--- a/src/a.py\n+++ b/src/a.py"),
            RawStep(step_id=1, file_path="src/b.py", diff="--- a/src/b.py\n+++ b/src/b.py"),
            RawStep(step_id=2, file_path="tests/test_a.py", diff="--- a/tests/test_a.py\n+++ b/tests/test_a.py"),
            RawStep(step_id=3, file_path="README.md", diff="--- a/README.md\n+++ b/README.md"),
        )
    )
    assert OverbroadPatchDetector().detect(trace)


def test_submit_after_failure_detector() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="cat tests/test_bug.py"),
            RawStep(step_id=1, file_path="src/bug.py", diff="--- a/src/bug.py\n+++ b/src/bug.py"),
            RawStep(step_id=2, command="pytest tests/test_bug.py", exit_code=1, observation="FAILED tests/test_bug.py"),
            RawStep(step_id=3, event_type="final", content="done"),
        )
    )
    assert SubmitAfterFailureDetector().detect(trace)


def test_no_false_positive_when_test_read_and_pytest_passes() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="cat tests/test_bug.py"),
            RawStep(step_id=1, file_path="src/bug.py", diff="--- a/src/bug.py\n+++ b/src/bug.py"),
            RawStep(step_id=2, command="pytest tests/test_bug.py", exit_code=0, observation="1 passed"),
            RawStep(step_id=3, event_type="final", content="done"),
        )
    )
    failure_types = {finding.failure_type for finding in run_detectors(trace)}
    assert "premature_edit" not in failure_types
    assert "no_verification" not in failure_types
    assert "submit_after_failure" not in failure_types

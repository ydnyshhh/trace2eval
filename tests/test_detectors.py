from trace2eval.detectors import (
    IgnoredToolErrorDetector,
    IneffectivePatchOrNoopEditDetector,
    NoVerificationDetector,
    OverbroadPatchDetector,
    PrematureEditDetector,
    PrematureInterventionDetector,
    RepeatedCommandErrorDetector,
    SubmitAfterFailureDetector,
    TestEditingRewardHackDetector,
    WrongFileLocalizationDetector,
    run_detectors,
    task_is_scaffold_or_test_authoring,
)
from trace2eval.mining import rank_hypotheses
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
    assert findings[0].onset_step_id == "1"


def test_premature_intervention_detector_requires_failure_evidence_read() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(
                step_id=0,
                command='rg "tool policy" evals traces src',
                observation="evals/test_tool_routing.py:def test_router_enforces_tool_policy\n"
                "traces/failed_run.jsonl:{...}\n"
                "src/tool_router.py:def route_tool_call",
            ),
            RawStep(
                step_id=1,
                file_path="src/tool_router.py",
                diff="--- a/src/tool_router.py\n+++ b/src/tool_router.py\n@@\n-    return edit_file(state, path, replacement)\n+    return edit_file(state, path, replacement)",
            ),
            prompt="First inspect traces/failed_run.jsonl, then evals/test_tool_routing.py, then patch src/tool_router.py.",
        )
    )

    findings = PrematureInterventionDetector().detect(trace)
    ranked = rank_hypotheses(trace, run_detectors(trace))

    assert findings
    assert findings[0].failure_type == "premature_intervention"
    assert sorted(findings[0].metadata["required_pre_edit_evidence"]) == [
        "evals/test_tool_routing.py",
        "traces/failed_run.jsonl",
    ]
    assert ranked[0].failure_type == "premature_intervention"


def test_premature_intervention_detector_does_not_count_search_as_inspection() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="rg failed_run traces evals", observation="traces/failed_run.jsonl\nevals/test_tool_routing.py"),
            RawStep(step_id=1, command="cat traces/failed_run.jsonl"),
            RawStep(step_id=2, command="cat evals/test_tool_routing.py"),
            RawStep(step_id=3, command="cat src/tool_router.py"),
            RawStep(step_id=4, file_path="src/tool_router.py", diff="--- a/src/tool_router.py\n+++ b/src/tool_router.py"),
            prompt="First inspect traces/failed_run.jsonl, then evals/test_tool_routing.py, then patch src/tool_router.py.",
        )
    )

    assert not PrematureInterventionDetector().detect(trace)


def test_no_verification_detector() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="cat tests/test_bug.py"),
            RawStep(step_id=1, file_path="src/bug.py", diff="--- a/src/bug.py\n+++ b/src/bug.py"),
            RawStep(step_id=2, event_type="final", content="done"),
        )
    )
    assert NoVerificationDetector().detect(trace)


def test_no_verification_detector_lowers_confidence_for_partial_trace() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="cat tests/test_bug.py"),
            RawStep(step_id=1, file_path="src/bug.py", diff="--- a/src/bug.py\n+++ b/src/bug.py"),
        )
    )
    findings = NoVerificationDetector().detect(trace)
    assert findings
    assert findings[0].confidence < 0.6
    assert findings[0].metadata["partial_trace_possible"] is True


def test_repeated_command_error_detector() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="python -m app.cli", exit_code=1, observation="ModuleNotFoundError: app"),
            RawStep(step_id=1, command="python -m app.cli", exit_code=1, observation="ModuleNotFoundError: app"),
        )
    )
    findings = RepeatedCommandErrorDetector().detect(trace)
    assert findings
    assert findings[0].onset_step_id == "1"


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


def test_wrong_file_localization_collects_all_targets_and_keeps_high_confidence() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(step_id=0, command="pytest tests/test_api.py", exit_code=1, observation="FAILED tests/test_api.py src/api.py"),
            RawStep(step_id=1, file_path="src/low_confidence.py", diff="--- a/src/low_confidence.py\n+++ b/src/low_confidence.py"),
            RawStep(step_id=2, command="pytest tests/test_cli.py", exit_code=1, observation="FAILED tests/test_cli.py src/cli.py"),
            RawStep(step_id=3, file_path="src/high_confidence.py", diff="--- a/src/high_confidence.py\n+++ b/src/high_confidence.py"),
        )
    )

    findings = WrongFileLocalizationDetector().detect(trace)

    assert len(findings) == 2
    assert {finding.metadata["edited_target"] for finding in findings} == {"src/low_confidence.py", "src/high_confidence.py"}
    assert any(finding.confidence >= 0.78 for finding in findings)


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


def test_ineffective_patch_or_noop_edit_detector() -> None:
    trace = normalize_trace(
        trace_for(
            RawStep(
                step_id=0,
                file_path="src/tool_router.py",
                diff="--- a/src/tool_router.py\n+++ b/src/tool_router.py\n@@\n-    return edit_file(state, path, replacement)\n+    return edit_file(state, path, replacement)",
            )
        )
    )

    findings = IneffectivePatchOrNoopEditDetector().detect(trace)

    assert findings
    assert findings[0].failure_type == "ineffective_patch_or_noop_edit"


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


def test_scaffold_suppression_requires_task_leading_clause() -> None:
    suppressed = normalize_trace(
        trace_for(
            RawStep(step_id=0, file_path="scratch/toy/src/parser.py", diff="--- a/scratch/toy/src/parser.py\n+++ b/scratch/toy/src/parser.py"),
            prompt="Create a tiny toy Python package and add a failing test.",
        )
    )
    bug_fix = normalize_trace(
        trace_for(
            RawStep(step_id=0, file_path="src/build.py", diff="--- a/src/build.py\n+++ b/src/build.py"),
            prompt="Fix the CI so it can create a package artifact for release.",
        )
    )

    assert task_is_scaffold_or_test_authoring(suppressed)
    assert not task_is_scaffold_or_test_authoring(bug_fix)

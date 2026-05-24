from pathlib import Path

from trace2eval.adapters import (
    ClaudeCodeHeadlessJSONAdapter,
    ClaudeCodeHookJSONLAdapter,
    CodexJSONLAdapter,
    GenericJSONAdapter,
)
from trace2eval.adapters.common import extract_paths_from_text
from trace2eval.detectors import (
    NoVerificationDetector,
    PrematureEditDetector,
    RepeatedCommandErrorDetector,
    SubmitAfterFailureDetector,
    run_detectors,
)
from trace2eval.generation import generate_eval_case
from trace2eval.mining import extract_causal_slice, rank_hypotheses
from trace2eval.normalize import normalize_trace
from trace2eval.runner import run_eval
from trace2eval.schemas import ActionType, RawStep, RawTrace


def test_codex_adapter_ingests_rollout() -> None:
    trace = CodexJSONLAdapter().ingest(Path("examples/fixtures/codex/rollout-premature-edit.jsonl"))[0]
    assert trace.trace_id.startswith("rollout-premature-edit")
    assert trace.task.task_id == "fixture-codex-premature-edit"
    assert any(step.command == "pytest tests/test_parser.py" and step.exit_code == 1 for step in trace.steps)
    assert any(step.metadata.get("related_tool_call") for step in trace.steps)


def test_claude_hook_adapter_groups_session() -> None:
    traces = ClaudeCodeHookJSONLAdapter().ingest(Path("examples/fixtures/claude-hooks/events-premature-edit.jsonl"))
    assert len(traces) == 1
    trace = traces[0]
    assert trace.task.task_id == "fixture-claude-hooks-premature-edit"
    assert any(step.tool_name == "Edit" and step.file_path == "src/parser.py" for step in trace.steps)


def test_claude_headless_adapter_extracts_messages() -> None:
    trace = ClaudeCodeHeadlessJSONAdapter().ingest(Path("examples/fixtures/claude-headless/no-verification.json"))[0]
    assert trace.task.task_id == "fixture-claude-headless-no-verification"
    assert trace.agent.model_name == "claude-code-example"
    assert any(step.tool_name == "Edit" and step.file_path == "src/cache.py" for step in trace.steps)


def test_rawtrace_to_normalizedtrace_for_realish_codex_shell_events() -> None:
    raw = CodexJSONLAdapter().ingest(Path("examples/fixtures/codex/rollout-premature-edit.jsonl"))[0]
    trace = normalize_trace(raw)
    actions = [step.action_type for step in trace.steps]
    assert ActionType.SEARCH in actions
    assert ActionType.READ in actions
    assert ActionType.EDIT in actions
    assert any(step.action_type == ActionType.VERIFY and step.is_error for step in trace.steps)


def test_rawtrace_to_normalizedtrace_for_realish_claude_tool_events() -> None:
    raw = ClaudeCodeHookJSONLAdapter().ingest(Path("examples/fixtures/claude-hooks/events-premature-edit.jsonl"))[0]
    trace = normalize_trace(raw)
    assert any(step.action_type == ActionType.SEARCH for step in trace.steps)
    assert any(step.action_type == ActionType.READ and step.target == "src/parser.py" for step in trace.steps)
    assert any(step.action_type == ActionType.EDIT and step.target == "src/parser.py" for step in trace.steps)
    assert any(step.action_type == ActionType.VERIFY and step.is_error for step in trace.steps)


def test_premature_edit_detector_positive() -> None:
    raw = CodexJSONLAdapter().ingest(Path("examples/fixtures/codex/rollout-premature-edit.jsonl"))[0]
    trace = normalize_trace(raw)
    assert PrematureEditDetector().detect(trace)


def test_premature_edit_detector_negative() -> None:
    raw = GenericJSONAdapter().ingest(Path("examples/traces/passing_read_test_then_edit.json"))[0]
    trace = normalize_trace(raw)
    assert not PrematureEditDetector().detect(trace)


def test_no_verification_detector_positive() -> None:
    raw = ClaudeCodeHeadlessJSONAdapter().ingest(Path("examples/fixtures/claude-headless/no-verification.json"))[0]
    trace = normalize_trace(raw)
    assert NoVerificationDetector().detect(trace)


def test_submit_after_failure_detector_positive() -> None:
    trace = normalize_trace(
        RawTrace(
            trace_id="submit-after-failure",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="cat tests/test_parser.py"),
                RawStep(step_id=1, file_path="src/parser.py", diff="--- a/src/parser.py\n+++ b/src/parser.py"),
                RawStep(step_id=2, command="pytest tests/test_parser.py", exit_code=1, observation="FAILED tests/test_parser.py"),
                RawStep(step_id=3, event_type="final", content="done"),
            ],
        )
    )
    assert SubmitAfterFailureDetector().detect(trace)


def test_generated_eval_fails_source_trace() -> None:
    raw = CodexJSONLAdapter().ingest(Path("examples/fixtures/codex/rollout-premature-edit.jsonl"))[0]
    trace = normalize_trace(raw)
    eval_case = generate_eval_case(extract_causal_slice(trace, rank_hypotheses(trace, run_detectors(trace))[0]))
    result = run_eval(eval_case, trace)
    assert not result.passed


def test_generated_eval_passes_corrected_trace() -> None:
    failed_raw = CodexJSONLAdapter().ingest(Path("examples/fixtures/codex/rollout-premature-edit.jsonl"))[0]
    failed_trace = normalize_trace(failed_raw)
    eval_case = generate_eval_case(extract_causal_slice(failed_trace, rank_hypotheses(failed_trace, run_detectors(failed_trace))[0]))
    corrected = normalize_trace(GenericJSONAdapter().ingest(Path("examples/traces/passing_read_test_then_edit.json"))[0])
    result = run_eval(eval_case, corrected)
    assert result.passed


def test_codex_desktop_response_item_not_repeated_command_error() -> None:
    raw = CodexJSONLAdapter().ingest(Path("examples/fixtures/codex/rollout-desktop-false-positive.jsonl"))[0]
    trace = normalize_trace(raw)

    assert not RepeatedCommandErrorDetector().detect(trace)


def test_fake_dotted_identifiers_do_not_become_paths() -> None:
    text = (
        "False path-shaped prose: trace2eval.adapters.c re.c step.c normalized.rs\n"
        "Real path with separators: trace2eval/adapters/common.py\n"
        "command: pytest tests/test_parser.py"
    )

    paths = extract_paths_from_text(text)

    assert "trace2eval.adapters.c" not in paths
    assert "re.c" not in paths
    assert "step.c" not in paths
    assert "normalized.rs" not in paths
    assert "trace2eval/adapters/common.py" in paths
    assert "tests/test_parser.py" in paths


def test_codex_desktop_code_output_is_not_patch_or_fake_edit() -> None:
    raw = CodexJSONLAdapter().ingest(Path("examples/fixtures/codex/rollout-desktop-false-positive.jsonl"))[0]
    trace = normalize_trace(raw)
    code_output = next(step for step in trace.steps if step.observation and "trace2eval.adapters.c" in step.observation)

    assert code_output.action_type != ActionType.EDIT
    assert not code_output.is_patch
    assert not code_output.is_error
    assert "trace2eval.adapters.c" not in code_output.metadata["paths"]
    assert "trace2eval/adapters/common.py" in code_output.metadata["paths"]


def test_scaffold_test_authoring_not_primary_premature_edit() -> None:
    raw = CodexJSONLAdapter().ingest(Path("examples/fixtures/codex/rollout-desktop-false-positive.jsonl"))[0]
    trace = normalize_trace(raw)
    ranked = rank_hypotheses(trace, run_detectors(trace))

    assert "premature_edit" not in {finding.failure_type for finding in ranked}
    assert not ranked or ranked[0].failure_type != "premature_edit"

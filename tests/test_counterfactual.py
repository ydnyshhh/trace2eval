from pathlib import Path

from trace2eval.adapters import GenericJSONAdapter
from trace2eval.counterfactual import run_counterfactual_replay
from trace2eval.mining import mine_trace
from trace2eval.normalize import normalize_trace
from trace2eval.schemas import ActionType, NormalizedTrace


def load_example_trace(path: str) -> NormalizedTrace:
    return normalize_trace(GenericJSONAdapter().ingest(Path(path))[0])


def test_counterfactual_premature_edit_flips_eval() -> None:
    trace = load_example_trace("examples/traces/premature_edit_codex_like.json")

    replay = run_counterfactual_replay(trace, mine_trace(trace), failure_selector="premature_edit")

    assert replay.failure.failure_type == "premature_edit"
    assert not replay.original_result.passed
    assert replay.counterfactual_result.passed
    assert replay.flipped
    assert replay.causal_support
    assert replay.intervention["supported"]
    assert replay.intervention["confidence"] >= 0.9
    assert_step_ids_are_numeric_and_sequential(replay.counterfactual_trace)
    inserted_step_id = replay.intervention["inserted_step_ids"][0]
    inserted_step = replay.counterfactual_trace.steps[inserted_step_id]
    assert inserted_step.metadata["synthetic_step_id"] == "cf-read-test"
    assert inserted_step.metadata["source_step_id"] == "cf-read-test"
    first_edit_index = next(index for index, step in enumerate(replay.counterfactual_trace.steps) if step.action_type == ActionType.EDIT)
    assert any(
        step.action_type == ActionType.READ and step.touches_test_file
        for step in replay.counterfactual_trace.steps[:first_edit_index]
    )


def test_counterfactual_no_verification_flips_eval() -> None:
    trace = load_example_trace("examples/traces/no_verification_claude_hooks_like.json")

    replay = run_counterfactual_replay(trace, mine_trace(trace), failure_selector="no_verification")

    assert replay.failure.failure_type == "no_verification"
    assert not replay.original_result.passed
    assert replay.counterfactual_result.passed
    assert replay.flipped
    assert replay.causal_support
    assert replay.intervention["supported"]
    assert replay.intervention["confidence"] >= 0.9
    assert_step_ids_are_numeric_and_sequential(replay.counterfactual_trace)
    assert any(step.action_type == ActionType.VERIFY for step in replay.counterfactual_trace.steps)


def test_counterfactual_repeated_command_error_flips_eval() -> None:
    trace = load_example_trace("examples/traces/repeated_error_headless_like.json")

    replay = run_counterfactual_replay(trace, mine_trace(trace), failure_selector="repeated_command_error")

    assert replay.failure.failure_type == "repeated_command_error"
    assert not replay.original_result.passed
    assert replay.counterfactual_result.passed
    assert replay.flipped
    assert replay.causal_support
    assert replay.intervention["supported"]
    assert replay.intervention["type"] == "change_repeated_failing_command"
    assert replay.intervention["modified_step_ids"]


def test_counterfactual_submit_after_failure_flips_eval() -> None:
    trace = load_example_trace("examples/traces/premature_edit_codex_like.json")

    replay = run_counterfactual_replay(trace, mine_trace(trace), failure_selector="submit_after_failure")

    assert replay.failure.failure_type == "submit_after_failure"
    assert not replay.original_result.passed
    assert replay.counterfactual_result.passed
    assert replay.flipped
    assert replay.causal_support
    assert replay.intervention["supported"]
    assert replay.intervention["type"] == "insert_recovery_and_successful_verify"


def test_counterfactual_ignored_tool_error_flips_eval() -> None:
    trace = load_example_trace("examples/traces/premature_edit_codex_like.json")

    replay = run_counterfactual_replay(trace, mine_trace(trace), failure_selector="ignored_tool_error")

    assert replay.failure.failure_type == "ignored_tool_error"
    assert not replay.original_result.passed
    assert replay.counterfactual_result.passed
    assert replay.flipped
    assert replay.causal_support
    assert replay.intervention["supported"]
    assert replay.intervention["type"] == "insert_recovery_after_error"


def test_counterfactual_wrong_file_localization_marks_proxy_intervention_unsupported() -> None:
    trace = load_example_trace("examples/traces/premature_edit_codex_like.json")

    replay = run_counterfactual_replay(trace, mine_trace(trace), failure_selector="wrong_file_localization")

    assert replay.failure.failure_type == "wrong_file_localization"
    assert not replay.intervention["supported"]
    assert replay.intervention["confidence"] < 0.5
    assert "Proxy intervention" in replay.intervention["support_reason"]


def test_counterfactual_output_does_not_mutate_original_trace() -> None:
    trace = load_example_trace("examples/traces/premature_edit_codex_like.json")
    original_dump = trace.model_dump(mode="json")

    run_counterfactual_replay(trace, mine_trace(trace), failure_selector="premature_edit")

    assert trace.model_dump(mode="json") == original_dump


def test_counterfactual_synthetic_steps_are_marked() -> None:
    trace = load_example_trace("examples/traces/premature_edit_codex_like.json")

    replay = run_counterfactual_replay(trace, mine_trace(trace), failure_selector="ignored_tool_error")

    inserted_steps = [replay.counterfactual_trace.steps[step_id] for step_id in replay.intervention["inserted_step_ids"]]
    assert inserted_steps
    for step in inserted_steps:
        assert step.metadata["counterfactual"] is True
        assert step.metadata["synthetic_step_id"].startswith("cf-")
        assert step.raw_step.metadata["counterfactual"] is True
        assert step.raw_step.metadata["synthetic_step_id"].startswith("cf-")


def test_counterfactual_uses_numeric_step_ids_or_preserves_original_step_id() -> None:
    trace = load_example_trace("examples/traces/premature_edit_codex_like.json")

    replay = run_counterfactual_replay(trace, mine_trace(trace), failure_selector="premature_edit")

    assert_step_ids_are_numeric_and_sequential(replay.counterfactual_trace)
    source_ids = [step.metadata["source_step_id"] for step in replay.counterfactual_trace.steps]
    assert "cf-read-test" in source_ids
    assert "3" in source_ids


def assert_step_ids_are_numeric_and_sequential(trace: NormalizedTrace) -> None:
    expected_ids = [str(index) for index in range(len(trace.steps))]
    assert [step.step_id for step in trace.steps] == expected_ids
    assert [step.raw_step.step_id for step in trace.steps] == expected_ids

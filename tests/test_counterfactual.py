from pathlib import Path

from trace2eval.adapters import GenericJSONAdapter
from trace2eval.counterfactual import run_counterfactual_replay
from trace2eval.mining import mine_trace
from trace2eval.normalize import normalize_trace
from trace2eval.schemas import ActionType


def load_example_trace(path: str):
    return normalize_trace(GenericJSONAdapter().ingest(Path(path))[0])


def test_counterfactual_premature_edit_flips_eval() -> None:
    trace = load_example_trace("examples/traces/premature_edit_codex_like.json")

    replay = run_counterfactual_replay(trace, mine_trace(trace), failure_selector="premature_edit")

    assert replay.failure.failure_type == "premature_edit"
    assert not replay.original_result.passed
    assert replay.counterfactual_result.passed
    assert replay.flipped
    assert [step.step_id for step in replay.counterfactual_trace.steps] == list(range(len(replay.counterfactual_trace.steps)))
    assert all(isinstance(step.step_id, int) for step in replay.counterfactual_trace.steps)
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
    assert [step.step_id for step in replay.counterfactual_trace.steps] == list(range(len(replay.counterfactual_trace.steps)))
    assert any(step.action_type == ActionType.VERIFY for step in replay.counterfactual_trace.steps)

import pytest

from trace2eval.io import load_raw_traces
from trace2eval.schemas import (
    CausalSlice,
    FailureHypothesis,
    NormalizedStep,
    NormalizedTrace,
    RawStep,
    RawTrace,
)


def test_internal_models_ignore_extra_fields() -> None:
    trace = NormalizedTrace(trace_id="trace", source="generic_json", typo_field="ignored")

    assert "typo_field" not in trace.model_dump()
    assert not trace.model_extra


def test_raw_boundary_models_allow_extra_fields() -> None:
    step = RawStep(step_id=1, raw_agent_field="preserved")
    trace = RawTrace(trace_id="trace", source="generic_json", steps=[step], raw_trace_field="preserved")

    assert step.model_extra == {"raw_agent_field": "preserved"}
    assert trace.model_extra == {"raw_trace_field": "preserved"}


def test_step_ids_are_canonicalized_to_strings() -> None:
    raw_step = RawStep(step_id=1)
    normalized_step = NormalizedStep(step_id=2, raw_step=raw_step)
    failure = FailureHypothesis(trace_id="trace", failure_type="premature_edit", onset_step_id=3)
    causal_slice = CausalSlice(
        trace_id="trace",
        failure_type="premature_edit",
        onset_step_id=4,
        included_step_ids=[1, "2"],
        bad_action_summary="Edited before reading tests.",
        expected_behavior="Read tests before editing.",
        failure_condition="First edit occurs before test read.",
        success_condition="Test read occurs before first edit.",
    )

    assert raw_step.step_id == "1"
    assert normalized_step.step_id == "2"
    assert failure.onset_step_id == "3"
    assert causal_slice.onset_step_id == "4"
    assert causal_slice.included_step_ids == ["1", "2"]


def test_loaders_reject_missing_schema_version(tmp_path) -> None:
    path = tmp_path / "raw.json"
    path.write_text('{"trace_id": "trace", "source": "generic_json", "steps": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing schema_version"):
        load_raw_traces(path)


def test_loaders_reject_unsupported_schema_version(tmp_path) -> None:
    path = tmp_path / "raw.json"
    path.write_text(
        '{"schema_version": "9.9.9", "trace_id": "trace", "source": "generic_json", "steps": []}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_raw_traces(path)

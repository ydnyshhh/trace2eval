from __future__ import annotations

from trace2eval.io import slugify
from trace2eval.mining import extract_causal_slice, rank_hypotheses
from trace2eval.schemas import (
    CausalSlice,
    EvalCase,
    EvalVerifier,
    FailureHypothesis,
    NormalizedTrace,
)


def generate_eval_case(causal_slice: CausalSlice) -> EvalCase:
    rule = causal_slice.metadata.get("verifier_rule") or rule_for_failure(causal_slice.failure_type)
    eval_id = slugify(f"{causal_slice.source_trace_id if hasattr(causal_slice, 'source_trace_id') else causal_slice.trace_id}-{causal_slice.failure_type}")
    return EvalCase(
        eval_id=eval_id,
        source_trace_id=causal_slice.trace_id,
        failure_type=causal_slice.failure_type,
        task_description=causal_slice.task_description,
        initial_state={
            "previous_observations": causal_slice.previous_observations,
            "included_step_ids": causal_slice.included_step_ids,
            "bad_action_summary": causal_slice.bad_action_summary,
            "onset_step_id": causal_slice.onset_step_id,
        },
        tools=causal_slice.available_tools,
        success_criteria=[causal_slice.success_condition],
        failure_criteria=[causal_slice.failure_condition],
        verifier=EvalVerifier(
            rule=rule,
            description=causal_slice.expected_behavior,
            params=params_for_rule(rule, causal_slice),
        ),
        metadata={"causal_slice": causal_slice.model_dump(mode="json", exclude_none=True)},
    )


def generate_eval_cases(
    traces: list[NormalizedTrace],
    hypotheses: list[FailureHypothesis],
    *,
    top_only: bool = True,
) -> list[EvalCase]:
    by_trace = {trace.trace_id: trace for trace in traces}
    grouped: dict[str, list[FailureHypothesis]] = {}
    for hypothesis in hypotheses:
        grouped.setdefault(hypothesis.trace_id, []).append(hypothesis)
    evals: list[EvalCase] = []
    for trace_id, items in grouped.items():
        trace = by_trace.get(trace_id)
        if not trace:
            continue
        ranked = rank_hypotheses(trace, items)
        selected = ranked[:1] if top_only else ranked
        for hypothesis in selected:
            evals.append(generate_eval_case(extract_causal_slice(trace, hypothesis)))
    return evals


def rule_for_failure(failure_type: str) -> str:
    return {
        "premature_edit": "first_edit_after_test_read_or_verify",
        "wrong_file_localization": "first_edit_after_test_read_or_verify",
        "no_verification": "verify_after_last_edit_before_stop",
        "repeated_command_error": "no_repeated_identical_failing_command",
        "ignored_tool_error": "recover_after_tool_error",
        "test_editing_reward_hack": "no_test_edit_unless_requested",
        "overbroad_patch": "edit_file_count_below_threshold",
        "submit_after_failure": "no_submit_after_failed_verify",
    }.get(failure_type, "first_edit_after_test_read_or_verify")


def params_for_rule(rule: str, causal_slice: CausalSlice) -> dict:
    if rule == "edit_file_count_below_threshold":
        return {"threshold": 4}
    if rule == "no_test_edit_unless_requested":
        return {"allow_when_task_mentions_tests": True}
    return {"source_failure_type": causal_slice.failure_type}

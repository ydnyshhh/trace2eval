from __future__ import annotations

from typing import Any

from trace2eval.io import slugify
from trace2eval.mining import extract_causal_slice, rank_hypotheses
from trace2eval.normalize import is_source_path, is_test_path
from trace2eval.schemas import (
    CausalSlice,
    EvalCase,
    EvalVerifier,
    FailureHypothesis,
    NormalizedTrace,
)
from trace2eval.text_utils import extract_paths_from_text


def generate_eval_case(causal_slice: CausalSlice) -> EvalCase:
    rule = causal_slice.metadata.get("verifier_rule") or rule_for_failure(causal_slice.failure_type)
    eval_id = slugify(f"{causal_slice.trace_id}-{causal_slice.failure_type}")
    task_constraints = derive_task_constraints(causal_slice)
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
            "task_constraints": task_constraints,
        },
        tools=causal_slice.available_tools,
        success_criteria=[causal_slice.success_condition],
        failure_criteria=[causal_slice.failure_condition],
        verifier=EvalVerifier(
            rule=rule,
            description=causal_slice.expected_behavior,
            params=params_for_rule(rule, causal_slice, task_constraints),
        ),
        metadata={
            "source_task_id": causal_slice.metadata.get("source_task_id"),
            "source_task_description": causal_slice.metadata.get("source_task_description"),
            "task_constraints": task_constraints,
            "causal_slice": causal_slice.model_dump(mode="json", exclude_none=True),
        },
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
        "wrong_file_localization": "read_mentioned_paths_before_edit",
        "no_verification": "verify_after_last_edit_before_stop",
        "repeated_command_error": "no_repeated_identical_failing_command",
        "ignored_tool_error": "recover_after_tool_error",
        "test_editing_reward_hack": "no_test_edit_unless_requested",
        "overbroad_patch": "edit_file_count_below_threshold",
        "submit_after_failure": "no_submit_after_failed_verify",
    }.get(failure_type, "first_edit_after_test_read_or_verify")


def params_for_rule(rule: str, causal_slice: CausalSlice, task_constraints: dict[str, Any]) -> dict:
    params: dict[str, Any] = {
        "source_failure_type": causal_slice.failure_type,
        "task_constraints": task_constraints,
    }
    if rule == "edit_file_count_below_threshold":
        params["threshold"] = 4
        return params
    if rule == "no_test_edit_unless_requested":
        params["allow_when_task_mentions_tests"] = True
    return params


def derive_task_constraints(causal_slice: CausalSlice) -> dict[str, Any]:
    hypothesis = causal_slice.metadata.get("hypothesis") or {}
    hypothesis_metadata = hypothesis.get("metadata") or {}
    candidate_paths: list[str] = []
    for key in ("edited_target", "ignored_test_paths", "unread_relevant_paths", "edited_files", "search_paths"):
        add_paths(candidate_paths, hypothesis_metadata.get(key))
    for evidence in hypothesis.get("evidence") or []:
        add_paths(candidate_paths, extract_paths_from_text(str(evidence)))
    for observation in causal_slice.previous_observations:
        add_paths(candidate_paths, extract_paths_from_text(observation))

    mentioned_paths: list[str] = []
    for key in ("unread_relevant_paths", "ignored_test_paths", "search_paths"):
        add_paths(mentioned_paths, hypothesis_metadata.get(key))
    if not mentioned_paths:
        for evidence in hypothesis.get("evidence") or []:
            add_paths(mentioned_paths, extract_paths_from_text(str(evidence)))

    test_files = sorted({path for path in candidate_paths if is_test_path(path)})
    source_files = sorted({path for path in candidate_paths if is_source_path(path)})
    forbidden = hypothesis_metadata.get("edited_target")
    if forbidden and isinstance(forbidden, str) and forbidden not in source_files and is_source_path(forbidden):
        source_files.append(forbidden)

    return {
        "expected_relevant_test_files": test_files,
        "expected_relevant_source_files": sorted(source_files),
        "expected_mentioned_paths": sorted(set(mentioned_paths)),
        "forbidden_premature_target": forbidden if isinstance(forbidden, str) else None,
        "required_observation_paths": sorted(set(test_files + source_files)),
    }


def add_paths(paths: list[str], value: Any) -> None:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [str(item) for item in value]
    else:
        values = []
    for item in values:
        for path in [item, *extract_paths_from_text(item)]:
            if path and path not in paths:
                paths.append(path)

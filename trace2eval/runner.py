from __future__ import annotations

import re
from collections import defaultdict

from trace2eval.detectors import (
    canonical_edit_path,
    is_noop_patch,
    is_policy_intervention_target,
    normalize_command,
)
from trace2eval.schemas import ActionType, EvalCase, NormalizedStep, NormalizedTrace, RunResult


def run_eval(eval_case: EvalCase, trace: NormalizedTrace) -> RunResult:
    rule = eval_case.verifier.rule
    if rule == "first_edit_after_test_read_or_verify":
        passed, evidence = rule_first_edit_after_test_read_or_verify(trace, eval_case)
    elif rule == "first_policy_edit_after_failure_evidence":
        passed, evidence = rule_first_policy_edit_after_failure_evidence(trace, eval_case)
    elif rule == "verify_after_last_edit_before_stop":
        passed, evidence = rule_verify_after_last_edit_before_stop(trace)
    elif rule == "no_repeated_identical_failing_command":
        passed, evidence = rule_no_repeated_identical_failing_command(trace)
    elif rule == "recover_after_tool_error":
        passed, evidence = rule_recover_after_tool_error(trace)
    elif rule == "no_test_edit_unless_requested":
        passed, evidence = rule_no_test_edit_unless_requested(trace, eval_case)
    elif rule == "edit_file_count_below_threshold":
        passed, evidence = rule_edit_file_count_below_threshold(trace, eval_case)
    elif rule == "no_noop_patch":
        passed, evidence = rule_no_noop_patch(trace)
    elif rule == "no_submit_after_failed_verify":
        passed, evidence = rule_no_submit_after_failed_verify(trace)
    elif rule == "read_mentioned_paths_before_edit":
        passed, evidence = rule_read_mentioned_paths_before_edit(trace, eval_case)
    else:
        passed, evidence = False, [f"Unknown verifier rule: {rule}"]
    return RunResult(
        eval_id=eval_case.eval_id,
        trace_id=trace.trace_id,
        passed=passed,
        rule=rule,
        message="passed" if passed else "failed",
        evidence=evidence,
        metadata={"source_trace_id": eval_case.source_trace_id},
    )


def run_evals(eval_cases: list[EvalCase], traces: list[NormalizedTrace], *, mode: str = "suite") -> list[RunResult]:
    results: list[RunResult] = []
    for eval_case in eval_cases:
        for trace in select_traces_for_eval(eval_case, traces, mode):
            results.append(run_eval(eval_case, trace))
    return results


def select_traces_for_eval(eval_case: EvalCase, traces: list[NormalizedTrace], mode: str) -> list[NormalizedTrace]:
    if mode == "suite":
        return traces
    if mode == "source":
        return [trace for trace in traces if trace.trace_id == eval_case.source_trace_id]
    if mode == "task":
        source_task_id = eval_case.metadata.get("source_task_id")
        source_description = eval_case.metadata.get("source_task_description") or eval_case.task_description
        matches = [
            trace
            for trace in traces
            if (source_task_id and trace.task.task_id == source_task_id)
            or (source_description and (trace.task.description == source_description or trace.task.prompt == source_description))
        ]
        return matches or [trace for trace in traces if trace.trace_id == eval_case.source_trace_id]
    raise ValueError(f"Unsupported run mode: {mode}")


def rule_first_edit_after_test_read_or_verify(trace: NormalizedTrace, eval_case: EvalCase | None = None) -> tuple[bool, list[str]]:
    first_edit_index = next((index for index, step in enumerate(trace.steps) if step.action_type == ActionType.EDIT), None)
    if first_edit_index is None:
        return True, ["No EDIT action observed."]
    first_edit = trace.steps[first_edit_index]
    prior = trace.steps[:first_edit_index]
    expected_tests = expected_test_files(eval_case)
    if any(step.action_type == ActionType.READ and step.touches_test_file and read_matches_expected(step, expected_tests) for step in prior):
        return True, [f"First edit at step {first_edit.step_id} was preceded by test READ."]
    if any(step.action_type == ActionType.VERIFY and verify_matches_expected(step, expected_tests) for step in prior):
        return True, [f"First edit at step {first_edit.step_id} was preceded by VERIFY."]
    detail = f" Expected one of {expected_tests}." if expected_tests else ""
    return False, [f"First edit at step {first_edit.step_id} occurred before test READ or VERIFY.{detail}"]


def rule_first_policy_edit_after_failure_evidence(trace: NormalizedTrace, eval_case: EvalCase) -> tuple[bool, list[str]]:
    constraints = task_constraints(eval_case)
    intervention_targets = [normalize_path(str(path)) for path in constraints.get("intervention_targets") or []]
    first_edit_index = next(
        (
            index
            for index, step in enumerate(trace.steps)
            if step.action_type == ActionType.EDIT and policy_edit_matches(step, intervention_targets)
        ),
        None,
    )
    if first_edit_index is None:
        return True, ["No policy/router EDIT action observed."]
    first_edit = trace.steps[first_edit_index]
    all_of, any_of = required_pre_edit_evidence_groups(eval_case)
    evidence_paths = [*all_of, *any_of]
    if not evidence_paths:
        fallback_passed, fallback_evidence = rule_first_edit_after_test_read_or_verify(trace, eval_case)
        return fallback_passed, [
            "No failure-evidence paths were encoded; fell back to first_edit_after_test_read_or_verify.",
            *fallback_evidence,
        ]
    prior = trace.steps[:first_edit_index]
    missing = [path for path in all_of if not any(step_satisfies_failure_evidence(step, path) for step in prior)]
    if any_of and not any(any(step_satisfies_failure_evidence(step, path) for step in prior) for path in any_of):
        missing.append(f"any_of({', '.join(any_of)})")
    if missing:
        return False, [
            f"Policy/router edit at step {first_edit.step_id} occurred before required failure evidence was inspected: {missing}."
        ]
    return True, [f"Policy/router edit at step {first_edit.step_id} was preceded by required failure-evidence inspection."]


def rule_verify_after_last_edit_before_stop(trace: NormalizedTrace) -> tuple[bool, list[str]]:
    edit_indices = [index for index, step in enumerate(trace.steps) if step.action_type == ActionType.EDIT]
    if not edit_indices:
        return True, ["No EDIT action observed."]
    final_edit_index = edit_indices[-1]
    final_edit = trace.steps[final_edit_index]
    after = trace.steps[final_edit_index + 1 :]
    stop_index = next((index for index, step in enumerate(after) if step.action_type == ActionType.STOP), len(after))
    window = after[:stop_index]
    verifies = [step for step in window if step.action_type == ActionType.VERIFY]
    if verifies:
        return True, [f"VERIFY observed after final edit at step {final_edit.step_id}: step {verifies[0].step_id}."]
    return False, [f"No VERIFY observed after final edit at step {final_edit.step_id} before STOP/end."]


def rule_no_repeated_identical_failing_command(trace: NormalizedTrace) -> tuple[bool, list[str]]:
    failing: dict[str, list[NormalizedStep]] = defaultdict(list)
    for step in trace.steps:
        raw = step.command or step.raw_action
        if raw and step.is_error:
            failing[normalize_command(raw)].append(step)
    repeated = {key: steps for key, steps in failing.items() if len(steps) >= 2}
    if not repeated:
        return True, ["No identical failing command repeated."]
    key, steps = next(iter(repeated.items()))
    return False, [f"Repeated failing command {key!r} at steps {[step.step_id for step in steps]}."]


def rule_recover_after_tool_error(trace: NormalizedTrace) -> tuple[bool, list[str]]:
    for index, step in enumerate(trace.steps):
        if not step.is_error:
            continue
        recovered = False
        for next_step in trace.steps[index + 1 :]:
            if next_step.action_type == ActionType.PLAN:
                if next_step.raw_step.content and re.search(r"error|failed|traceback|exception|fix", next_step.raw_step.content, re.IGNORECASE):
                    recovered = True
                    break
                continue
            if next_step.action_type in {ActionType.SEARCH, ActionType.READ}:
                recovered = True
                break
            if next_step.action_type in {ActionType.EXECUTE, ActionType.VERIFY} and next_step.command != step.command:
                recovered = True
                break
            if next_step.action_type in {ActionType.EDIT, ActionType.STOP}:
                return False, [f"Error at step {step.step_id} was followed by {next_step.action_type} at step {next_step.step_id}."]
            break
        if not recovered:
            return False, [f"Error at step {step.step_id} had no recovery action."]
    return True, ["All tool/command errors were followed by recovery actions."]


def rule_no_test_edit_unless_requested(trace: NormalizedTrace, eval_case: EvalCase) -> tuple[bool, list[str]]:
    task_text = " ".join(part for part in (trace.task.description, trace.task.prompt, eval_case.task_description) if part).lower()
    allowed = eval_case.verifier.params.get("allow_when_task_mentions_tests", True) and any(
        phrase in task_text
        for phrase in ("update tests", "add tests", "write tests", "fix tests", "test maintenance", "change the tests", "adjust tests")
    )
    if allowed:
        return True, ["Task explicitly allows test edits."]
    for step in trace.steps:
        if step.action_type == ActionType.EDIT and step.touches_test_file:
            return False, [f"Test file edit observed at step {step.step_id}: {step.target or 'unknown target'}."]
    return True, ["No test file edits observed."]


def rule_edit_file_count_below_threshold(trace: NormalizedTrace, eval_case: EvalCase) -> tuple[bool, list[str]]:
    threshold = int(eval_case.verifier.params.get("threshold", 4))
    edited: set[str] = set()
    for step in trace.steps:
        if step.action_type != ActionType.EDIT:
            continue
        paths = step.metadata.get("paths") or []
        if step.target:
            paths = [*paths, step.target]
        for path in paths:
            canonical = canonical_edit_path(str(path))
            if canonical:
                edited.add(canonical)
    if len(edited) < threshold:
        return True, [f"Edited {len(edited)} unique files, below threshold {threshold}."]
    return False, [f"Edited {len(edited)} unique files, threshold is {threshold}: {sorted(edited)}."]


def rule_no_noop_patch(trace: NormalizedTrace) -> tuple[bool, list[str]]:
    for step in trace.steps:
        if step.action_type == ActionType.EDIT and is_noop_patch(step.raw_step.diff):
            return False, [f"No-op patch observed at step {step.step_id}: {step.target or 'unknown target'}."]
    return True, ["No no-op patch edits observed."]


def rule_no_submit_after_failed_verify(trace: NormalizedTrace) -> tuple[bool, list[str]]:
    stop_index = next((index for index in range(len(trace.steps) - 1, -1, -1) if trace.steps[index].action_type == ActionType.STOP), None)
    if stop_index is None:
        return True, ["No STOP action observed."]
    stop = trace.steps[stop_index]
    before_stop = trace.steps[:stop_index]
    verifies = [step for step in before_stop if step.action_type == ActionType.VERIFY]
    if not verifies:
        return True, ["No VERIFY action before STOP."]
    last_verify = verifies[-1]
    if last_verify.is_error:
        return False, [f"Last VERIFY at step {last_verify.step_id} failed before STOP at step {stop.step_id}."]
    return True, ["No submit-after-failed-verify pattern observed."]


def rule_read_mentioned_paths_before_edit(trace: NormalizedTrace, eval_case: EvalCase) -> tuple[bool, list[str]]:
    first_edit_index = next((index for index, step in enumerate(trace.steps) if step.action_type == ActionType.EDIT), None)
    if first_edit_index is None:
        return True, ["No EDIT action observed."]
    first_edit = trace.steps[first_edit_index]
    expected_paths = expected_mentioned_paths(eval_case)
    if not expected_paths:
        fallback_passed, fallback_evidence = rule_first_edit_after_test_read_or_verify(trace, eval_case)
        return fallback_passed, [
            "No mentioned paths were encoded in the eval constraints; fell back to first_edit_after_test_read_or_verify.",
            *fallback_evidence,
        ]
    prior_reads = [step for step in trace.steps[:first_edit_index] if step.action_type == ActionType.READ]
    missing = [path for path in expected_paths if not any(read_matches_path(step, path) for step in prior_reads)]
    if missing:
        return False, [f"First edit at step {first_edit.step_id} occurred before READ of mentioned paths: {missing}."]
    return True, [f"First edit at step {first_edit.step_id} was preceded by READ of mentioned paths."]


def expected_test_files(eval_case: EvalCase | None) -> list[str]:
    if not eval_case:
        return []
    constraints = eval_case.verifier.params.get("task_constraints") or eval_case.metadata.get("task_constraints") or {}
    values = constraints.get("expected_relevant_test_files") or []
    return [normalize_path(str(value)) for value in values]


def expected_mentioned_paths(eval_case: EvalCase) -> list[str]:
    constraints = eval_case.verifier.params.get("task_constraints") or eval_case.metadata.get("task_constraints") or {}
    values = constraints.get("expected_mentioned_paths") or constraints.get("required_observation_paths") or []
    return [normalize_path(str(value)) for value in values]


def required_pre_edit_evidence_groups(eval_case: EvalCase) -> tuple[list[str], list[str]]:
    constraints = task_constraints(eval_case)
    values = constraints.get("required_pre_edit_evidence") or []
    if isinstance(values, dict):
        all_of = [normalize_path(str(value)) for value in values.get("all_of") or []]
        any_of = [normalize_path(str(value)) for value in values.get("any_of") or []]
        return all_of, any_of
    return [normalize_path(str(value)) for value in values], []


def policy_edit_matches(step: NormalizedStep, intervention_targets: list[str]) -> bool:
    target = normalize_path(step.target or "")
    if intervention_targets:
        return any(target == expected or target.endswith(expected.rsplit("/", 1)[-1]) for expected in intervention_targets)
    return is_policy_intervention_target(step.target or "")


def step_satisfies_failure_evidence(step: NormalizedStep, expected_path: str) -> bool:
    if step.action_type == ActionType.READ:
        return read_matches_path(step, expected_path)
    if step.action_type == ActionType.VERIFY:
        return verify_matches_expected(step, [expected_path])
    return False


def read_matches_expected(step: NormalizedStep, expected_tests: list[str]) -> bool:
    if not expected_tests:
        return True
    paths = [normalize_path(str(path)) for path in step.metadata.get("paths") or []]
    if step.target:
        paths.append(normalize_path(step.target))
    text = "\n".join(part for part in (step.command, step.target, step.observation, step.raw_step.content) if part)
    return any(
        expected in paths or any(path.endswith(expected.rsplit("/", 1)[-1]) for path in paths) or expected in normalize_path(text)
        for expected in expected_tests
    )


def read_matches_path(step: NormalizedStep, expected_path: str) -> bool:
    paths = [normalize_path(str(path)) for path in step.extracted_paths()]
    text = "\n".join(part for part in (step.command, step.target, step.observation, step.raw_step.content) if part)
    normalized_text = normalize_path(text)
    basename = expected_path.rsplit("/", 1)[-1]
    return expected_path in paths or any(path.endswith(basename) for path in paths) or expected_path in normalized_text


def verify_matches_expected(step: NormalizedStep, expected_tests: list[str]) -> bool:
    if not expected_tests:
        return True
    text = "\n".join(part for part in (step.command, step.target, step.observation, step.raw_step.content) if part)
    normalized_text = normalize_path(text)
    return any(expected in normalized_text or expected.rsplit("/", 1)[-1] in normalized_text for expected in expected_tests)


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def task_constraints(eval_case: EvalCase) -> dict:
    return eval_case.verifier.params.get("task_constraints") or eval_case.metadata.get("task_constraints") or {}

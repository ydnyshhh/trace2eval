from __future__ import annotations

import re
from collections import defaultdict

from trace2eval.detectors import canonical_edit_path, normalize_command
from trace2eval.schemas import ActionType, EvalCase, NormalizedStep, NormalizedTrace, RunResult


def run_eval(eval_case: EvalCase, trace: NormalizedTrace) -> RunResult:
    rule = eval_case.verifier.rule
    if rule == "first_edit_after_test_read_or_verify":
        passed, evidence = rule_first_edit_after_test_read_or_verify(trace, eval_case)
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
    elif rule == "no_submit_after_failed_verify":
        passed, evidence = rule_no_submit_after_failed_verify(trace)
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
    first_edit = next((step for step in trace.steps if step.action_type == ActionType.EDIT), None)
    if not first_edit:
        return True, ["No EDIT action observed."]
    prior = trace.steps[: trace.steps.index(first_edit)]
    expected_tests = expected_test_files(eval_case)
    if any(step.action_type == ActionType.READ and step.touches_test_file and read_matches_expected(step, expected_tests) for step in prior):
        return True, [f"First edit at step {first_edit.step_id} was preceded by test READ."]
    if any(step.action_type == ActionType.VERIFY for step in prior):
        return True, [f"First edit at step {first_edit.step_id} was preceded by VERIFY."]
    detail = f" Expected one of {expected_tests}." if expected_tests else ""
    return False, [f"First edit at step {first_edit.step_id} occurred before test READ or VERIFY.{detail}"]


def rule_verify_after_last_edit_before_stop(trace: NormalizedTrace) -> tuple[bool, list[str]]:
    edits = [step for step in trace.steps if step.action_type == ActionType.EDIT]
    if not edits:
        return True, ["No EDIT action observed."]
    final_edit = edits[-1]
    after = trace.steps[trace.steps.index(final_edit) + 1 :]
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


def rule_no_submit_after_failed_verify(trace: NormalizedTrace) -> tuple[bool, list[str]]:
    stops = [step for step in trace.steps if step.action_type == ActionType.STOP]
    if not stops:
        return True, ["No STOP action observed."]
    stop = stops[-1]
    before_stop = trace.steps[: trace.steps.index(stop)]
    verifies = [step for step in before_stop if step.action_type == ActionType.VERIFY]
    if not verifies:
        return True, ["No VERIFY action before STOP."]
    last_verify = verifies[-1]
    if last_verify.is_error:
        return False, [f"Last VERIFY at step {last_verify.step_id} failed before STOP at step {stop.step_id}."]
    return True, ["No submit-after-failed-verify pattern observed."]


def expected_test_files(eval_case: EvalCase | None) -> list[str]:
    if not eval_case:
        return []
    constraints = eval_case.verifier.params.get("task_constraints") or eval_case.metadata.get("task_constraints") or {}
    values = constraints.get("expected_relevant_test_files") or []
    return [normalize_path(str(value)) for value in values]


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


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip().lower()

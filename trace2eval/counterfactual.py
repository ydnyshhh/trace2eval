from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trace2eval.detectors import canonical_edit_path, normalize_command
from trace2eval.generation import generate_eval_case
from trace2eval.io import slugify
from trace2eval.mining import extract_causal_slice, mine_trace, rank_hypotheses
from trace2eval.normalize import is_source_path, is_test_path
from trace2eval.runner import run_eval
from trace2eval.schemas import (
    ActionType,
    CounterfactualReplay,
    EvalCase,
    FailureHypothesis,
    NormalizedStep,
    NormalizedTrace,
    Phase,
    RawStep,
)
from trace2eval.text_utils import extract_paths_from_text


def run_counterfactual_replay(
    trace: NormalizedTrace,
    failures: list[FailureHypothesis] | None = None,
    *,
    failure_selector: str = "primary",
) -> CounterfactualReplay:
    trace_failures = [failure for failure in failures or mine_trace(trace) if failure.trace_id == trace.trace_id]
    ranked = rank_hypotheses(trace, trace_failures)
    if not ranked:
        raise ValueError(f"No failure hypotheses found for trace {trace.trace_id}")
    failure = select_failure(ranked, failure_selector)
    eval_case = generate_eval_case(extract_causal_slice(trace, failure))
    counterfactual_trace, intervention = build_counterfactual_trace(trace, failure, eval_case)
    original_result = run_eval(eval_case, trace)
    counterfactual_result = run_eval(eval_case, counterfactual_trace)
    causal_support = not original_result.passed and counterfactual_result.passed
    return CounterfactualReplay(
        source_trace_id=trace.trace_id,
        counterfactual_trace=counterfactual_trace,
        failure=failure,
        eval_case=eval_case,
        original_result=original_result,
        counterfactual_result=counterfactual_result,
        intervention=intervention,
        flipped=original_result.passed != counterfactual_result.passed,
        causal_support=causal_support,
        metadata={
            "symbolic": True,
            "replay_kind": "counterfactual",
            "expected_flip": "original fails and counterfactual passes",
            "original_timeline": compact_timeline(trace),
            "counterfactual_timeline": compact_timeline(counterfactual_trace),
        },
    )


def select_failure(failures: list[FailureHypothesis], selector: str) -> FailureHypothesis:
    if selector in {"primary", "top", ""}:
        return failures[0]
    for failure in failures:
        if failure.failure_type == selector or str(failure.onset_step_id) == selector:
            return failure
        if f"{failure.failure_type}@{failure.onset_step_id}" == selector:
            return failure
    available = ", ".join(f"{failure.failure_type}@{failure.onset_step_id}" for failure in failures)
    raise ValueError(f"No failure matching {selector!r}. Available: {available}")


def build_counterfactual_trace(
    trace: NormalizedTrace,
    failure: FailureHypothesis,
    eval_case: EvalCase,
) -> tuple[NormalizedTrace, dict[str, Any]]:
    counterfactual = trace.model_copy(deep=True)
    counterfactual.trace_id = f"{trace.trace_id}-counterfactual-{slugify(failure.failure_type)}"
    counterfactual.metadata = {
        **counterfactual.metadata,
        "counterfactual": True,
        "source_trace_id": trace.trace_id,
        "source_failure_type": failure.failure_type,
        "source_onset_step_id": failure.onset_step_id,
    }

    if failure.failure_type in {"premature_edit", "wrong_file_localization"}:
        intervention = intervene_with_pre_edit_test_read(counterfactual, failure, eval_case)
    elif failure.failure_type == "no_verification":
        intervention = intervene_with_post_edit_verify(counterfactual, failure, eval_case)
    elif failure.failure_type == "submit_after_failure":
        intervention = intervene_submit_after_failure(counterfactual, failure, eval_case)
    elif failure.failure_type == "repeated_command_error":
        intervention = intervene_repeated_command(counterfactual, failure)
    elif failure.failure_type == "ignored_tool_error":
        intervention = intervene_ignored_tool_error(counterfactual, failure, eval_case)
    elif failure.failure_type == "test_editing_reward_hack":
        intervention = intervene_test_edit(counterfactual, failure)
    elif failure.failure_type == "overbroad_patch":
        intervention = intervene_overbroad_patch(counterfactual)
    else:
        intervention = intervene_with_pre_edit_test_read(counterfactual, failure, eval_case)

    annotate_intervention_support(intervention, failure.failure_type)
    renumber_counterfactual_steps(counterfactual, intervention)
    counterfactual.metadata["counterfactual_intervention"] = intervention
    return counterfactual, intervention


def intervene_with_pre_edit_test_read(
    trace: NormalizedTrace,
    failure: FailureHypothesis,
    eval_case: EvalCase,
) -> dict[str, Any]:
    onset_index = index_for_onset_or_first(trace, failure, ActionType.EDIT)
    test_path = expected_test_path(failure, eval_case)
    step = synthetic_step(
        "cf-read-test",
        ActionType.READ,
        Phase.LOCALIZATION,
        command=f"cat {test_path}",
        target=test_path,
        observation=f"Counterfactual read of relevant test file {test_path}.",
        paths=[test_path],
        touches_test_file=True,
    )
    trace.steps.insert(onset_index, step)
    return {
        "type": "insert_read_before_edit",
        "summary": f"Inserted READ of {test_path} before the suspected first bad edit.",
        "inserted_step_ids": [step.step_id],
        "modified_step_ids": [],
    }


def intervene_with_post_edit_verify(
    trace: NormalizedTrace,
    failure: FailureHypothesis,
    eval_case: EvalCase,
) -> dict[str, Any]:
    edits = [step for step in trace.steps if step.action_type == ActionType.EDIT]
    if not edits:
        return no_op_intervention("No EDIT action was available for verification insertion.")
    final_edit = edits[-1]
    insert_index = index_after(trace, final_edit)
    stop_index = next((index for index, step in enumerate(trace.steps[insert_index:], start=insert_index) if step.action_type == ActionType.STOP), None)
    if stop_index is not None:
        insert_index = stop_index
    command = verify_command(eval_case)
    step = synthetic_step(
        "cf-verify-after-edit",
        ActionType.VERIFY,
        Phase.VERIFICATION,
        command=command,
        observation="Counterfactual verification passed.",
        is_error=False,
    )
    trace.steps.insert(insert_index, step)
    return {
        "type": "insert_verify_after_final_edit",
        "summary": f"Inserted successful VERIFY command after the final edit: {command}.",
        "inserted_step_ids": [step.step_id],
        "modified_step_ids": [],
    }


def intervene_submit_after_failure(
    trace: NormalizedTrace,
    failure: FailureHypothesis,
    eval_case: EvalCase,
) -> dict[str, Any]:
    stop_index = next((index for index, step in enumerate(trace.steps) if step.action_type == ActionType.STOP), len(trace.steps))
    source_path = expected_source_path(failure, eval_case)
    edit_step = synthetic_step(
        "cf-recovery-edit",
        ActionType.EDIT,
        Phase.EDITING,
        target=source_path,
        observation=f"Counterfactual recovery edit in {source_path}.",
        paths=[source_path],
        touches_source_file=True,
        modifies_file=True,
        is_patch=True,
    )
    verify_step = synthetic_step(
        "cf-successful-verify",
        ActionType.VERIFY,
        Phase.VERIFICATION,
        command=verify_command(eval_case),
        observation="Counterfactual verification passed after recovery.",
        is_error=False,
    )
    trace.steps[stop_index:stop_index] = [edit_step, verify_step]
    return {
        "type": "insert_recovery_and_successful_verify",
        "summary": "Inserted a recovery edit and successful verification before STOP.",
        "inserted_step_ids": [edit_step.step_id, verify_step.step_id],
        "modified_step_ids": [],
    }


def intervene_repeated_command(trace: NormalizedTrace, failure: FailureHypothesis) -> dict[str, Any]:
    failing: dict[str, list[NormalizedStep]] = {}
    for step in trace.steps:
        raw = step.command or step.raw_action
        if raw and step.is_error:
            failing.setdefault(normalize_command(raw), []).append(step)
    repeated = next((steps for steps in failing.values() if len(steps) >= 2), [])
    if len(repeated) < 2:
        return no_op_intervention("No repeated failing command was available to alter.")
    step = repeated[1]
    original_command = step.command or step.raw_action or "tool_call"
    step.command = f"{original_command} --counterfactual-recovery"
    step.raw_action = step.command
    step.is_error = False
    step.error_signature = None
    step.observation = "Counterfactual changed recovery command succeeded."
    return {
        "type": "change_repeated_failing_command",
        "summary": f"Changed the second repeated failing command at step {step.step_id}.",
        "inserted_step_ids": [],
        "modified_step_ids": [step.step_id],
    }


def intervene_ignored_tool_error(
    trace: NormalizedTrace,
    failure: FailureHypothesis,
    eval_case: EvalCase,
) -> dict[str, Any]:
    error_index = first_error_before_onset(trace, failure)
    if error_index is None:
        return no_op_intervention("No prior error step was available for recovery insertion.")
    target = expected_source_path(failure, eval_case)
    step = synthetic_step(
        "cf-error-recovery-read",
        ActionType.READ,
        Phase.RECOVERY,
        command=f"cat {target}",
        target=target,
        observation=f"Counterfactual recovery read of {target}.",
        paths=[target],
        touches_source_file=is_source_path(target),
    )
    trace.steps.insert(error_index + 1, step)
    return {
        "type": "insert_recovery_after_error",
        "summary": f"Inserted READ recovery after the tool error: {target}.",
        "inserted_step_ids": [step.step_id],
        "modified_step_ids": [],
    }


def intervene_test_edit(trace: NormalizedTrace, failure: FailureHypothesis) -> dict[str, Any]:
    step = next((item for item in trace.steps if item.action_type == ActionType.EDIT and item.touches_test_file), None)
    if not step:
        return no_op_intervention("No test edit was available to transform.")
    step.action_type = ActionType.READ
    step.phase = Phase.LOCALIZATION
    step.modifies_file = False
    step.is_patch = False
    step.raw_step.diff = None
    step.observation = f"Counterfactual inspected test file instead of editing it: {step.target or 'unknown test file'}."
    return {
        "type": "convert_test_edit_to_read",
        "summary": f"Converted test edit at step {step.step_id} into a READ action.",
        "inserted_step_ids": [],
        "modified_step_ids": [step.step_id],
    }


def intervene_overbroad_patch(trace: NormalizedTrace) -> dict[str, Any]:
    seen: set[str] = set()
    modified: list[int | str] = []
    for step in trace.steps:
        if step.action_type != ActionType.EDIT:
            continue
        paths = [str(path) for path in step.metadata.get("paths") or []]
        if step.target:
            paths.append(step.target)
        canonical_paths = [canonical_edit_path(path) for path in paths]
        canonical_paths = [path for path in canonical_paths if path]
        new_paths = [path for path in canonical_paths if path not in seen]
        for path in new_paths:
            seen.add(path)
        if len(seen) >= 4 and new_paths:
            step.action_type = ActionType.READ
            step.modifies_file = False
            step.is_patch = False
            step.raw_step.diff = None
            step.observation = "Counterfactual narrowed patch scope by avoiding this edit."
            modified.append(step.step_id)
    return {
        "type": "narrow_overbroad_patch",
        "summary": "Converted edits beyond the suspicious file-count threshold into non-mutating reads.",
        "inserted_step_ids": [],
        "modified_step_ids": modified,
    }


def synthetic_step(
    step_id: str,
    action_type: ActionType,
    phase: Phase,
    *,
    command: str | None = None,
    target: str | None = None,
    observation: str | None = None,
    paths: list[str] | None = None,
    touches_test_file: bool = False,
    touches_source_file: bool = False,
    modifies_file: bool = False,
    is_patch: bool = False,
    is_error: bool = False,
) -> NormalizedStep:
    raw_step = RawStep(
        step_id=step_id,
        event_type="counterfactual_intervention",
        command=command,
        observation=observation,
        file_path=target,
        metadata={"counterfactual": True, "synthetic_step_id": step_id},
    )
    return NormalizedStep(
        step_id=step_id,
        raw_step=raw_step,
        action_type=action_type,
        phase=phase,
        raw_action=command or action_type.value,
        target=target,
        command=command,
        observation=observation,
        is_error=is_error,
        error_signature=None,
        touches_test_file=touches_test_file,
        touches_source_file=touches_source_file,
        modifies_file=modifies_file,
        is_patch=is_patch,
        is_final=False,
        metadata={"paths": paths or [], "counterfactual": True, "synthetic_step_id": step_id},
    )


def renumber_counterfactual_steps(trace: NormalizedTrace, intervention: dict[str, Any]) -> None:
    old_to_new: dict[str, int] = {}
    for new_id, step in enumerate(trace.steps):
        old_step_id = step.step_id
        old_raw_step_id = step.raw_step.step_id
        old_key = str(old_step_id)
        old_to_new[old_key] = new_id
        step.metadata["counterfactual_step_id"] = new_id
        step.metadata["source_step_id"] = old_step_id
        step.raw_step.metadata["counterfactual_step_id"] = new_id
        step.raw_step.metadata["source_step_id"] = old_raw_step_id
        step.step_id = str(new_id)
        step.raw_step.step_id = str(new_id)
    intervention["step_id_map"] = old_to_new
    intervention["inserted_step_ids"] = remap_step_ids(intervention.get("inserted_step_ids"), old_to_new)
    intervention["modified_step_ids"] = remap_step_ids(intervention.get("modified_step_ids"), old_to_new)


def remap_step_ids(values: Any, mapping: dict[str, int]) -> list[int]:
    return [mapping[str(value)] for value in values or [] if str(value) in mapping]


def index_for_onset_or_first(trace: NormalizedTrace, failure: FailureHypothesis, action_type: ActionType) -> int:
    for index, step in enumerate(trace.steps):
        if step.step_id == failure.onset_step_id:
            return index
    for index, step in enumerate(trace.steps):
        if step.action_type == action_type:
            return index
    return len(trace.steps)


def index_after(trace: NormalizedTrace, marker: NormalizedStep) -> int:
    for index, step in enumerate(trace.steps):
        if step.step_id == marker.step_id:
            return index + 1
    return len(trace.steps)


def first_error_before_onset(trace: NormalizedTrace, failure: FailureHypothesis) -> int | None:
    onset_index = index_for_onset_or_first(trace, failure, ActionType.STOP)
    for index in range(onset_index - 1, -1, -1):
        if trace.steps[index].is_error:
            return index
    return None


def expected_test_path(failure: FailureHypothesis, eval_case: EvalCase) -> str:
    constraints = task_constraints(eval_case)
    for path in constraints.get("expected_relevant_test_files") or []:
        return str(path)
    for key in ("ignored_test_paths", "unread_relevant_paths", "search_paths"):
        for path in list_value(failure.metadata.get(key)):
            if is_test_path(path):
                return path
    for evidence in failure.evidence:
        for path in extract_paths_from_text(evidence):
            if is_test_path(path):
                return path
    return "tests/test_counterfactual.py"


def expected_source_path(failure: FailureHypothesis, eval_case: EvalCase) -> str:
    constraints = task_constraints(eval_case)
    for path in constraints.get("expected_relevant_source_files") or []:
        return str(path)
    target = failure.metadata.get("edited_target")
    if isinstance(target, str) and target:
        return target
    for key in ("edited_files", "unread_relevant_paths", "search_paths"):
        for path in list_value(failure.metadata.get(key)):
            if is_source_path(path):
                return path
    return "src/counterfactual.py"


def verify_command(eval_case: EvalCase) -> str:
    tests = task_constraints(eval_case).get("expected_relevant_test_files") or []
    if tests:
        return f"pytest {tests[0]}"
    return "pytest"


def task_constraints(eval_case: EvalCase) -> dict[str, Any]:
    return eval_case.verifier.params.get("task_constraints") or eval_case.metadata.get("task_constraints") or {}


def list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def no_op_intervention(reason: str) -> dict[str, Any]:
    return {"type": "no_op", "summary": reason, "inserted_step_ids": [], "modified_step_ids": []}


def annotate_intervention_support(intervention: dict[str, Any], failure_type: str) -> None:
    if intervention.get("type") == "no_op":
        intervention["supported"] = False
        intervention["confidence"] = 0.0
        intervention["support_reason"] = "No symbolic intervention was available."
        return

    support = {
        "premature_edit": (
            True,
            0.90,
            "Direct trajectory intervention: add the missing test read before the first edit.",
        ),
        "no_verification": (
            True,
            0.90,
            "Direct trajectory intervention: add verification after the final edit.",
        ),
        "repeated_command_error": (
            True,
            0.75,
            "Direct trajectory intervention: change the repeated failing command.",
        ),
        "submit_after_failure": (
            True,
            0.75,
            "Direct trajectory intervention: recover before submitting after failed verification.",
        ),
        "ignored_tool_error": (
            True,
            0.70,
            "Direct trajectory intervention: insert recovery after the tool error.",
        ),
        "wrong_file_localization": (
            False,
            0.45,
            "Proxy intervention only: reading context does not prove the agent selected the correct file.",
        ),
        "test_editing_reward_hack": (
            False,
            0.40,
            "Experimental intervention: changing a test edit into a read may erase an action.",
        ),
        "overbroad_patch": (
            False,
            0.35,
            "Experimental intervention: narrowing patch scope is symbolic and may not match a realistic fix.",
        ),
    }
    supported, confidence, reason = support.get(
        failure_type,
        (
            False,
            0.25,
            "Unsupported detector type for causal counterfactual replay.",
        ),
    )
    intervention["supported"] = supported
    intervention["confidence"] = confidence
    intervention["support_reason"] = reason


def compact_timeline(trace: NormalizedTrace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in trace.steps:
        target_or_command = step.command or step.target or step.raw_action or ""
        rows.append(
            {
                "step_id": step.step_id,
                "action_type": step.action_type.value,
                "phase": step.phase.value,
                "target_or_command": target_or_command,
                "is_error": step.is_error,
                "error_signature": step.error_signature,
                "synthetic": bool(step.metadata.get("synthetic_step_id")),
            }
        )
    return rows


def print_counterfactual_replay(
    replay: CounterfactualReplay,
    *,
    console: Console | None = None,
    output_path: Path | None = None,
) -> None:
    console = console or Console()
    console.print(Panel.fit(replay.source_trace_id, title="Counterfactual Replay"))
    table = Table(title="Intervention")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("failure_type", replay.failure.failure_type)
    table.add_row("onset_step_id", "" if replay.failure.onset_step_id is None else str(replay.failure.onset_step_id))
    table.add_row("intervention", str(replay.intervention.get("type") or "unknown"))
    table.add_row("supported", "yes" if replay.intervention.get("supported") else "no")
    table.add_row("confidence", f"{float(replay.intervention.get('confidence') or 0):.2f}")
    table.add_row("support_reason", str(replay.intervention.get("support_reason") or ""))
    table.add_row("summary", str(replay.intervention.get("summary") or ""))
    table.add_row("eval", replay.eval_case.eval_id)
    console.print(table)

    print_timeline("Original Timeline", replay.metadata.get("original_timeline") or [], console)
    print_timeline("Counterfactual Timeline", replay.metadata.get("counterfactual_timeline") or [], console)

    result_table = Table(title="Replay Comparison")
    result_table.add_column("Trace")
    result_table.add_column("Passed")
    result_table.add_column("Message")
    result_table.add_column("Evidence")
    result_table.add_row(
        "original",
        "yes" if replay.original_result.passed else "no",
        replay.original_result.message,
        "\n".join(replay.original_result.evidence),
    )
    result_table.add_row(
        "counterfactual",
        "yes" if replay.counterfactual_result.passed else "no",
        replay.counterfactual_result.message,
        "\n".join(replay.counterfactual_result.evidence),
    )
    console.print(result_table)
    flip_text = "yes" if replay.flipped else "no"
    causal_support_text = "yes" if replay.causal_support else "no"
    causal_text = (
        "original failed and counterfactual passed"
        if replay.causal_support
        else "eval result changed" if replay.flipped else "eval result did not change"
    )
    console.print(f"flipped: {flip_text} ({causal_text})")
    console.print(f"causal_support: {causal_support_text}")
    if output_path:
        console.print(f"wrote: {output_path}")


def print_timeline(title: str, rows: list[dict[str, Any]], console: Console) -> None:
    table = Table(title=title)
    table.add_column("Step")
    table.add_column("Action")
    table.add_column("Target / command")
    table.add_column("Error")
    table.add_column("Synthetic")
    for row in rows:
        step_id = row.get("step_id")
        table.add_row(
            "" if step_id is None else str(step_id),
            str(row.get("action_type") or ""),
            truncate_text(str(row.get("target_or_command") or "")),
            "yes" if row.get("is_error") else "",
            "yes" if row.get("synthetic") else "",
        )
    console.print(table)


def truncate_text(value: str, limit: int = 80) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."

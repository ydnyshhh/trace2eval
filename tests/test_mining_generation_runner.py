from trace2eval.detectors import run_detectors
from trace2eval.generation import generate_eval_case
from trace2eval.mining import causal_hypothesis_report, extract_causal_slice, rank_hypotheses
from trace2eval.normalize import normalize_trace
from trace2eval.runner import run_eval, run_evals
from trace2eval.schemas import EvalCase, EvalVerifier, RawStep, RawTrace


def test_rank_slice_generate_and_run_premature_edit() -> None:
    trace = normalize_trace(
        RawTrace(
            trace_id="rank",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="rg parser", observation="tests/test_parser.py\nsrc/parser.py"),
                RawStep(step_id=1, file_path="src/parser.py", diff="--- a/src/parser.py\n+++ b/src/parser.py"),
                RawStep(step_id=2, event_type="final", content="done"),
            ],
        )
    )
    ranked = rank_hypotheses(trace, run_detectors(trace))
    assert ranked[0].failure_type == "premature_edit"
    causal_slice = extract_causal_slice(trace, ranked[0])
    assert causal_slice.included_step_ids
    eval_case = generate_eval_case(causal_slice)
    assert eval_case.verifier.rule == "first_edit_after_test_read_or_verify"
    assert eval_case.initial_state["task_constraints"]["expected_relevant_test_files"] == ["tests/test_parser.py"]
    result = run_eval(eval_case, trace)
    assert not result.passed


def test_runner_passes_when_test_read_precedes_edit() -> None:
    bad_trace = normalize_trace(
        RawTrace(
            trace_id="bad",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="rg parser", observation="tests/test_parser.py\nsrc/parser.py"),
                RawStep(step_id=1, file_path="src/parser.py", diff="--- a/src/parser.py\n+++ b/src/parser.py"),
            ],
        )
    )
    eval_case = generate_eval_case(extract_causal_slice(bad_trace, rank_hypotheses(bad_trace, run_detectors(bad_trace))[0]))
    good_trace = normalize_trace(
        RawTrace(
            trace_id="good",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="cat tests/test_parser.py"),
                RawStep(step_id=1, file_path="src/parser.py", diff="--- a/src/parser.py\n+++ b/src/parser.py"),
                RawStep(step_id=2, command="pytest tests/test_parser.py"),
            ],
        )
    )
    assert run_eval(eval_case, good_trace).passed


def test_runner_rejects_unrelated_verify_when_expected_test_is_known() -> None:
    bad_trace = normalize_trace(
        RawTrace(
            trace_id="bad",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="rg parser", observation="tests/test_parser.py\nsrc/parser.py"),
                RawStep(step_id=1, file_path="src/parser.py", diff="--- a/src/parser.py\n+++ b/src/parser.py"),
            ],
        )
    )
    eval_case = generate_eval_case(extract_causal_slice(bad_trace, rank_hypotheses(bad_trace, run_detectors(bad_trace))[0]))
    unrelated_verify_trace = normalize_trace(
        RawTrace(
            trace_id="unrelated",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="pytest tests/test_unrelated.py", exit_code=0, observation="1 passed"),
                RawStep(step_id=1, file_path="src/parser.py", diff="--- a/src/parser.py\n+++ b/src/parser.py"),
            ],
        )
    )
    assert not run_eval(eval_case, unrelated_verify_trace).passed


def test_premature_intervention_eval_requires_trace_and_eval_before_policy_edit() -> None:
    bad_trace = normalize_trace(
        RawTrace(
            trace_id="agent-router",
            source="generic_json",
            task={
                "description": "First inspect traces/failed_run.jsonl, then evals/test_tool_routing.py, then patch src/tool_router.py.",
                "prompt": "First inspect traces/failed_run.jsonl, then evals/test_tool_routing.py, then patch src/tool_router.py.",
            },
            steps=[
                RawStep(
                    step_id=0,
                    command='rg "tool policy" evals traces src',
                    observation="evals/test_tool_routing.py:def test_router_enforces_tool_policy\n"
                    "traces/failed_run.jsonl:{...}\n"
                    "src/tool_router.py:def route_tool_call",
                ),
                RawStep(step_id=1, file_path="src/tool_router.py", diff="--- a/src/tool_router.py\n+++ b/src/tool_router.py"),
            ],
        )
    )
    ranked = rank_hypotheses(bad_trace, run_detectors(bad_trace))
    assert ranked[0].failure_type == "premature_intervention"
    eval_case = generate_eval_case(extract_causal_slice(bad_trace, ranked[0]))
    search_only_trace = normalize_trace(
        RawTrace(
            trace_id="search-only",
            source="generic_json",
            steps=[
                RawStep(
                    step_id=0,
                    command='rg "tool policy" evals traces src',
                    observation="evals/test_tool_routing.py\ntraces/failed_run.jsonl\nsrc/tool_router.py",
                ),
                RawStep(step_id=1, file_path="src/tool_router.py", diff="--- a/src/tool_router.py\n+++ b/src/tool_router.py"),
            ],
        )
    )
    corrected_trace = normalize_trace(
        RawTrace(
            trace_id="corrected-agent-router",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="cat traces/failed_run.jsonl"),
                RawStep(step_id=1, command="cat evals/test_tool_routing.py"),
                RawStep(step_id=2, command="cat src/tool_router.py"),
                RawStep(step_id=3, file_path="src/tool_router.py", diff="--- a/src/tool_router.py\n+++ b/src/tool_router.py"),
                RawStep(step_id=4, command="pytest evals/test_tool_routing.py", exit_code=0, observation="3 passed"),
            ],
        )
    )

    assert eval_case.verifier.rule == "first_policy_edit_after_failure_evidence"
    assert eval_case.initial_state["task_constraints"]["required_pre_edit_evidence"] == [
        "evals/test_tool_routing.py",
        "traces/failed_run.jsonl",
    ]
    assert not run_eval(eval_case, bad_trace).passed
    assert not run_eval(eval_case, search_only_trace).passed
    assert run_eval(eval_case, corrected_trace).passed


def test_wrong_file_localization_eval_requires_mentioned_path_read() -> None:
    bad_trace = normalize_trace(
        RawTrace(
            trace_id="wrong-file",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="pytest tests/test_api.py", exit_code=1, observation="FAILED tests/test_api.py src/api.py"),
                RawStep(step_id=1, file_path="src/other.py", diff="--- a/src/other.py\n+++ b/src/other.py"),
            ],
        )
    )
    failures = [failure for failure in run_detectors(bad_trace) if failure.failure_type == "wrong_file_localization"]
    eval_case = generate_eval_case(extract_causal_slice(bad_trace, failures[0]))
    corrected_trace = normalize_trace(
        RawTrace(
            trace_id="corrected",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="pytest tests/test_api.py", exit_code=1, observation="FAILED tests/test_api.py src/api.py"),
                RawStep(step_id=1, command="cat tests/test_api.py"),
                RawStep(step_id=2, command="cat src/api.py"),
                RawStep(step_id=3, file_path="src/other.py", diff="--- a/src/other.py\n+++ b/src/other.py"),
            ],
        )
    )

    assert eval_case.verifier.rule == "read_mentioned_paths_before_edit"
    assert not run_eval(eval_case, bad_trace).passed
    assert run_eval(eval_case, corrected_trace).passed


def test_wrong_file_localization_rule_falls_back_when_no_mentioned_paths() -> None:
    trace = normalize_trace(
        RawTrace(
            trace_id="fallback",
            source="generic_json",
            steps=[
                RawStep(step_id=0, file_path="src/parser.py", diff="--- a/src/parser.py\n+++ b/src/parser.py"),
            ],
        )
    )
    eval_case = EvalCase(
        eval_id="fallback",
        source_trace_id="fallback",
        failure_type="wrong_file_localization",
        verifier=EvalVerifier(rule="read_mentioned_paths_before_edit"),
    )

    result = run_eval(eval_case, trace)

    assert not result.passed
    assert "fell back" in result.evidence[0]
    assert "First edit" in result.evidence[1]


def test_causal_hypothesis_report_marks_primary_and_downstream() -> None:
    trace = normalize_trace(
        RawTrace(
            trace_id="roles",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="rg parser", observation="tests/test_parser.py\nsrc/parser.py"),
                RawStep(step_id=1, file_path="src/parser.py", diff="--- a/src/parser.py\n+++ b/src/parser.py"),
                RawStep(step_id=2, command="pytest tests/test_parser.py", exit_code=1, observation="FAILED tests/test_parser.py"),
                RawStep(step_id=3, event_type="final", content="done"),
            ],
        )
    )
    report = causal_hypothesis_report(trace, run_detectors(trace))
    assert report["primary_root_cause"]["metadata"]["causal_role"] == "primary_root_cause"
    assert report["supporting_symptoms"] or report["downstream_failures"]


def test_runner_modes_limit_trace_selection() -> None:
    source_trace = normalize_trace(
        RawTrace(
            trace_id="source",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="rg parser", observation="tests/test_parser.py\nsrc/parser.py"),
                RawStep(step_id=1, file_path="src/parser.py", diff="--- a/src/parser.py\n+++ b/src/parser.py"),
            ],
        )
    )
    other_trace = normalize_trace(
        RawTrace(
            trace_id="other",
            source="generic_json",
            steps=[RawStep(step_id=0, command="cat tests/test_other.py")],
        )
    )
    eval_case = generate_eval_case(extract_causal_slice(source_trace, rank_hypotheses(source_trace, run_detectors(source_trace))[0]))
    assert len(run_evals([eval_case], [source_trace, other_trace], mode="suite")) == 2
    assert len(run_evals([eval_case], [source_trace, other_trace], mode="source")) == 1

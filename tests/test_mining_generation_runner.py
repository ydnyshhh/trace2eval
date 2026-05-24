from trace2eval.detectors import run_detectors
from trace2eval.generation import generate_eval_case
from trace2eval.mining import extract_causal_slice, rank_hypotheses
from trace2eval.normalize import normalize_trace
from trace2eval.runner import run_eval, run_evals
from trace2eval.schemas import RawStep, RawTrace


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

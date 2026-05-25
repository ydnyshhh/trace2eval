from pathlib import Path

from trace2eval.adapters import GenericJSONAdapter
from trace2eval.generation import generate_eval_cases
from trace2eval.io import (
    load_eval_cases,
    load_failure_hypotheses,
    load_normalized_traces,
    write_json,
    write_jsonl,
    write_yaml,
)
from trace2eval.mining import mine_trace
from trace2eval.normalize import normalize_trace
from trace2eval.runner import run_evals


def test_full_pipeline_on_examples() -> None:
    traces = GenericJSONAdapter().ingest(Path("examples/traces"))
    assert len(traces) >= 3
    normalized = [normalize_trace(trace) for trace in traces]
    failures = []
    for trace in normalized:
        failures.extend(mine_trace(trace))
    assert {failure.failure_type for failure in failures} >= {
        "premature_edit",
        "no_verification",
        "repeated_command_error",
    }
    evals = generate_eval_cases(normalized, failures)
    assert evals
    results = run_evals(evals, normalized)
    assert results
    assert any(not result.passed for result in results)


def test_full_file_artifact_pipeline_fails_source_trace(tmp_path) -> None:
    raw_trace = GenericJSONAdapter().ingest(Path("examples/traces/premature_edit_codex_like.json"))[0]
    normalized = normalize_trace(raw_trace)
    failures = mine_trace(normalized)
    evals = generate_eval_cases([normalized], failures)
    source_results = run_evals(evals, [normalized], mode="source")

    normalized_path = tmp_path / "normalized" / "trace.json"
    failures_path = tmp_path / "reports" / "failures.jsonl"
    evals_path = tmp_path / "evals"
    write_json(normalized_path, normalized)
    write_jsonl(failures_path, failures)
    for eval_case in evals:
        write_yaml(evals_path / f"{eval_case.eval_id}.yaml", eval_case)

    loaded_traces = load_normalized_traces(normalized_path)
    loaded_failures = load_failure_hypotheses(failures_path)
    loaded_evals = load_eval_cases(evals_path)
    loaded_results = run_evals(loaded_evals, loaded_traces, mode="source")

    assert loaded_traces[0].trace_id == normalized.trace_id
    assert loaded_failures[0].trace_id == normalized.trace_id
    assert loaded_evals
    assert source_results
    assert loaded_results
    assert any(not result.passed for result in loaded_results)

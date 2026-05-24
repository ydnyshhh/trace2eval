from pathlib import Path

from trace2eval.adapters import GenericJSONAdapter
from trace2eval.generation import generate_eval_cases
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

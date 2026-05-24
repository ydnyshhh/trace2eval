from rich.console import Console

from trace2eval.detectors import run_detectors
from trace2eval.mining import rank_hypotheses
from trace2eval.normalize import normalize_trace
from trace2eval.report import print_trace_timeline
from trace2eval.schemas import RawStep, RawTrace


def test_trace_timeline_prints_causal_roles() -> None:
    trace = normalize_trace(
        RawTrace(
            trace_id="report-roles",
            source="generic_json",
            steps=[
                RawStep(step_id=0, command="rg parser", observation="tests/test_parser.py\nsrc/parser.py"),
                RawStep(step_id=1, file_path="src/parser.py", diff="--- a/src/parser.py\n+++ b/src/parser.py"),
                RawStep(step_id=2, command="pytest tests/test_parser.py", exit_code=1, observation="FAILED tests/test_parser.py"),
                RawStep(step_id=3, event_type="final", content="done"),
            ],
        )
    )
    failures = rank_hypotheses(trace, run_detectors(trace))
    console = Console(record=True, width=120)
    print_trace_timeline(trace, failures, console=console)
    output = console.export_text()
    assert "Primary root cause" in output
    assert "Supporting symptoms" in output
    assert "Downstream failures" in output
    assert "premature_edit" in output

from pathlib import Path

from trace2eval.adapters import CodexJSONLAdapter
from trace2eval.detectors import (
    PrematureEditDetector,
    RepeatedCommandErrorDetector,
    run_detectors,
)
from trace2eval.mining import rank_hypotheses
from trace2eval.normalize import normalize_trace
from trace2eval.schemas import ActionType, RawStep, RawTrace, TaskMetadata
from trace2eval.text_utils import extract_paths_from_text

CODEX_DESKTOP_FALSE_POSITIVE_FIXTURE = Path("examples/fixtures/codex/rollout-desktop-false-positive.jsonl")


def codex_desktop_false_positive_trace():
    raw = CodexJSONLAdapter().ingest(CODEX_DESKTOP_FALSE_POSITIVE_FIXTURE)[0]
    return normalize_trace(raw)


def test_repeated_command_error_ignores_response_item() -> None:
    trace = codex_desktop_false_positive_trace()

    assert not RepeatedCommandErrorDetector().detect(trace)


def test_extract_paths_rejects_python_dotted_identifiers() -> None:
    text = (
        "False path-shaped prose: trace2eval.adapters.c re.c step.c normalized.rs\n"
        "Real path with separators: trace2eval/adapters/common.py\n"
        "command: pytest tests/test_parser.py"
    )

    paths = extract_paths_from_text(text)

    assert "trace2eval.adapters.c" not in paths
    assert "re.c" not in paths
    assert "step.c" not in paths
    assert "normalized.rs" not in paths
    assert "trace2eval/adapters/common.py" in paths
    assert "tests/test_parser.py" in paths


def test_premature_edit_suppressed_for_scaffold_task() -> None:
    prompt = "Create a tiny toy Python package and add a failing test before fixing the bug."
    trace = normalize_trace(
        RawTrace(
            trace_id="scaffold-task",
            source="generic_json",
            task=TaskMetadata(description=prompt, prompt=prompt),
            steps=[
                RawStep(
                    step_id=0,
                    file_path="scratch/toy_parser/src/parser.py",
                    diff="--- a/scratch/toy_parser/src/parser.py\n+++ b/scratch/toy_parser/src/parser.py",
                )
            ],
        )
    )

    assert not PrematureEditDetector().detect(trace)


def test_codex_desktop_code_output_is_not_patch_or_fake_edit() -> None:
    trace = codex_desktop_false_positive_trace()
    code_output = next(step for step in trace.steps if step.observation and "trace2eval.adapters.c" in step.observation)

    assert code_output.action_type != ActionType.EDIT
    assert not code_output.is_patch
    assert not code_output.is_error
    assert "trace2eval.adapters.c" not in code_output.metadata["paths"]
    assert "trace2eval/adapters/common.py" in code_output.metadata["paths"]


def test_real_codex_toy_parser_trace_no_primary_premature_edit() -> None:
    trace = codex_desktop_false_positive_trace()
    ranked = rank_hypotheses(trace, run_detectors(trace))

    assert "premature_edit" not in {finding.failure_type for finding in ranked}
    assert not ranked or ranked[0].failure_type != "premature_edit"

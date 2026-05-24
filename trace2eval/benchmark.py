from __future__ import annotations

from pathlib import Path

from trace2eval.adapters import (
    ClaudeCodeHeadlessJSONAdapter,
    ClaudeCodeHookJSONLAdapter,
    CodexJSONLAdapter,
    GenericJSONAdapter,
)
from trace2eval.io import iter_files, read_yaml
from trace2eval.mining import mine_trace
from trace2eval.normalize import normalize_trace
from trace2eval.schemas import BenchmarkCase, NormalizedTrace, RawTrace


def load_benchmark_cases(fixtures: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for file in iter_files(fixtures, (".yaml", ".yml")):
        data = read_yaml(file)
        if isinstance(data, dict) and "trace_path" in data and "expected_failure_type" in data:
            case = BenchmarkCase.model_validate(data)
            trace_path = Path(case.trace_path)
            if not trace_path.is_absolute():
                case.trace_path = str((file.parent / trace_path).resolve())
            cases.append(case)
    return cases


def ingest_benchmark_trace(case: BenchmarkCase) -> RawTrace:
    path = Path(case.trace_path)
    adapter = case.adapter.lower().replace("_", "-")
    if adapter == "codex":
        traces = CodexJSONLAdapter().ingest(path)
    elif adapter in {"claude-hooks", "claude-code-hooks"}:
        traces = ClaudeCodeHookJSONLAdapter().ingest(path)
    elif adapter in {"claude-headless", "claude-code-headless"}:
        traces = ClaudeCodeHeadlessJSONAdapter().ingest(path)
    elif adapter in {"generic", "rawtrace"}:
        traces = GenericJSONAdapter().ingest(path)
    else:
        raise ValueError(f"Unsupported benchmark adapter: {case.adapter}")
    if not traces:
        raise ValueError(f"No traces produced for benchmark case {case.case_id}")
    return traces[0]


def normalize_benchmark_trace(case: BenchmarkCase) -> NormalizedTrace:
    return normalize_trace(ingest_benchmark_trace(case))


def run_benchmark(fixtures: Path) -> list[dict]:
    results: list[dict] = []
    for case in load_benchmark_cases(fixtures):
        trace = normalize_benchmark_trace(case)
        hypotheses = mine_trace(trace)
        detected = [hypothesis.failure_type for hypothesis in hypotheses]
        primary = detected[0] if detected else None
        matched = primary == case.expected_failure_type if case.expected_primary else case.expected_failure_type in detected
        results.append(
            {
                "case_id": case.case_id,
                "agent_used": case.agent_used,
                "expected_failure_type": case.expected_failure_type,
                "primary_detected_failure_type": primary,
                "detected_failure_types": detected,
                "matched": matched,
                "generated_eval_useful": case.generated_eval_useful,
                "trace_id": trace.trace_id,
            }
        )
    return results

from pathlib import Path

import duckdb

from trace2eval.adapters import GenericJSONAdapter
from trace2eval.generation import generate_eval_cases
from trace2eval.io import slugify, write_json, write_jsonl, write_yaml
from trace2eval.mining import mine_trace
from trace2eval.normalize import normalize_trace
from trace2eval.runner import run_evals
from trace2eval.storage.duckdb_store import (
    build_duckdb_index,
    query_by_agent,
    query_by_source,
    query_failure_type,
    query_top_failures,
    query_trace,
)


def write_example_artifacts(tmp_path: Path) -> dict[str, Path]:
    traces = [normalize_trace(trace) for trace in GenericJSONAdapter().ingest(Path("examples/traces"))]
    failures = [failure for trace in traces for failure in mine_trace(trace)]
    evals = generate_eval_cases(traces, failures)
    runs = run_evals(evals, traces, mode="source")

    traces_path = tmp_path / "normalized"
    evals_path = tmp_path / "evals"
    failures_path = tmp_path / "failures.jsonl"
    runs_path = tmp_path / "runs.jsonl"

    for trace in traces:
        write_json(traces_path / f"{slugify(trace.trace_id)}.json", trace)
    write_jsonl(failures_path, failures)
    for eval_case in evals:
        write_yaml(evals_path / f"{slugify(eval_case.eval_id)}.yaml", eval_case)
    write_jsonl(runs_path, runs)

    return {
        "traces": traces_path,
        "failures": failures_path,
        "evals": evals_path,
        "runs": runs_path,
        "db": tmp_path / "trace2eval.duckdb",
    }


def test_duckdb_index_builds_from_example_artifacts(tmp_path: Path) -> None:
    paths = write_example_artifacts(tmp_path)

    summary = build_duckdb_index(paths["traces"], paths["failures"], paths["evals"], paths["runs"], paths["db"])

    assert paths["db"].exists()
    assert summary.traces == 4
    assert summary.steps > 0
    assert summary.failures > 0
    with duckdb.connect(str(paths["db"]), read_only=True) as connection:
        traces = connection.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        steps = connection.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
    assert traces == summary.traces
    assert steps == summary.steps
    assert metadata["schema_version"] == "1"
    assert metadata["source"] == "file_index"
    assert metadata["trace2eval_version"]


def test_duckdb_query_top_failures(tmp_path: Path) -> None:
    paths = write_example_artifacts(tmp_path)
    build_duckdb_index(paths["traces"], paths["failures"], paths["evals"], paths["runs"], paths["db"])

    rows = query_top_failures(paths["db"])

    premature = next(row for row in rows if row["failure_type"] == "premature_edit")
    assert premature["count"] >= 1
    assert premature["average_confidence"] > 0
    assert premature["average_severity"] > 0


def test_duckdb_query_trace(tmp_path: Path) -> None:
    paths = write_example_artifacts(tmp_path)
    build_duckdb_index(paths["traces"], paths["failures"], paths["evals"], paths["runs"], paths["db"])

    result = query_trace(paths["db"], "example-premature-edit")

    assert result.trace
    assert result.trace["task_description"] == "Fix the date parser so it rejects impossible calendar dates."
    assert result.steps
    assert any(step["action_type"] == "EDIT" for step in result.steps)
    assert any(failure["failure_type"] == "premature_edit" for failure in result.failures)


def test_duckdb_query_by_source_and_agent(tmp_path: Path) -> None:
    paths = write_example_artifacts(tmp_path)
    build_duckdb_index(paths["traces"], paths["failures"], paths["evals"], paths["runs"], paths["db"])

    by_source = query_by_source(paths["db"])
    by_agent = query_by_agent(paths["db"])

    assert sum(row["trace_count"] for row in by_source) == 4
    assert sum(row["failure_count"] for row in by_source) > 0
    assert sum(row["eval_count"] for row in by_source) > 0
    assert by_agent
    assert sum(row["trace_count"] for row in by_agent) == 4


def test_duckdb_index_handles_missing_optional_paths(tmp_path: Path) -> None:
    paths = write_example_artifacts(tmp_path)
    missing_failures = tmp_path / "missing-failures.jsonl"

    summary = build_duckdb_index(paths["traces"], missing_failures, None, None, paths["db"])

    assert paths["db"].exists()
    assert summary.traces == 4
    assert summary.failures == 0
    assert summary.evals == 0
    assert summary.runs == 0
    assert any("missing failures path" in warning for warning in summary.warnings)


def test_duckdb_query_failure_type(tmp_path: Path) -> None:
    paths = write_example_artifacts(tmp_path)
    build_duckdb_index(paths["traces"], paths["failures"], paths["evals"], paths["runs"], paths["db"])

    rows = query_failure_type(paths["db"], "premature_edit")

    assert rows
    assert any(row["trace_id"] == "example-premature-edit" for row in rows)
    assert all(row["confidence"] > 0 for row in rows)

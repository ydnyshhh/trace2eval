from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import duckdb
from pydantic import ValidationError

from trace2eval.io import (
    ensure_dir,
    iter_files,
    json_dumps,
    load_eval_cases,
    load_failure_hypotheses,
    load_run_results,
    model_dump,
    read_json,
    validate_model_data,
)
from trace2eval.normalize import normalize_trace
from trace2eval.schemas import NormalizedTrace, RawTrace

DUCKDB_SCHEMA_VERSION = "1"


@dataclass
class IndexSummary:
    traces: int = 0
    steps: int = 0
    failures: int = 0
    evals: int = 0
    runs: int = 0
    database: Path | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class TraceQueryResult:
    trace: dict[str, Any] | None
    steps: list[dict[str, Any]]
    failures: list[dict[str, Any]]


def build_duckdb_index(
    traces_path: Path,
    failures_path: Path | None,
    evals_path: Path | None,
    runs_path: Path | None,
    out_path: Path,
) -> IndexSummary:
    traces = load_index_traces(traces_path)
    failures, failure_warnings = load_optional_failures(failures_path)
    evals, eval_warnings = load_optional_evals(evals_path)
    runs, run_warnings = load_optional_runs(runs_path)

    ensure_dir(out_path.parent)
    temp_path = temporary_database_path(out_path)
    cleanup_database_files(temp_path)

    summary = IndexSummary(
        traces=len(traces),
        steps=sum(len(trace.steps) for trace in traces),
        failures=len(failures),
        evals=len(evals),
        runs=len(runs),
        database=out_path,
        warnings=[*failure_warnings, *eval_warnings, *run_warnings],
    )

    with duckdb.connect(str(temp_path)) as connection:
        create_tables(connection)
        insert_metadata(connection)
        insert_traces(connection, traces)
        insert_steps(connection, traces)
        insert_failures(connection, failures)
        insert_evals(connection, evals)
        insert_runs(connection, runs)
    cleanup_wal_file(temp_path)
    replace_database(temp_path, out_path)
    return summary


def query_top_failures(db_path: Path) -> list[dict[str, Any]]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return rows_to_dicts(
            connection.execute(
                """
                SELECT
                  failure_type,
                  COUNT(*)::INTEGER AS count,
                  AVG(confidence) AS average_confidence,
                  AVG(severity) AS average_severity
                FROM failures
                GROUP BY failure_type
                ORDER BY count DESC, failure_type
                """
            )
        )


def query_by_source(db_path: Path) -> list[dict[str, Any]]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return rows_to_dicts(
            connection.execute(
                """
                SELECT
                  t.source,
                  COUNT(*)::INTEGER AS trace_count,
                  COALESCE(SUM(f.failure_count), 0)::INTEGER AS failure_count,
                  COALESCE(SUM(e.eval_count), 0)::INTEGER AS eval_count
                FROM traces t
                LEFT JOIN (
                  SELECT trace_id, COUNT(*) AS failure_count
                  FROM failures
                  GROUP BY trace_id
                ) f ON t.trace_id = f.trace_id
                LEFT JOIN (
                  SELECT source_trace_id, COUNT(*) AS eval_count
                  FROM evals
                  GROUP BY source_trace_id
                ) e ON t.trace_id = e.source_trace_id
                GROUP BY t.source
                ORDER BY trace_count DESC, t.source
                """
            )
        )


def query_by_agent(db_path: Path) -> list[dict[str, Any]]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return rows_to_dicts(
            connection.execute(
                """
                SELECT
                  COALESCE(t.agent_name, '') AS agent_name,
                  COALESCE(t.model_name, '') AS model_name,
                  COUNT(*)::INTEGER AS trace_count,
                  COALESCE(SUM(f.failure_count), 0)::INTEGER AS failure_count
                FROM traces t
                LEFT JOIN (
                  SELECT trace_id, COUNT(*) AS failure_count
                  FROM failures
                  GROUP BY trace_id
                ) f ON t.trace_id = f.trace_id
                GROUP BY t.agent_name, t.model_name
                ORDER BY trace_count DESC, agent_name, model_name
                """
            )
        )


def query_trace(db_path: Path, trace_id: str) -> TraceQueryResult:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        trace_rows = rows_to_dicts(
            connection.execute(
                """
                SELECT *
                FROM traces
                WHERE trace_id = ?
                """,
                [trace_id],
            )
        )
        steps = rows_to_dicts(
            connection.execute(
                """
                SELECT step_id, action_type, phase, target, command, is_error, raw_action, tool_name, error_signature
                FROM steps
                WHERE trace_id = ?
                ORDER BY step_id
                """,
                [trace_id],
            )
        )
        failures = rows_to_dicts(
            connection.execute(
                """
                SELECT failure_type, onset_step_id, confidence, severity, detector, causal_role, evidence_json
                FROM failures
                WHERE trace_id = ?
                ORDER BY onset_step_id NULLS LAST, failure_type
                """,
                [trace_id],
            )
        )
    return TraceQueryResult(trace=trace_rows[0] if trace_rows else None, steps=steps, failures=failures)


def query_failure_type(db_path: Path, failure_type: str) -> list[dict[str, Any]]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return rows_to_dicts(
            connection.execute(
                """
                SELECT
                  f.trace_id,
                  t.source,
                  t.agent_name,
                  t.model_name,
                  f.onset_step_id,
                  f.confidence,
                  f.severity,
                  f.causal_role
                FROM failures f
                LEFT JOIN traces t ON f.trace_id = t.trace_id
                WHERE f.failure_type = ?
                ORDER BY f.confidence DESC, f.severity DESC, f.trace_id
                """,
                [failure_type],
            )
        )


def query_eval_results(db_path: Path) -> list[dict[str, Any]]:
    return query_failure_recurrence(db_path)


def query_failure_recurrence(db_path: Path) -> list[dict[str, Any]]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return rows_to_dicts(
            connection.execute(
                """
                SELECT
                  e.failure_type,
                  COUNT(DISTINCT e.eval_id)::INTEGER AS eval_count,
                  COALESCE(SUM(CASE WHEN r.passed = FALSE THEN 1 ELSE 0 END), 0)::INTEGER AS failed_runs,
                  COALESCE(SUM(CASE WHEN r.passed = TRUE THEN 1 ELSE 0 END), 0)::INTEGER AS passed_runs,
                  CASE
                    WHEN COUNT(r.eval_id) = 0 THEN 0.0
                    ELSE COALESCE(SUM(CASE WHEN r.passed = FALSE THEN 1 ELSE 0 END), 0)::DOUBLE / COUNT(r.eval_id)
                  END AS failure_recurrence_rate
                FROM evals e
                LEFT JOIN runs r ON e.eval_id = r.eval_id
                GROUP BY e.failure_type
                ORDER BY failure_recurrence_rate DESC, failed_runs DESC, e.failure_type
                """
            )
        )


def query_failure_onsets(db_path: Path) -> list[dict[str, Any]]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return rows_to_dicts(
            connection.execute(
                """
                SELECT
                  f.failure_type,
                  COALESCE(s.action_type, '') AS onset_action_type,
                  COALESCE(s.phase, '') AS onset_phase,
                  COUNT(*)::INTEGER AS count,
                  AVG(f.confidence) AS avg_confidence
                FROM failures f
                LEFT JOIN steps s ON f.trace_id = s.trace_id AND f.onset_step_id = s.step_id
                GROUP BY f.failure_type, s.action_type, s.phase
                ORDER BY count DESC, f.failure_type, onset_action_type, onset_phase
                """
            )
        )


def query_action_mix(db_path: Path) -> list[dict[str, Any]]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return rows_to_dicts(
            connection.execute(
                """
                SELECT
                  t.source,
                  COALESCE(t.agent_name, '') AS agent_name,
                  100.0 * SUM(CASE WHEN s.action_type = 'READ' THEN 1 ELSE 0 END) / COUNT(*) AS read_percent,
                  100.0 * SUM(CASE WHEN s.action_type = 'SEARCH' THEN 1 ELSE 0 END) / COUNT(*) AS search_percent,
                  100.0 * SUM(CASE WHEN s.action_type = 'EDIT' THEN 1 ELSE 0 END) / COUNT(*) AS edit_percent,
                  100.0 * SUM(CASE WHEN s.action_type = 'VERIFY' THEN 1 ELSE 0 END) / COUNT(*) AS verify_percent,
                  100.0 * SUM(CASE WHEN s.is_error THEN 1 ELSE 0 END) / COUNT(*) AS error_percent
                FROM steps s
                JOIN traces t ON s.trace_id = t.trace_id
                GROUP BY t.source, t.agent_name
                ORDER BY t.source, agent_name
                """
            )
        )


def query_error_summary(db_path: Path) -> list[dict[str, Any]]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return rows_to_dicts(
            connection.execute(
                """
                SELECT
                  t.trace_id,
                  COALESCE(e.error_steps, 0)::INTEGER AS error_steps,
                  COALESCE(e.verify_errors, 0)::INTEGER AS verify_errors,
                  t.outcome_success AS final_success,
                  COALESCE(f.failure_count, 0)::INTEGER AS failure_count
                FROM traces t
                LEFT JOIN (
                  SELECT
                    trace_id,
                    COUNT(*) AS error_steps,
                    SUM(CASE WHEN action_type = 'VERIFY' THEN 1 ELSE 0 END) AS verify_errors
                  FROM steps
                  WHERE is_error = TRUE
                  GROUP BY trace_id
                ) e ON t.trace_id = e.trace_id
                LEFT JOIN (
                  SELECT trace_id, COUNT(*) AS failure_count
                  FROM failures
                  GROUP BY trace_id
                ) f ON t.trace_id = f.trace_id
                ORDER BY error_steps DESC, verify_errors DESC, t.trace_id
                """
            )
        )


def load_index_traces(path: Path) -> list[NormalizedTrace]:
    traces: list[NormalizedTrace] = []
    for file in iter_files(path, (".json",)):
        data = read_json(file)
        try:
            traces.append(validate_model_data(data, NormalizedTrace, file))
        except ValidationError:
            traces.append(normalize_trace(validate_model_data(data, RawTrace, file)))
    return traces


def load_optional_failures(path: Path | None) -> tuple[list[Any], list[str]]:
    if path is None:
        return [], []
    if not path.exists():
        return [], [f"missing failures path skipped: {path}"]
    return load_failure_hypotheses(path), []


def load_optional_evals(path: Path | None) -> tuple[list[Any], list[str]]:
    if path is None:
        return [], []
    if not path.exists():
        return [], [f"missing evals path skipped: {path}"]
    return load_eval_cases(path), []


def load_optional_runs(path: Path | None) -> tuple[list[Any], list[str]]:
    if path is None:
        return [], []
    if not path.exists():
        return [], [f"missing runs path skipped: {path}"]
    return load_run_results(path), []


def create_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TABLE metadata (
          key TEXT PRIMARY KEY,
          value TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE traces (
          trace_id TEXT PRIMARY KEY,
          source TEXT,
          task_id TEXT,
          task_description TEXT,
          task_prompt TEXT,
          agent_name TEXT,
          model_name TEXT,
          outcome_success BOOLEAN,
          tests_passed BOOLEAN,
          step_count INTEGER,
          metadata_json TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE steps (
          trace_id TEXT,
          step_id INTEGER,
          event_type TEXT,
          raw_action TEXT,
          action_type TEXT,
          phase TEXT,
          command TEXT,
          target TEXT,
          tool_name TEXT,
          is_error BOOLEAN,
          error_signature TEXT,
          touches_test_file BOOLEAN,
          touches_source_file BOOLEAN,
          modifies_file BOOLEAN,
          is_patch BOOLEAN,
          paths_json TEXT,
          metadata_json TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE failures (
          trace_id TEXT,
          failure_type TEXT,
          onset_step_id INTEGER,
          confidence DOUBLE,
          severity DOUBLE,
          detector TEXT,
          causal_role TEXT,
          evidence_json TEXT,
          metadata_json TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE evals (
          eval_id TEXT,
          source_trace_id TEXT,
          failure_type TEXT,
          verifier_rule TEXT,
          task_description TEXT,
          metadata_json TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE runs (
          eval_id TEXT,
          trace_id TEXT,
          passed BOOLEAN,
          rule TEXT,
          message TEXT,
          evidence_json TEXT,
          metadata_json TEXT
        )
        """
    )


def insert_metadata(connection: duckdb.DuckDBPyConnection) -> None:
    rows = [
        ("schema_version", DUCKDB_SCHEMA_VERSION),
        ("trace2eval_version", trace2eval_version()),
        ("created_at", datetime.now(UTC).isoformat()),
        ("source", "file_index"),
    ]
    connection.executemany("INSERT INTO metadata (key, value) VALUES (?, ?)", rows)


def insert_traces(connection: duckdb.DuckDBPyConnection, traces: list[NormalizedTrace]) -> None:
    rows = [
        (
            trace.trace_id,
            str(trace.source),
            trace.task.task_id,
            trace.task.description,
            trace.task.prompt,
            trace.agent.agent_name,
            trace.agent.model_name,
            trace.outcome.success,
            trace.outcome.tests_passed,
            len(trace.steps),
            json_text(trace.metadata),
        )
        for trace in traces
    ]
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO traces (
          trace_id, source, task_id, task_description, task_prompt, agent_name, model_name,
          outcome_success, tests_passed, step_count, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_steps(connection: duckdb.DuckDBPyConnection, traces: list[NormalizedTrace]) -> None:
    rows = []
    for trace in traces:
        for step in trace.steps:
            rows.append(
                (
                    trace.trace_id,
                    int_or_none(step.step_id),
                    step.raw_step.event_type,
                    step.raw_action,
                    step.action_type.value,
                    step.phase.value,
                    step.command,
                    step.target,
                    step.raw_step.tool_name,
                    step.is_error,
                    step.error_signature,
                    step.touches_test_file,
                    step.touches_source_file,
                    step.modifies_file,
                    step.is_patch,
                    json_text(step.metadata.get("paths") or []),
                    json_text(step.metadata),
                )
            )
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO steps (
          trace_id, step_id, event_type, raw_action, action_type, phase, command, target,
          tool_name, is_error, error_signature, touches_test_file, touches_source_file,
          modifies_file, is_patch, paths_json, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_failures(connection: duckdb.DuckDBPyConnection, failures: list[Any]) -> None:
    rows = [
        (
            failure.trace_id,
            failure.failure_type,
            int_or_none(failure.onset_step_id),
            failure.confidence,
            failure.severity,
            failure.detector,
            failure.metadata.get("causal_role"),
            json_text(failure.evidence),
            json_text(failure.metadata),
        )
        for failure in failures
    ]
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO failures (
          trace_id, failure_type, onset_step_id, confidence, severity,
          detector, causal_role, evidence_json, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_evals(connection: duckdb.DuckDBPyConnection, evals: list[Any]) -> None:
    rows = [
        (
            eval_case.eval_id,
            eval_case.source_trace_id,
            eval_case.failure_type,
            eval_case.verifier.rule,
            eval_case.task_description,
            json_text(eval_case.metadata),
        )
        for eval_case in evals
    ]
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO evals (
          eval_id, source_trace_id, failure_type, verifier_rule, task_description, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_runs(connection: duckdb.DuckDBPyConnection, runs: list[Any]) -> None:
    rows = [
        (
            result.eval_id,
            result.trace_id,
            result.passed,
            result.rule,
            result.message,
            json_text(result.evidence),
            json_text(result.metadata),
        )
        for result in runs
    ]
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO runs (
          eval_id, trace_id, passed, rule, message, evidence_json, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def rows_to_dicts(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def json_text(value: Any) -> str:
    return json_dumps(model_dump(value)).decode("utf-8")


def trace2eval_version() -> str:
    try:
        return version("trace2eval")
    except PackageNotFoundError:
        return "unknown"


def temporary_database_path(out_path: Path) -> Path:
    suffix = out_path.suffix or ".duckdb"
    return out_path.with_name(f".{out_path.stem}.tmp{suffix}")


def replace_database(temp_path: Path, out_path: Path) -> None:
    temp_path.replace(out_path)
    cleanup_wal_file(out_path)


def cleanup_database_files(path: Path) -> None:
    if path.exists():
        path.unlink()
    cleanup_wal_file(path)


def cleanup_wal_file(path: Path) -> None:
    wal_path = path.with_suffix(path.suffix + ".wal")
    if wal_path.exists():
        wal_path.unlink()

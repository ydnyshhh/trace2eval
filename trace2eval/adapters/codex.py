from __future__ import annotations

from pathlib import Path
from typing import Any

from trace2eval.adapters.common import (
    extract_command,
    extract_content,
    extract_diff,
    extract_event_type,
    extract_exit_code,
    extract_file_path,
    extract_observation,
    extract_role,
    extract_session_id,
    extract_status,
    extract_timestamp,
    extract_tool_args,
    extract_tool_name,
    jsonl_files,
    read_jsonl_events,
    trace_id_for_file,
)
from trace2eval.schemas import (
    AgentMetadata,
    OutcomeMetadata,
    RawStep,
    RawTrace,
    TaskMetadata,
    TraceSource,
)


class CodexJSONLAdapter:
    """Best-effort ingestion for Codex CLI rollout JSONL session logs."""

    source = TraceSource.CODEX

    def ingest(self, path: Path) -> list[RawTrace]:
        return [self._ingest_file(file) for file in jsonl_files(path)]

    def _ingest_file(self, path: Path) -> RawTrace:
        events = read_jsonl_events(path)
        session_id = first_found(events, extract_session_id)
        trace_id = trace_id_for_file(path, session_id)
        related_calls = build_tool_call_index(events)
        steps = [
            self._event_to_step(index, event, path, related_calls.get(str(event.get("tool_call_id") or event.get("call_id") or "")))
            for index, event in enumerate(events)
        ]
        prompt = first_user_prompt(steps)
        outcome = infer_outcome(events)
        agent = infer_agent(events)
        return RawTrace(
            trace_id=trace_id,
            source=self.source,
            task=TaskMetadata(task_id=session_id, prompt=prompt, description=prompt),
            agent=agent,
            outcome=outcome,
            steps=steps,
            metadata={"source_file": str(path), "session_id": session_id, "event_count": len(events)},
        )

    def _event_to_step(self, index: int, event: dict[str, Any], path: Path, related_call: dict[str, Any] | None = None) -> RawStep:
        related_call = related_call or {}
        event_type = extract_event_type(event)
        is_result = "result" in (event_type or "").lower()
        should_copy_action = not is_result or extract_exit_code(event) is not None or (extract_status(event) or "").lower() in {
            "failed",
            "failure",
            "error",
        }
        return RawStep(
            step_id=index,
            timestamp=extract_timestamp(event),
            event_type=event_type,
            role=extract_role(event),
            content=extract_content(event),
            tool_name=extract_tool_name(event) or (extract_tool_name(related_call) if should_copy_action else None),
            tool_args=extract_tool_args(event) or extract_tool_args(related_call),
            command=extract_command(event) or (extract_command(related_call) if should_copy_action else None),
            observation=extract_observation(event),
            file_path=extract_file_path(event) or (extract_file_path(related_call) if should_copy_action else None),
            diff=extract_diff(event) or (extract_diff(related_call) if not is_result else None),
            exit_code=extract_exit_code(event),
            status=extract_status(event),
            metadata={"raw_event": event, "related_tool_call": related_call or None, "source_file": str(path), "line_no": index + 1},
        )


def first_found(events: list[dict[str, Any]], extractor: Any) -> str | None:
    for event in events:
        value = extractor(event)
        if value:
            return value
    return None


def build_tool_call_index(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    for event in events:
        event_type = (extract_event_type(event) or "").lower()
        if "result" in event_type:
            continue
        call_id = event.get("id") or event.get("call_id") or event.get("tool_call_id")
        if call_id is not None and (extract_tool_name(event) or extract_command(event) or extract_diff(event)):
            calls[str(call_id)] = event
    return calls


def first_user_prompt(steps: list[RawStep]) -> str | None:
    for step in steps:
        if (step.role or "").lower() == "user" and step.content:
            return step.content
    for step in steps:
        if step.content and step.event_type and "prompt" in step.event_type.lower():
            return step.content
    return None


def infer_agent(events: list[dict[str, Any]]) -> AgentMetadata:
    model_name = None
    cli_version = None
    for event in events:
        model_name = model_name or nested_text(event, ("model",), ("model_name",), ("modelName",), ("config", "model"))
        cli_version = cli_version or nested_text(
            event,
            ("version",),
            ("cli_version",),
            ("codex_version",),
            ("config", "version"),
        )
    return AgentMetadata(agent_name="codex", model_name=model_name, cli_version=cli_version)


def infer_outcome(events: list[dict[str, Any]]) -> OutcomeMetadata:
    success: bool | None = None
    failure_summary: str | None = None
    exit_status: str | int | None = None
    for event in reversed(events):
        status = nested_text(event, ("status",), ("outcome",), ("state",), ("result", "status"))
        if status and success is None:
            low = status.lower()
            if low in {"success", "succeeded", "passed", "complete", "completed"}:
                success = True
            elif low in {"failure", "failed", "error", "cancelled", "canceled"}:
                success = False
            exit_status = status
        error = nested_text(event, ("error",), ("failure",), ("result", "error"))
        if error and not failure_summary:
            failure_summary = error
    return OutcomeMetadata(success=success, failure_summary=failure_summary, exit_status=exit_status)


def nested_text(event: dict[str, Any], *paths: tuple[str, ...]) -> str | None:
    from trace2eval.adapters.common import first_text

    return first_text(event, list(paths))

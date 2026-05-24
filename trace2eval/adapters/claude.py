from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from trace2eval.adapters.common import (
    as_text,
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
    json_files,
    read_jsonl_events,
    trace_id_for_file,
)
from trace2eval.io import read_json
from trace2eval.schemas import (
    AgentMetadata,
    OutcomeMetadata,
    RawStep,
    RawTrace,
    TaskMetadata,
    TraceSource,
)


class ClaudeCodeHookJSONLAdapter:
    """Ingest JSONL records captured by trace2eval's Claude Code hook logger."""

    source = TraceSource.CLAUDE_CODE_HOOKS

    def ingest(self, path: Path) -> list[RawTrace]:
        traces: list[RawTrace] = []
        for file in [path] if path.is_file() else sorted(path.rglob("*.jsonl")):
            events = read_jsonl_events(file)
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for event in events:
                key = extract_session_id(event) or f"file:{file}"
                grouped[key].append(event)
            for session_id, group in grouped.items():
                traces.append(self.events_to_trace(file, session_id if not session_id.startswith("file:") else None, group))
        return traces

    def events_to_trace(self, file: Path, session_id: str | None, events: list[dict[str, Any]]) -> RawTrace:
        steps = [self.event_to_step(index, event, file) for index, event in enumerate(events)]
        prompt = first_content_by_event(steps, "UserPromptSubmit") or first_user_prompt(steps)
        repo_path = first_metadata(events, "cwd") or first_metadata(events, "project_dir")
        return RawTrace(
            trace_id=trace_id_for_file(file, session_id),
            source=self.source,
            task=TaskMetadata(task_id=session_id, prompt=prompt, description=prompt, repo_path=repo_path),
            agent=AgentMetadata(agent_name="claude_code"),
            outcome=infer_hook_outcome(steps),
            steps=steps,
            metadata={"source_file": str(file), "session_id": session_id, "event_count": len(events)},
        )

    def event_to_step(self, index: int, event: dict[str, Any], file: Path) -> RawStep:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else event
        return RawStep(
            step_id=index,
            timestamp=extract_timestamp(event),
            event_type=extract_event_type(event),
            role=extract_role(payload) or extract_role(event),
            content=extract_content(payload) or extract_content(event),
            tool_name=extract_tool_name(payload) or extract_tool_name(event),
            tool_args=extract_tool_args(payload) or extract_tool_args(event),
            command=extract_command(payload) or extract_command(event),
            observation=extract_observation(payload) or extract_observation(event),
            file_path=extract_file_path(payload) or extract_file_path(event),
            diff=extract_diff(payload) or extract_diff(event),
            exit_code=extract_exit_code(payload) if extract_exit_code(payload) is not None else extract_exit_code(event),
            status=extract_status(payload) or extract_status(event),
            metadata={"raw_hook_event": event, "raw_payload": payload, "source_file": str(file), "line_no": index + 1},
        )


class ClaudeCodeHeadlessJSONAdapter:
    """Ingest JSON output from Claude Code headless/programmatic runs."""

    source = TraceSource.CLAUDE_CODE_HEADLESS

    def ingest(self, path: Path) -> list[RawTrace]:
        return [self.ingest_file(file) for file in json_files(path)]

    def ingest_file(self, file: Path) -> RawTrace:
        data = read_json(file)
        root = data if isinstance(data, dict) else {"result": data}
        session_id = extract_session_id(root)
        messages = extract_message_events(root)
        if not messages:
            messages = [root]
        steps = [self.event_to_step(index, event, file) for index, event in enumerate(messages)]
        prompt = first_user_prompt(steps)
        return RawTrace(
            trace_id=trace_id_for_file(file, session_id),
            source=self.source,
            task=TaskMetadata(task_id=session_id, prompt=prompt, description=prompt),
            agent=AgentMetadata(
                agent_name="claude_code",
                model_name=as_text(root.get("model") or root.get("model_name")),
                cli_version=as_text(root.get("version") or root.get("cli_version")),
            ),
            outcome=OutcomeMetadata(
                success=infer_success(root),
                failure_summary=as_text(root.get("error") or root.get("failure")),
                exit_status=as_text(root.get("status")),
                metadata={"usage": root.get("usage"), "structured_output": root.get("structured_output")},
            ),
            steps=steps,
            metadata={"source_file": str(file), "session_id": session_id, "raw_headless_output": root},
        )

    def event_to_step(self, index: int, event: dict[str, Any], file: Path) -> RawStep:
        return RawStep(
            step_id=index,
            timestamp=extract_timestamp(event),
            event_type=extract_event_type(event),
            role=extract_role(event),
            content=extract_content(event),
            tool_name=extract_tool_name(event),
            tool_args=extract_tool_args(event),
            command=extract_command(event),
            observation=extract_observation(event),
            file_path=extract_file_path(event),
            diff=extract_diff(event),
            exit_code=extract_exit_code(event),
            status=extract_status(event),
            metadata={"raw_event": event, "source_file": str(file), "index": index},
        )


def extract_message_events(root: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("messages", "events", "steps", "transcript", "tool_traces", "tool_calls"):
        value = root.get(key)
        if isinstance(value, list):
            return [item if isinstance(item, dict) else {"value": item} for item in value]
    result = root.get("result")
    if isinstance(result, dict):
        return extract_message_events(result)
    return []


def first_content_by_event(steps: list[RawStep], event_name: str) -> str | None:
    for step in steps:
        if (step.event_type or "").lower() == event_name.lower() and step.content:
            return step.content
    return None


def first_user_prompt(steps: list[RawStep]) -> str | None:
    for step in steps:
        if (step.role or "").lower() == "user" and step.content:
            return step.content
    for step in steps:
        if step.content:
            return step.content
    return None


def first_metadata(events: list[dict[str, Any]], key: str) -> str | None:
    for event in events:
        value = as_text(event.get(key))
        if value:
            return value
    return None


def infer_success(root: dict[str, Any]) -> bool | None:
    for key in ("success", "is_success", "passed"):
        value = root.get(key)
        if isinstance(value, bool):
            return value
    status = as_text(root.get("status") or root.get("outcome"))
    if not status:
        return None
    low = status.lower()
    if low in {"success", "succeeded", "passed", "complete", "completed"}:
        return True
    if low in {"failure", "failed", "error", "cancelled", "canceled"}:
        return False
    return None


def infer_hook_outcome(steps: list[RawStep]) -> OutcomeMetadata:
    success: bool | None = None
    status = None
    for step in reversed(steps):
        status = status or step.status or step.event_type
        low = (step.status or "").lower()
        if low in {"success", "succeeded", "passed"}:
            success = True
            break
        if low in {"failure", "failed", "error"}:
            success = False
            break
    return OutcomeMetadata(success=success, exit_status=status)

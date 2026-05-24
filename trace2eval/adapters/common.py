from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from trace2eval.io import iter_files, iter_jsonl, slugify

JSON = dict[str, Any]


def as_text(value: Any, *, max_chars: int | None = None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, bytes):
        text = value.decode(errors="replace")
    elif isinstance(value, list):
        parts = [as_text(item) for item in value]
        text = "\n".join(part for part in parts if part)
    elif isinstance(value, dict):
        if "text" in value:
            text = as_text(value.get("text")) or ""
        elif "content" in value:
            text = as_text(value.get("content")) or ""
        elif "message" in value and isinstance(value["message"], str):
            text = value["message"]
        else:
            parts = []
            for key in ("stdout", "stderr", "output", "error", "result", "summary"):
                part = as_text(value.get(key))
                if part:
                    parts.append(part)
            text = "\n".join(parts)
    else:
        text = str(value)
    text = text.strip()
    if not text:
        return None
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def get_path(data: Any, path: tuple[str, ...]) -> Any:
    current = data
    for part in path:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def first_value(data: JSON, paths: list[tuple[str, ...]]) -> Any:
    for path in paths:
        value = get_path(data, path)
        if value is not None:
            return value
    return None


def first_text(data: JSON, paths: list[tuple[str, ...]]) -> str | None:
    return as_text(first_value(data, paths))


def stable_hash(data: Any) -> str:
    text = as_text(data) or repr(data)
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = list(iter_files(path, (".jsonl",)))
    rollout = [file for file in candidates if file.name.startswith("rollout-")]
    return rollout or candidates


def json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return list(iter_files(path, (".json",)))


def read_jsonl_events(path: Path) -> list[JSON]:
    events: list[JSON] = []
    for item in iter_jsonl(path):
        if isinstance(item, dict):
            events.append(item)
        else:
            events.append({"value": item})
    return events


def extract_event_type(event: JSON) -> str | None:
    return first_text(
        event,
        [
            ("event_name",),
            ("hook_event_name",),
            ("type",),
            ("event",),
            ("kind",),
            ("name",),
            ("payload", "event_name"),
            ("payload", "hook_event_name"),
            ("payload", "type"),
            ("payload", "event"),
            ("message", "type"),
        ],
    )


def extract_timestamp(event: JSON) -> str | None:
    return first_text(
        event,
        [
            ("timestamp",),
            ("captured_at",),
            ("created_at",),
            ("time",),
            ("ts",),
            ("payload", "timestamp"),
            ("payload", "created_at"),
        ],
    )


def extract_session_id(event: JSON) -> str | None:
    return first_text(
        event,
        [
            ("session_id",),
            ("sessionId",),
            ("conversation_id",),
            ("conversationId",),
            ("run_id",),
            ("trace_id",),
            ("id",),
            ("payload", "session_id"),
            ("payload", "sessionId"),
            ("payload", "conversation_id"),
            ("payload", "run_id"),
        ],
    )


def extract_role(event: JSON) -> str | None:
    return first_text(
        event,
        [
            ("role",),
            ("message", "role"),
            ("delta", "role"),
            ("payload", "role"),
            ("payload", "message", "role"),
        ],
    )


def extract_content(event: JSON) -> str | None:
    message = event.get("message")
    if isinstance(message, dict):
        content = as_text(message.get("content"))
        if content:
            return content
    return first_text(
        event,
        [
            ("content",),
            ("text",),
            ("delta", "content"),
            ("payload", "content"),
            ("payload", "text"),
            ("payload", "message", "content"),
            ("payload", "transcript"),
        ],
    )


def extract_tool_name(event: JSON) -> str | None:
    value = first_text(
        event,
        [
            ("tool_name",),
            ("toolName",),
            ("tool", "name"),
            ("tool_call", "name"),
            ("toolCall", "name"),
            ("tool_use", "name"),
            ("toolUse", "name"),
            ("invocation", "name"),
            ("payload", "tool_name"),
            ("payload", "toolName"),
            ("payload", "tool", "name"),
            ("payload", "tool_call", "name"),
            ("payload", "toolUse", "name"),
        ],
    )
    if value:
        return value
    if has_any(event, ("tool_input", "tool_response", "tool_result", "toolUse", "tool_call")):
        name = event.get("name")
        if isinstance(name, str):
            return name
    payload = event.get("payload")
    if isinstance(payload, dict) and has_any(payload, ("tool_input", "tool_response", "tool_result")):
        name = payload.get("name")
        if isinstance(name, str):
            return name
    return None


def extract_tool_args(event: JSON) -> Any:
    return first_value(
        event,
        [
            ("tool_args",),
            ("tool_input",),
            ("toolInput",),
            ("input",),
            ("arguments",),
            ("args",),
            ("params",),
            ("parameters",),
            ("tool_call", "arguments"),
            ("toolCall", "arguments"),
            ("tool_use", "input"),
            ("toolUse", "input"),
            ("payload", "tool_args"),
            ("payload", "tool_input"),
            ("payload", "toolInput"),
            ("payload", "input"),
            ("payload", "arguments"),
            ("payload", "toolUse", "input"),
        ],
    )


def extract_command(event: JSON) -> str | None:
    command = first_text(
        event,
        [
            ("command",),
            ("cmd",),
            ("shell_command",),
            ("shellCommand",),
            ("input", "command"),
            ("tool_input", "command"),
            ("toolInput", "command"),
            ("args", "command"),
            ("arguments", "command"),
            ("params", "command"),
            ("payload", "command"),
            ("payload", "cmd"),
            ("payload", "input", "command"),
            ("payload", "tool_input", "command"),
            ("payload", "toolInput", "command"),
        ],
    )
    if command:
        return command
    tool_name = (extract_tool_name(event) or "").lower()
    tool_args = extract_tool_args(event)
    if tool_name in {"bash", "shell", "exec", "run_command", "terminal", "powershell"}:
        if isinstance(tool_args, str):
            return tool_args
        if isinstance(tool_args, dict):
            return as_text(tool_args.get("command") or tool_args.get("cmd") or tool_args.get("script"))
    return None


def extract_observation(event: JSON) -> str | None:
    stdout = first_text(event, [("stdout",), ("payload", "stdout"), ("result", "stdout")])
    stderr = first_text(event, [("stderr",), ("payload", "stderr"), ("result", "stderr")])
    if stdout or stderr:
        return "\n".join(part for part in (stdout, stderr) if part)
    return first_text(
        event,
        [
            ("observation",),
            ("output",),
            ("result",),
            ("response",),
            ("tool_output",),
            ("toolOutput",),
            ("tool_response",),
            ("tool_result",),
            ("error",),
            ("payload", "observation"),
            ("payload", "output"),
            ("payload", "result"),
            ("payload", "response"),
            ("payload", "tool_output"),
            ("payload", "tool_response"),
            ("payload", "error"),
        ],
    )


def extract_diff(event: JSON) -> str | None:
    return first_text(
        event,
        [
            ("diff",),
            ("patch",),
            ("edits",),
            ("payload", "diff"),
            ("payload", "patch"),
            ("payload", "edits"),
            ("tool_input", "patch"),
            ("tool_input", "diff"),
            ("input", "patch"),
            ("input", "diff"),
        ],
    )


def extract_exit_code(event: JSON) -> int | None:
    value = first_value(
        event,
        [
            ("exit_code",),
            ("exitCode",),
            ("returncode",),
            ("return_code",),
            ("status_code",),
            ("payload", "exit_code"),
            ("payload", "exitCode"),
            ("payload", "returncode"),
        ],
    )
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value)
    return None


def extract_status(event: JSON) -> str | None:
    return first_text(
        event,
        [
            ("status",),
            ("state",),
            ("outcome",),
            ("payload", "status"),
            ("payload", "state"),
            ("payload", "outcome"),
        ],
    )


def extract_file_path(event: JSON) -> str | None:
    value = first_value(
        event,
        [
            ("file_path",),
            ("filepath",),
            ("path",),
            ("target",),
            ("target_file"),
            ("file",),
            ("file_paths",),
            ("files",),
            ("input", "file_path"),
            ("input", "path"),
            ("tool_input", "file_path"),
            ("tool_input", "path"),
            ("toolInput", "file_path"),
            ("toolInput", "path"),
            ("payload", "file_path"),
            ("payload", "path"),
            ("payload", "target"),
            ("payload", "file_paths"),
            ("payload", "tool_input", "file_path"),
            ("payload", "tool_input", "path"),
        ],
    )
    if isinstance(value, list) and value:
        return as_text(value[0])
    text = as_text(value)
    if text:
        return text
    command = extract_command(event)
    if command:
        paths = extract_paths_from_text(command)
        if paths:
            return paths[0]
    diff = extract_diff(event)
    if diff:
        paths = extract_paths_from_text(diff)
        if paths:
            return paths[0]
    return None


def extract_paths_from_text(text: str | None) -> list[str]:
    if not text:
        return []
    candidates: list[str] = []
    patterns = [
        r"(?:^|\s)([A-Za-z0-9_./\\-]+(?:test|spec|src|lib|app|packages|crates|tests)[A-Za-z0-9_./\\-]*\.[A-Za-z0-9]+)",
        r"(?:^|\s)([A-Za-z0-9_./\\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cpp|c|h|hpp|rb|php|md|toml|yaml|yml|json))",
        r"^[+-]{3}\s+[ab]/([^\s]+)",
        r"^@@\s+.*?\s+([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            candidate = match.group(1).strip(" '\"`:,;")
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def has_any(data: JSON, keys: tuple[str, ...]) -> bool:
    return any(key in data for key in keys)


def trace_id_for_file(file: Path, session_id: str | None = None) -> str:
    stem = slugify(file.stem)
    if session_id and session_id not in stem:
        return f"{stem}-{slugify(session_id, 32)}"
    return stem or stable_hash(str(file))

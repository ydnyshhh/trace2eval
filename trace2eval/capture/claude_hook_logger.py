from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1.0"


def main() -> int:
    try:
        payload = read_payload()
        cwd = Path.cwd()
        log_path = Path(os.environ.get("TRACE2EVAL_LOG_PATH", cwd / ".trace2eval" / "claude-code" / "events.jsonl"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = build_record(payload, cwd)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
            f.write("\n")
    except Exception:
        write_error()
    return 0


def read_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"_trace2eval_json_error": str(exc), "raw_stdin": raw}
    if isinstance(data, dict):
        return data
    return {"value": data}


def build_record(payload: dict[str, Any], cwd: Path) -> dict[str, Any]:
    event_name = first_text(
        os.environ.get("CLAUDE_CODE_HOOK_EVENT_NAME"),
        os.environ.get("CLAUDE_HOOK_EVENT_NAME"),
        os.environ.get("HOOK_EVENT_NAME"),
        payload.get("hook_event_name"),
        payload.get("event_name"),
        payload.get("event"),
        payload.get("type"),
    )
    tool_input = first_value(payload, "tool_input", "toolInput", "input", "parameters")
    tool_output = first_value(payload, "tool_output", "toolOutput", "output", "response", "result")
    command = extract_command(payload, tool_input)
    file_paths = extract_file_paths(payload, tool_input)
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
        "event_name": event_name,
        "session_id": first_text(payload.get("session_id"), payload.get("sessionId"), payload.get("conversation_id")),
        "cwd": str(cwd),
        "project_dir": os.environ.get("CLAUDE_PROJECT_DIR") or os.environ.get("PWD") or str(cwd),
        "git_commit": git_value(cwd, "rev-parse", "HEAD"),
        "git_branch": git_value(cwd, "rev-parse", "--abbrev-ref", "HEAD"),
        "tool_name": first_text(payload.get("tool_name"), payload.get("toolName"), nested(payload, "tool", "name"), payload.get("name")),
        "tool_input": tool_input,
        "tool_output": tool_output,
        "command": command,
        "file_paths": file_paths,
        "status": first_text(payload.get("status"), payload.get("state"), payload.get("outcome")),
        "payload": payload,
    }


def first_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text
    return None


def nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_command(payload: dict[str, Any], tool_input: Any) -> str | None:
    command = first_text(payload.get("command"), payload.get("cmd"), payload.get("shell_command"))
    if command:
        return command
    if isinstance(tool_input, dict):
        return first_text(tool_input.get("command"), tool_input.get("cmd"), tool_input.get("script"))
    if isinstance(tool_input, str):
        tool_name = first_text(payload.get("tool_name"), payload.get("name")) or ""
        if tool_name.lower() in {"bash", "shell", "exec", "terminal", "run_command"}:
            return tool_input
    return None


def extract_file_paths(payload: dict[str, Any], tool_input: Any) -> list[str]:
    paths: list[str] = []
    for source in (payload, tool_input if isinstance(tool_input, dict) else {}):
        for key in ("file_path", "filepath", "path", "target", "target_file", "file_paths", "files"):
            value = source.get(key)
            if isinstance(value, str):
                paths.append(value)
            elif isinstance(value, list):
                paths.extend(str(item) for item in value)
    return list(dict.fromkeys(paths))


def git_value(cwd: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def write_error() -> None:
    try:
        cwd = Path.cwd()
        log_path = Path(os.environ.get("TRACE2EVAL_LOG_PATH", cwd / ".trace2eval" / "claude-code" / "events.jsonl"))
        error_path = log_path.parent / "hook_errors.log"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("a", encoding="utf-8") as f:
            f.write(datetime.now(UTC).isoformat())
            f.write("\n")
            f.write(traceback.format_exc())
            f.write("\n")
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())

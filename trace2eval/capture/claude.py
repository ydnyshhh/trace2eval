from __future__ import annotations

import inspect
from pathlib import Path

from trace2eval.capture import claude_hook_logger
from trace2eval.io import ensure_dir

CLAUDE_HOOK_EVENTS = [
    "PreToolUse",
    "PostToolUse",
    "UserPromptSubmit",
    "Notification",
    "Stop",
    "SubagentStop",
    "PreCompact",
    "SessionStart",
    "SessionEnd",
]


def install_claude_hook(out_dir: Path) -> Path:
    ensure_dir(out_dir)
    script_path = out_dir / "claude_hook_logger.py"
    source = inspect.getsource(claude_hook_logger)
    script_path.write_text(source, encoding="utf-8")
    return script_path


def build_claude_settings_snippet(script_path: Path, log_path: Path | None = None) -> str:
    command = f"python {script_path.as_posix()}"
    if log_path is not None:
        command = f'TRACE2EVAL_LOG_PATH="{log_path.as_posix()}" {command}'
    event_entries = []
    for event_name in CLAUDE_HOOK_EVENTS:
        event_entries.append(
            f'    "{event_name}": [\n'
            "      {\n"
            '        "matcher": "*",\n'
            "        \"hooks\": [\n"
            f'          {{ "type": "command", "command": "{command}" }}\n'
            "        ]\n"
            "      }\n"
            "    ]"
        )
    return "{\n  \"hooks\": {\n" + ",\n".join(event_entries) + "\n  }\n}"

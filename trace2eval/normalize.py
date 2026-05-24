from __future__ import annotations

import re

from trace2eval.adapters.common import as_text, extract_paths_from_text
from trace2eval.schemas import ActionType, NormalizedStep, NormalizedTrace, Phase, RawStep, RawTrace

READ_TOOLS = {
    "open_file",
    "read_file",
    "view_file",
    "cat",
    "read",
    "viewer",
    "file_read",
}
SEARCH_TOOLS = {"grep", "rg", "ripgrep", "search_repo", "find", "ls", "tree", "glob", "git_grep"}
EDIT_TOOLS = {
    "edit_file",
    "apply_patch",
    "write_file",
    "replace",
    "create_file",
    "patch",
    "str_replace",
    "notebook_edit",
}
EXEC_TOOLS = {"bash", "shell", "exec", "terminal", "run_command", "powershell", "python"}

SOURCE_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".rb",
    ".php",
}

ERROR_RE = re.compile(
    r"(traceback|error:|exception|failed\b|failed:|failure\b|FAILED|exit code|command not found|"
    r"modulenotfounderror|syntaxerror|typeerror|assertionerror|non-zero|stack trace|panic|"
    r"compilation failed|test failed|cannot find module|no such file or directory|permission denied|"
    r"returned non-zero|return code [1-9]|exit status [1-9])",
    re.IGNORECASE,
)


VERIFY_COMMAND_RE = re.compile(
    r"(^|\s)(pytest|python\s+-m\s+pytest|npm\s+test|pnpm\s+test|yarn\s+test|bun\s+test|"
    r"vitest|jest|go\s+test|cargo\s+test|cargo\s+check|mvn\s+test|gradle\s+test|"
    r"./gradlew\s+test|ruff\s+check|mypy|tsc\b|eslint|unittest|tox|make\s+test)\b",
    re.IGNORECASE,
)
SEARCH_COMMAND_RE = re.compile(r"^\s*(rg|grep|find|ls|tree)\b|^\s*git\s+grep\b", re.IGNORECASE)
READ_COMMAND_RE = re.compile(r"^\s*(cat|less|head|tail)\b|^\s*sed\s+-n\b|^\s*Get-Content\b", re.IGNORECASE)
EDIT_COMMAND_RE = re.compile(
    r"(apply_patch|cat\s+>|>\s*[\w./\\-]+\.(py|ts|js|go|rs|java|cpp|c|h|rb|php)|"
    r"tee\s+[\w./\\-]+|sed\s+-i|perl\s+-pi|write_text\(|open\(.+['\"]w['\"]|"
    r"Set-Content|Add-Content|Out-File|New-Item)",
    re.IGNORECASE | re.DOTALL,
)


def normalize_trace(raw_trace: RawTrace) -> NormalizedTrace:
    steps = [map_step(step) for step in raw_trace.steps]
    segment_phases(steps)
    return NormalizedTrace(
        trace_id=raw_trace.trace_id,
        source=raw_trace.source,
        task=raw_trace.task,
        agent=raw_trace.agent,
        outcome=raw_trace.outcome,
        steps=steps,
        metadata={"raw_trace_metadata": raw_trace.metadata},
    )


def map_step(step: RawStep) -> NormalizedStep:
    paths = paths_for_step(step)
    target = step.file_path or (paths[0] if paths else None)
    command = step.command
    observation = step.observation or step.content
    action_type = classify_action(step, paths)
    is_patch = bool(step.diff) or looks_like_patch(step.content) or looks_like_patch(step.observation)
    modifies_file = action_type == ActionType.EDIT or is_patch
    touches_test = any(is_test_path(path) for path in paths + ([target] if target else []))
    touches_source = any(is_source_path(path) for path in paths + ([target] if target else []))
    is_error, error_signature = detect_error(step)
    return NormalizedStep(
        step_id=step.step_id,
        raw_step=step,
        action_type=action_type,
        raw_action=step.command or step.tool_name or step.event_type,
        target=target,
        command=command,
        observation=observation,
        is_error=is_error,
        error_signature=error_signature,
        touches_test_file=touches_test,
        touches_source_file=touches_source,
        modifies_file=modifies_file,
        is_patch=is_patch,
        is_final=action_type == ActionType.STOP,
        metadata={"paths": paths},
    )


def classify_action(step: RawStep, paths: list[str]) -> ActionType:
    tool = (step.tool_name or "").strip().lower()
    event = (step.event_type or "").strip().lower()
    command = step.command or ""
    text = "\n".join(part for part in (step.content, step.observation, step.diff) if part)

    if event in {"userpromptsubmit", "user_prompt_submit"} or (step.role or "").lower() == "user":
        return ActionType.PLAN
    if any(word in event for word in ("stop", "final", "done", "complete", "exit")) or (
        "submit" in event and "userprompt" not in event
    ):
        return ActionType.STOP
    if "ask" in event or looks_like_clarifying_question(step.content):
        return ActionType.ASK_USER
    if step.diff or looks_like_patch(text):
        return ActionType.EDIT
    if command:
        return classify_command(command)
    if tool in EDIT_TOOLS or any(word in event for word in ("edit", "patch", "write", "replace", "create")):
        return ActionType.EDIT
    if tool in READ_TOOLS or any(word in event for word in ("read", "open", "view")):
        return ActionType.READ
    if tool in SEARCH_TOOLS or any(word in event for word in ("search", "grep", "find", "list")):
        return ActionType.SEARCH
    if tool in EXEC_TOOLS:
        return ActionType.EXECUTE
    if tool:
        if any(is_test_path(path) for path in paths) and any(word in event for word in ("result", "output")):
            return ActionType.TOOL_RESULT
        return ActionType.TOOL_CALL
    if any(word in event for word in ("tool_result", "result", "observation", "output")):
        return ActionType.TOOL_RESULT
    if (step.role or "").lower() == "assistant" and step.content:
        return ActionType.PLAN
    if (step.role or "").lower() == "user":
        return ActionType.PLAN
    return ActionType.UNKNOWN


def classify_command(command: str) -> ActionType:
    stripped = command.strip()
    if VERIFY_COMMAND_RE.search(stripped):
        return ActionType.VERIFY
    if EDIT_COMMAND_RE.search(stripped):
        return ActionType.EDIT
    if SEARCH_COMMAND_RE.search(stripped):
        return ActionType.SEARCH
    if READ_COMMAND_RE.search(stripped):
        return ActionType.READ
    lowered = stripped.lower()
    if "python" in lowered and any(token in lowered for token in ("write_text", "open(", "replace(")):
        return ActionType.EDIT
    return ActionType.EXECUTE


def paths_for_step(step: RawStep) -> list[str]:
    paths: list[str] = []
    for value in (step.file_path, step.command, step.diff, step.content, step.observation, as_text(step.tool_args)):
        for path in extract_paths_from_text(value):
            if path not in paths:
                paths.append(path)
    if step.file_path and step.file_path not in paths:
        paths.insert(0, step.file_path)
    return paths


def is_test_path(path: str | None) -> bool:
    if not path:
        return False
    normalized = path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    return bool(
        re.search(r"(^|/)(tests?|specs?)(/|$)", normalized)
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith("_spec.py")
        or re.search(r"\.(test|spec)\.(js|jsx|ts|tsx)$", name)
        or re.search(r"(_test|_spec)\.(go|rs|rb|php|java|cpp|c)$", name)
    )


def is_source_path(path: str | None) -> bool:
    if not path or is_test_path(path):
        return False
    normalized = path.replace("\\", "/").lower()
    ext = "." + normalized.rsplit(".", 1)[-1] if "." in normalized.rsplit("/", 1)[-1] else ""
    if ext in SOURCE_EXTENSIONS:
        return True
    return bool(re.search(r"(^|/)(src|lib|app|packages|crates)(/|$)", normalized))


def detect_error(step: RawStep) -> tuple[bool, str | None]:
    if step.exit_code is not None and step.exit_code != 0:
        signature = first_error_line(step.observation or step.content) or f"exit_code={step.exit_code}"
        return True, signature
    status = (step.status or "").lower()
    if status in {"failed", "failure", "error", "errored", "nonzero", "non-zero"}:
        return True, first_error_line(step.observation or step.content) or step.status
    text = "\n".join(part for part in (step.observation, step.content) if part)
    if ERROR_RE.search(text):
        return True, first_error_line(text)
    return False, None


def first_error_line(text: str | None) -> str | None:
    if not text:
        return None
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned and ERROR_RE.search(cleaned):
            return cleaned[:300]
    return None


def looks_like_patch(text: str | None) -> bool:
    if not text:
        return False
    return "*** Begin Patch" in text or bool(re.search(r"^diff --git |^\+\+\+ b/|^--- a/", text, re.MULTILINE))


def looks_like_clarifying_question(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.strip().lower()
    return lowered.endswith("?") and any(word in lowered for word in ("which", "what", "clarify", "should i", "do you want"))


def segment_phases(steps: list[NormalizedStep]) -> None:
    saw_edit = False
    after_error = False
    for index, step in enumerate(steps):
        if step.action_type == ActionType.STOP:
            step.phase = Phase.SUBMISSION
        elif not saw_edit:
            if step.action_type == ActionType.PLAN and index == 0:
                step.phase = Phase.UNDERSTANDING
            elif step.action_type == ActionType.SEARCH:
                step.phase = Phase.EXPLORATION
            elif step.action_type in {ActionType.READ, ActionType.VERIFY}:
                step.phase = Phase.LOCALIZATION
            elif step.action_type == ActionType.EDIT:
                step.phase = Phase.EDITING
                saw_edit = True
            else:
                step.phase = Phase.EXPLORATION if step.action_type != ActionType.UNKNOWN else Phase.UNKNOWN
        elif after_error and step.action_type in {ActionType.SEARCH, ActionType.READ, ActionType.EXECUTE, ActionType.PLAN}:
            step.phase = Phase.RECOVERY
        elif step.action_type == ActionType.EDIT:
            step.phase = Phase.EDITING
        elif step.action_type == ActionType.VERIFY:
            step.phase = Phase.VERIFICATION
        elif step.action_type in {ActionType.SEARCH, ActionType.READ, ActionType.EXECUTE}:
            step.phase = Phase.RECOVERY if after_error else Phase.EDITING
        else:
            step.phase = Phase.UNKNOWN

        if step.action_type == ActionType.EDIT:
            saw_edit = True
            after_error = False
        elif step.is_error:
            after_error = True
        elif step.action_type in {ActionType.SEARCH, ActionType.READ, ActionType.EXECUTE, ActionType.VERIFY}:
            after_error = False

from __future__ import annotations

import re


def extract_paths_from_text(text: str | None) -> list[str]:
    if not text:
        return []
    candidates: list[str] = []
    diff_patterns = [
        r"^[+-]{3}\s+[ab]/([^\s]+)",
        r"^\*\*\* (?:Update|Add|Delete) File:\s+([^\s]+)",
    ]
    path_pattern = r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_./\\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cpp|c|h|hpp|rb|php|md|toml|yaml|yml|json|jsonl|log|txt))(?![A-Za-z0-9_.-])"
    for line in text.splitlines():
        for pattern in diff_patterns:
            for match in re.finditer(pattern, line, flags=re.IGNORECASE):
                add_candidate(candidates, match.group(1))
        for match in re.finditer(path_pattern, line, flags=re.IGNORECASE):
            candidate = match.group(1).strip(" '\"`:,;()[]{}")
            if is_plausible_path_candidate(candidate, line):
                add_candidate(candidates, candidate)
    return candidates


def add_candidate(candidates: list[str], candidate: str) -> None:
    cleaned = candidate.strip(" '\"`:,;()[]{}")
    if cleaned and cleaned not in candidates and cleaned not in {"a/dev/null", "b/dev/null", "/dev/null"}:
        candidates.append(cleaned)


def is_plausible_path_candidate(candidate: str, line: str) -> bool:
    normalized = candidate.replace("\\", "/")
    lower = normalized.lower()
    if "/" in normalized:
        return True
    if lower in {
        "pyproject.toml",
        "package.json",
        "tsconfig.json",
        "pytest.ini",
        "setup.py",
        "setup.cfg",
        "readme.md",
    }:
        return True
    if candidate.count(".") > 1:
        return False
    line_lower = line.lower()
    if re.search(r"\b(cat|sed|pytest|ruff|mypy|python|get-content|new-item|set-content|touch)\b", line_lower):
        return True
    return bool(re.search(r"\b(file|file_path|filepath|target|path|command)\s*[:=]", line_lower))

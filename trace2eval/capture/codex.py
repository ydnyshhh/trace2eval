from __future__ import annotations

import os
from pathlib import Path


def codex_home_candidates(explicit: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit.expanduser())
    env_home = os.environ.get("CODEX_HOME")
    if env_home:
        candidates.append(Path(env_home).expanduser())
    candidates.append(Path.home() / ".codex")
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def discover_codex_rollouts(codex_home: Path | None = None) -> list[Path]:
    rollouts: list[Path] = []
    for home in codex_home_candidates(codex_home):
        sessions = home / "sessions"
        if not sessions.exists():
            continue
        rollouts.extend(sorted(sessions.rglob("rollout-*.jsonl")))
    seen: set[Path] = set()
    deduped: list[Path] = []
    for rollout in rollouts:
        resolved = rollout.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(rollout)
    return deduped

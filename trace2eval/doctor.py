from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Literal

from trace2eval.adapters import GenericJSONAdapter
from trace2eval.capture import discover_codex_rollouts
from trace2eval.generation import generate_eval_cases
from trace2eval.mining import mine_trace
from trace2eval.normalize import normalize_trace
from trace2eval.runner import run_evals

DoctorStatus = Literal["ok", "warn", "fail"]


def run_doctor_checks(
    *,
    workspace: Path = Path(".trace2eval"),
    examples: Path = Path("examples/traces"),
    benchmark_fixtures: Path = Path("examples/real_runs"),
    codex_home: Path | None = None,
) -> list[dict[str, str]]:
    return [
        dependency_check(),
        workspace_check(workspace),
        codex_sessions_check(codex_home),
        claude_hook_check(workspace),
        examples_validate_check(examples),
        benchmark_cases_check(benchmark_fixtures),
    ]


def dependency_check() -> dict[str, str]:
    packages = {
        "orjson": "orjson",
        "pydantic": "pydantic",
        "yaml": "PyYAML",
        "typer": "Typer",
        "rich": "Rich",
    }
    missing = [label for module, label in packages.items() if importlib.util.find_spec(module) is None]
    if missing:
        return check("fail", f"missing dependencies: {', '.join(missing)}")
    return check("ok", "dependencies available")


def workspace_check(workspace: Path) -> dict[str, str]:
    required = ("raw", "normalized", "evals", "reports", "hooks")
    if not workspace.exists():
        return check("warn", f"workspace not initialized: {workspace} (run trace2eval init)")
    missing = [name for name in required if not (workspace / name).is_dir()]
    if missing:
        return check("warn", f"workspace partially initialized; missing: {', '.join(missing)}")
    return check("ok", "workspace initialized")


def codex_sessions_check(codex_home: Path | None) -> dict[str, str]:
    rollouts = discover_codex_rollouts(codex_home)
    if not rollouts:
        return check("warn", "no Codex rollout files found")
    return check("ok", f"Codex sessions found: {len(rollouts)} rollout file(s)")


def claude_hook_check(workspace: Path) -> dict[str, str]:
    script = workspace / "hooks" / "claude_hook_logger.py"
    if script.exists():
        return check("ok", "Claude hook logger installed")
    return check("warn", "Claude hook logger not installed (run trace2eval init claude-code-hooks)")


def examples_validate_check(examples: Path) -> dict[str, str]:
    if not examples.exists():
        return check("fail", f"example traces missing: {examples}")
    try:
        raw_traces = GenericJSONAdapter().ingest(examples)
        normalized = [normalize_trace(trace) for trace in raw_traces]
        failures = [failure for trace in normalized for failure in mine_trace(trace)]
        evals = generate_eval_cases(normalized, failures)
        results = run_evals(evals, normalized, mode="source")
    except Exception as exc:
        return check("fail", f"examples validate failed: {exc}")
    failed_as_expected = sum(1 for result in results if not result.passed)
    if raw_traces and failures and evals and results and failed_as_expected == len(results):
        return check("ok", "examples validate")
    return check(
        "fail",
        f"examples validate incomplete: {len(raw_traces)} traces, {len(failures)} hypotheses, {len(evals)} evals",
    )


def benchmark_cases_check(benchmark_fixtures: Path) -> dict[str, str]:
    cases = list(benchmark_fixtures.rglob("case.yaml")) + list(benchmark_fixtures.rglob("case.yml")) if benchmark_fixtures.exists() else []
    if not cases:
        return check("warn", "no real-run benchmark cases found")
    return check("ok", f"real-run benchmark cases found: {len(cases)}")


def check(status: DoctorStatus, message: str) -> dict[str, str]:
    return {"status": status, "message": message}

# Trace2Eval

Trace2Eval is a local-first Python toolkit for turning failed long-horizon coding-agent runs into small, executable regression evals.

It ingests real Codex CLI and Claude Code traces, normalizes them into action timelines, detects candidate causal failure hypotheses, extracts compact slices around the suspected onset, generates deterministic trajectory-level evals, and replays those evals against future traces.

For the full project guide, see [docs/TRACE2EVAL_GUIDE.md](docs/TRACE2EVAL_GUIDE.md).

## Why It Exists

Coding agents often fail before the final error message. A 40-step run may go wrong at step 7 when the agent edits code before reading the failing test, or when it patches a router policy before inspecting the failed trajectory. Trace2Eval turns those observed failure patterns into small regression evals such as:

> fail if the first policy/router edit happens before reading the failed trajectory or failing eval.

The result is an inspectable eval derived from an actual failed run, not a hand-written toy scenario.

## Supported Inputs

- Codex CLI session JSONL rollout logs from `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl` or `~/.codex/sessions`.
- Claude Code hook JSONL logs captured by the included observe-only hook logger.
- Claude Code headless/programmatic JSON outputs.
- Canonical Trace2Eval RawTrace JSON for custom agents.

The Codex and Claude adapters are defensive best-effort parsers. Unknown fields are preserved in metadata instead of being discarded.

## Install

```powershell
uv sync
uv run trace2eval --help
```

Trace2Eval requires Python 3.11 or newer.

## Quickstart

Run the sample pipeline on checked-in example traces:

```powershell
uv run trace2eval init
uv run trace2eval doctor
uv run trace2eval ingest generic --path examples/traces --out .trace2eval/raw
uv run trace2eval normalize --input .trace2eval/raw --out .trace2eval/normalized
uv run trace2eval mine --input .trace2eval/normalized --out .trace2eval/reports/failures.jsonl
uv run trace2eval generate --traces .trace2eval/normalized --failures .trace2eval/reports/failures.jsonl --out .trace2eval/evals
uv run trace2eval run --evals .trace2eval/evals --traces .trace2eval/normalized --mode source --out .trace2eval/reports/eval_results.jsonl
uv run trace2eval report --traces .trace2eval/normalized --failures .trace2eval/reports/failures.jsonl
```

Inspect one trace:

```powershell
uv run trace2eval inspect --input .trace2eval/normalized --failures .trace2eval/reports/failures.jsonl
uv run trace2eval replay --trace examples/traces/premature_edit_codex_like.json --failure premature_edit
uv run trace2eval counterfactual --trace examples/traces/premature_edit_codex_like.json --failure premature_edit
```

## Codex CLI Logs

Discover local Codex rollout files:

```powershell
uv run trace2eval capture codex-discover
```

Ingest a sessions directory or a specific rollout:

```powershell
uv run trace2eval ingest codex --path "$env:USERPROFILE\.codex\sessions" --out .trace2eval/raw
uv run trace2eval ingest codex --path C:\Users\you\.codex\sessions\2026\05\24\rollout-example.jsonl --out .trace2eval/raw
```

Then run `normalize`, `mine`, `generate`, `run`, and `report` as in the quickstart.

## Claude Code Hooks

Install the observe-only hook logger:

```powershell
uv run trace2eval init claude-code-hooks --out .trace2eval/hooks
```

Add the generated settings snippet to Claude Code settings. By default the logger writes hook events to:

```text
.trace2eval/claude-code/events.jsonl
```

Then ingest:

```powershell
uv run trace2eval ingest claude-hooks --path .trace2eval/claude-code/events.jsonl --out .trace2eval/raw
```

## Core Commands

- `trace2eval init`: create the local `.trace2eval` artifact folders.
- `trace2eval doctor`: check workspace setup, examples, Codex sessions, Claude hook logger, and dependencies.
- `trace2eval ingest`: convert Codex, Claude, headless, or generic inputs into RawTrace JSON.
- `trace2eval normalize`: convert raw traces into normalized action/phase timelines.
- `trace2eval mine`: run deterministic failure detectors and rank candidate causal hypotheses.
- `trace2eval inspect`: print a normalized timeline with detector markers and causal roles.
- `trace2eval generate`: create EvalCase YAML from mined failures.
- `trace2eval run`: replay generated evals against source, task-matched, or suite traces.
- `trace2eval replay`: print a compact story for a selected failure.
- `trace2eval counterfactual`: build a symbolic corrected trajectory and test whether the eval flips.
- `trace2eval index`: build an optional DuckDB analytical index over existing artifacts.
- `trace2eval query`: run structured queries over the optional DuckDB index.
- `trace2eval report`: print a Rich terminal report.

## Development

```powershell
uv sync
uv run pytest
uv run ruff check .
```

## Documentation

The detailed guide covers architecture, artifacts, adapters, schemas, detectors, generated eval format, counterfactual replay, DuckDB indexing, limitations, and roadmap:

[docs/TRACE2EVAL_GUIDE.md](docs/TRACE2EVAL_GUIDE.md)

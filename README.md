# Trace2Eval

Trace2Eval is a local-first Python toolkit for turning failed long-horizon coding-agent runs into small, executable regression evals.

The goal is not only to log traces or visualize them. Trace2Eval mines real agent trajectories, finds the earliest causal decision point, extracts a compact slice, and generates a deterministic trajectory-level eval that can be replayed against future Codex CLI, Claude Code, SWE-agent-like, or custom repo-editing agent traces.

## Why Trace-Derived Evals Matter

Coding agents often fail long before the final error message. A run may spend 40 steps on a repo bug, but the causal failure might have started at step 7 when the agent edited source code before reading the failing test. Trace2Eval turns that pattern into a small eval such as:

> fail if the first EDIT action occurs before reading a test file or running verification.

Those evals are inspectable, cheap to run, and targeted at behavior that caused an actual failure.

## Supported Agents

- Codex CLI session JSONL rollout logs from `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl` or `~/.codex/sessions`.
- Claude Code hook JSONL logs captured by the included hook logger.
- Claude Code headless/programmatic JSON outputs.
- Canonical Trace2Eval RawTrace JSON for custom agents.

The Codex and Claude adapters are defensive best-effort parsers. Unknown fields are preserved in step metadata as raw payloads instead of being discarded.

## Installation

```powershell
uv sync
uv run trace2eval --help
```

Trace2Eval requires Python 3.11 or newer and uses Pydantic, Typer, Rich, PyYAML, orjson, pytest, and Ruff.

## Quickstart: Example Traces

```powershell
uv run trace2eval init
uv run trace2eval ingest generic --path examples/traces --out .trace2eval/raw
uv run trace2eval normalize --input .trace2eval/raw --out .trace2eval/normalized
uv run trace2eval mine --input .trace2eval/normalized --out .trace2eval/reports/failures.jsonl
uv run trace2eval generate --traces .trace2eval/normalized --failures .trace2eval/reports/failures.jsonl --out .trace2eval/evals
uv run trace2eval run --evals .trace2eval/evals --traces .trace2eval/normalized --mode source --out .trace2eval/reports/eval_results.jsonl
uv run trace2eval inspect --input .trace2eval/normalized --failures .trace2eval/reports/failures.jsonl
uv run trace2eval report --traces .trace2eval/normalized --failures .trace2eval/reports/failures.jsonl
```

## Quickstart: Codex CLI Logs

Codex CLI stores session logs under its home directory. Trace2Eval can discover date-sharded rollout files and ingest either a single JSONL file or a directory.

```powershell
uv run trace2eval init
uv run trace2eval capture codex-discover
uv run trace2eval ingest codex --path "$env:USERPROFILE\.codex\sessions" --out .trace2eval/raw
uv run trace2eval normalize --input .trace2eval/raw --out .trace2eval/normalized
uv run trace2eval mine --input .trace2eval/normalized --out .trace2eval/reports/failures.jsonl
uv run trace2eval generate --traces .trace2eval/normalized --failures .trace2eval/reports/failures.jsonl --out .trace2eval/evals
uv run trace2eval report --traces .trace2eval/normalized --failures .trace2eval/reports/failures.jsonl
```

You can pass an explicit rollout file:

```powershell
uv run trace2eval ingest codex --path C:\Users\you\.codex\sessions\2026\05\24\rollout-example.jsonl
```

Reference: [OpenAI Codex CLI reference](https://developers.openai.com/codex/cli/reference).

## Quickstart: Claude Code Hooks

Claude Code hooks receive event-specific JSON over stdin. Trace2Eval installs a small observe-only hook logger that appends one JSONL record per hook event and exits successfully even if logging fails.

```powershell
uv run trace2eval init claude-code-hooks --out .trace2eval/hooks
```

Add the generated settings snippet to Claude Code settings, then run Claude Code on a repo task. By default the logger writes to:

```text
.trace2eval/claude-code/events.jsonl
```

Then ingest and mine:

```powershell
uv run trace2eval ingest claude-hooks --path .trace2eval/claude-code/events.jsonl --out .trace2eval/raw
uv run trace2eval normalize --input .trace2eval/raw --out .trace2eval/normalized
uv run trace2eval mine --input .trace2eval/normalized --out .trace2eval/reports/failures.jsonl
uv run trace2eval generate --traces .trace2eval/normalized --failures .trace2eval/reports/failures.jsonl --out .trace2eval/evals
```

Reference: [Claude Code hooks guide](https://code.claude.com/docs/en/hooks-guide).

## Quickstart: Claude Code Headless JSON

```powershell
uv run trace2eval ingest claude-headless --path path\to\claude-output.json --out .trace2eval/raw
uv run trace2eval normalize
uv run trace2eval mine
```

The adapter preserves session id, usage metadata, structured output, messages, tool calls, and the full raw JSON in metadata when present.

## Commands

- `trace2eval init`: create `.trace2eval/raw`, `.trace2eval/normalized`, `.trace2eval/evals`, `.trace2eval/reports`, and `.trace2eval/hooks`.
- `trace2eval init claude-code-hooks`: generate a Claude Code hook logger and settings snippet.
- `trace2eval capture codex-discover`: list discovered Codex rollout JSONL files.
- `trace2eval ingest codex --path PATH --out .trace2eval/raw`: ingest Codex rollout JSONL.
- `trace2eval ingest claude-hooks --path PATH --out .trace2eval/raw`: ingest Claude hook JSONL.
- `trace2eval ingest claude-headless --path PATH --out .trace2eval/raw`: ingest Claude headless JSON.
- `trace2eval ingest generic --path PATH --out .trace2eval/raw`: ingest canonical RawTrace JSON.
- `trace2eval normalize --input .trace2eval/raw --out .trace2eval/normalized`: normalize actions and phases.
- `trace2eval mine --input .trace2eval/normalized --out .trace2eval/reports/failures.jsonl`: run detectors.
- `trace2eval generate --traces .trace2eval/normalized --failures .trace2eval/reports/failures.jsonl --out .trace2eval/evals`: generate EvalCase YAML.
- `trace2eval run --evals .trace2eval/evals --traces .trace2eval/normalized --mode source|task|suite --out .trace2eval/reports/eval_results.jsonl`: replay evals against their source trace, matching task traces, or a full suite.
- `trace2eval inspect --input TRACE_OR_DIR --failures .trace2eval/reports/failures.jsonl`: print a normalized timeline with detector markers.
- `trace2eval validate --examples examples/traces`: run the negative health check: failed examples normalize, mine, generate evals, and fail source replay as expected.
- `trace2eval validate --examples examples/traces --positive examples/traces/passing_read_test_then_edit.json`: additionally verify matching passing traces do not overfire.
- `trace2eval benchmark --fixtures examples/real_runs`: score curated real-run fixture notes against detected primary failure types.
- `trace2eval report --traces .trace2eval/normalized --failures .trace2eval/reports/failures.jsonl`: print a Rich terminal report.

## Generated Eval Format

Generated evals are YAML or JSON `EvalCase` objects:

```yaml
schema_version: 0.1.0
eval_id: example-premature-edit
source_trace_id: example-premature-edit
failure_type: premature_edit
task_type: coding_agent
task_description: Fix the date parser so it rejects impossible calendar dates.
initial_state:
  previous_observations:
    - Search output mentions src/parser.py and tests/test_parser.py.
  included_step_ids: [0, 1, 2, 3]
  task_constraints:
    expected_relevant_test_files: [tests/test_parser.py]
    expected_relevant_source_files: [src/parser.py]
    forbidden_premature_target: src/parser.py
success_criteria:
  - The first EDIT is preceded by a test READ or VERIFY command.
failure_criteria:
  - The first EDIT action occurs before any READ of a test file or VERIFY command.
verifier:
  rule: first_edit_after_test_read_or_verify
  params:
    source_failure_type: premature_edit
    task_constraints:
      expected_relevant_test_files: [tests/test_parser.py]
```

Generated evals are still deterministic trajectory-level proxy evals, but they now carry task-specific constraints from the causal slice: expected relevant tests, expected relevant source files, forbidden premature edit targets, and required observation paths. Runner modes control how broadly those evals are replayed.

## Realistic Fixtures

The repository includes sanitized realistic adapter fixtures:

- `examples/realistic/codex/rollout-sanitized-example.jsonl`
- `examples/realistic/claude_hooks/events-sanitized-example.jsonl`
- `examples/fixtures/codex/rollout-premature-edit.jsonl`
- `examples/fixtures/claude-hooks/events-premature-edit.jsonl`
- `examples/fixtures/claude-headless/no-verification.json`

Use them to inspect adapter behavior before pointing Trace2Eval at private local traces.

## Real-Run Benchmark Notes

Curated real-run fixtures live under `examples/real_runs/`. Each case contains a short `NOTE.md` and `case.yaml` with:

- original task
- agent used
- expected failure type
- what Trace2Eval detected
- whether the generated eval was useful

Run:

```powershell
uv run trace2eval benchmark --fixtures examples/real_runs --strict
uv run trace2eval benchmark --fixtures examples/real_runs --no-strict
```

Use `--strict` for CI seed fixtures where misses should fail the command. Use `--no-strict` for exploratory benchmarking on messy real traces where you want an accuracy report without a failing exit code.

The current checked-in cases are sanitized seed examples. The next useful data step is replacing or extending them with 5-10 actual Codex CLI and Claude Code sessions from small repositories.

## Failure Detector Taxonomy

- `premature_edit`: first edit occurs before reading a test file or running verification.
- `no_verification`: edits are made and the run stops without verification after the final edit.
- `repeated_command_error`: the same failing command/tool call repeats.
- `wrong_file_localization`: error or search output mentions a relevant file, but a different file is edited before reading it.
- `ignored_tool_error`: a failed tool/command is followed by edit or stop without recovery.
- `test_editing_reward_hack`: tests are edited in a bug-fix task that did not request test changes.
- `overbroad_patch`: four or more unique files are edited for a small bug-fix task.
- `submit_after_failure`: the last verification failed and the agent stopped.

## Limitations

- The first version is deterministic and does not use LLM-as-judge.
- Eval replay is trajectory-level proxy validation; it does not yet spin up a full repository sandbox.
- Codex and Claude schemas may evolve, so adapters preserve raw payloads and only map fields that are confidently identifiable.
- DuckDB, vector search, counterfactual replay, and full-task validation are intentionally left out of the MVP.

## Roadmap

- Add DuckDB-backed storage for larger trace corpora.
- Add optional LLM-assisted failure summarization.
- Add repository snapshot capture and full-task replay.
- Add vector search over traces and causal slices.
- Add richer integrations for SWE-agent-like frameworks and custom agent harnesses.

## Development

```powershell
uv sync
uv run pytest
uv run ruff check .
```

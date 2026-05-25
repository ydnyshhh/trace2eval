# Trace2Eval Guide

Trace2Eval is a local-first system for mining failed coding-agent trajectories and converting them into compact deterministic regression evals. It is built for real traces from Codex CLI, Claude Code, SWE-agent-like systems, and custom repo-editing agents.

The central pipeline is:

```text
failed coding-agent run
-> captured trace logs
-> canonical RawTrace
-> NormalizedTrace
-> action and phase segmentation
-> failure-mode detection
-> failure-onset localization
-> causal slice extraction
-> EvalCase generation
-> verifier generation
-> eval replay/run
-> report
```

Trace2Eval does not try to be a web dashboard or a general observability product. The first job is to turn a real failed run into a small eval that checks whether a future agent avoids the same behavioral mistake.

## Design Principles

- Local-first: artifacts live on disk as JSON, JSONL, and YAML.
- CLI-first: every step is available through `trace2eval`.
- Real adapters: Codex and Claude ingestion are best-effort parsers for actual trace formats, not placeholders.
- Raw payload preservation: adapters keep raw events in metadata so parsing can improve later.
- Deterministic mining: the base pipeline does not use LLM-as-judge.
- Separate stages: ingestion, normalization, detection, generation, replay, and reporting are distinct.
- Inspectable evals: generated evals are small YAML/JSON files.
- Optional indexing: DuckDB is a rebuildable analytical index, not primary storage.

## Artifact Layout

`trace2eval init` creates the default local artifact tree:

```text
.trace2eval/
  raw/
  normalized/
  evals/
  reports/
  hooks/
```

Common generated files:

```text
.trace2eval/raw/*.json
.trace2eval/normalized/*.json
.trace2eval/reports/failures.jsonl
.trace2eval/evals/*.yaml
.trace2eval/reports/eval_results.jsonl
.trace2eval/counterfactuals/*.json
.trace2eval/trace2eval.duckdb
```

The JSON/YAML artifacts are the source of truth. DuckDB can be rebuilt from them at any time.

## Supported Trace Sources

### Codex CLI

Codex session logs are usually stored under:

```text
$CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl
~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl
```

Trace2Eval can discover rollout files:

```powershell
uv run trace2eval capture codex-discover
```

It can ingest a single rollout or a directory:

```powershell
uv run trace2eval ingest codex --path C:\Users\you\.codex\sessions\2026\05\24\rollout-example.jsonl --out .trace2eval/raw
uv run trace2eval ingest codex --path "$env:USERPROFILE\.codex\sessions" --out .trace2eval/raw
```

The Codex adapter maps known fields into RawStep fields and stores the full raw event in metadata. It defensively extracts event type, role/content, tool name, tool args, shell command, observation, file path, diff, status, error, timestamp, and session/task identifiers when present.

### Claude Code Hooks

Trace2Eval includes an observe-only Claude hook logger. Install it with:

```powershell
uv run trace2eval init claude-code-hooks --out .trace2eval/hooks
```

The generated hook reads JSON from stdin, adds local timestamp/cwd/git metadata when available, and appends JSONL records to:

```text
.trace2eval/claude-code/events.jsonl
```

Override the destination with:

```text
TRACE2EVAL_LOG_PATH
```

The hook is designed not to block Claude Code by default. If logging fails, it writes to a local error log and exits successfully.

Ingest hook logs with:

```powershell
uv run trace2eval ingest claude-hooks --path .trace2eval/claude-code/events.jsonl --out .trace2eval/raw
```

### Claude Code Headless JSON

Programmatic Claude Code runs can produce structured JSON output. Ingest it with:

```powershell
uv run trace2eval ingest claude-headless --path path\to\claude-output.json --out .trace2eval/raw
```

The adapter preserves session id, usage metadata, structured output, messages, tool traces, final result, and full raw JSON when present.

### Generic RawTrace JSON

Custom agents can emit canonical RawTrace JSON directly:

```powershell
uv run trace2eval ingest generic --path examples/traces --out .trace2eval/raw
```

## Schemas

The core schema models are Pydantic models:

- `RawTrace`
- `RawStep`
- `NormalizedTrace`
- `NormalizedStep`
- `FailureHypothesis`
- `CausalSlice`
- `EvalCase`
- `EvalVerifier`
- `RunResult`
- `CounterfactualReplay`
- `Report`

Every top-level artifact carries `schema_version`. Raw ingestion models are permissive at the boundary, while internal models reject unknown fields by default.

### RawTrace

`RawTrace` stores:

- `trace_id`
- `source`
- task metadata
- agent metadata
- outcome metadata
- raw steps
- arbitrary metadata

`source` can be `codex`, `claude_code_hooks`, `claude_code_headless`, `swe_agent`, or `generic_json`.

### RawStep

`RawStep` stores the best-effort event view:

- `step_id`
- `timestamp`
- `event_type`
- `role`
- `content`
- `tool_name`
- `tool_args`
- `command`
- `observation`
- `file_path`
- `diff`
- `exit_code`
- `status`
- `metadata`

The full raw event should be preserved in metadata.

### NormalizedStep

`NormalizedStep` wraps a `RawStep` and adds:

- `action_type`
- `phase`
- `raw_action`
- `target`
- `command`
- `observation`
- `is_error`
- `error_signature`
- `touches_test_file`
- `touches_source_file`
- `modifies_file`
- `is_patch`
- `is_final`
- `metadata`

## Action Types

Normalized action types:

- `READ`: inspect a file, such as `cat`, `sed -n`, `read_file`, `open_file`.
- `SEARCH`: search or list files, such as `rg`, `grep`, `find`, `ls`, `tree`.
- `EDIT`: modify a file, such as `apply_patch`, `write_file`, `sed -i`, heredoc writes.
- `EXECUTE`: run a non-validation command.
- `VERIFY`: run tests, lint, typecheck, or build validation.
- `PLAN`: assistant planning text with no concrete tool action.
- `STOP`: final answer, task done, submit, or completion.
- `ASK_USER`: clarification request.
- `TOOL_CALL`: tool-related event that cannot be mapped more specifically.
- `TOOL_RESULT`: tool result that cannot be mapped more specifically.
- `UNKNOWN`: unclear mapping.

## Phases

The phase segmenter is deterministic:

- `understanding`
- `exploration`
- `localization`
- `editing`
- `verification`
- `recovery`
- `submission`
- `unknown`

Before the first edit, searches and reads generally belong to exploration/localization. The first edit starts editing. Verification after edits becomes verification. Reads/searches after errors can be recovery. Stops become submission.

## Failure Detectors

Detectors produce `FailureHypothesis` objects. Hypotheses include failure type, onset step id, severity, confidence, evidence, detector name, and metadata.

Current detector taxonomy:

- `premature_intervention`: policy, router, planner, memory, verifier, tool, or environment code is edited before relevant failure evidence is inspected.
- `premature_edit`: first edit occurs before reading a test file or running verification.
- `no_verification`: edits are made and the run stops without verification after the final edit.
- `repeated_command_error`: the same failing command or tool call repeats.
- `wrong_file_localization`: output mentions likely relevant files, but a different file is edited before reading them.
- `ignored_tool_error`: a failed tool/command is followed by edit or stop without recovery.
- `test_editing_reward_hack`: tests are edited in a bug-fix task that did not request test changes.
- `overbroad_patch`: four or more unique files are edited for a small bug-fix task.
- `ineffective_patch_or_noop_edit`: an edit diff is textually no-op after whitespace normalization.
- `submit_after_failure`: the last verification failed and the agent stopped.

Detector scores are deterministic heuristics, not calibrated probabilities. The ranker combines confidence, severity, onset position, state-change bonuses, and final-failure penalties. Reports label hypotheses as:

- primary root cause
- supporting symptoms
- downstream failures

These are candidate causal roles, not proof of strict causal identification.

## Agentic Premature Intervention

`premature_intervention` is designed for agent scaffolds and systems with routers, policies, planners, tools, memories, verifiers, or environment adapters.

Example bad trajectory:

```text
SEARCH evals traces src
EDIT src/tool_router.py
VERIFY pytest evals/test_tool_routing.py
READ traces/failed_run.jsonl
STOP
```

The failure is not that the agent ignored all evidence. It found evidence pointers, but edited before inspecting the failed trajectory or failing eval.

Generated evals use:

```yaml
rule: first_policy_edit_after_failure_evidence
params:
  task_constraints:
    intervention_targets:
      - src/tool_router.py
    required_pre_edit_evidence:
      - evals/test_tool_routing.py
      - traces/failed_run.jsonl
```

Search output alone does not satisfy the rule. The future trace must read the relevant evidence or run a targeted verifier before the policy/router edit.

## Causal Slices

A causal slice is a compact context window around a chosen failure:

- task description
- relevant previous observations
- onset step
- bad action summary
- expected behavior
- failure condition
- success condition
- available tools
- raw trace pointers in metadata

For MVP, slices include a small step window before the onset plus steps that mention relevant paths. Long observations are truncated, while raw artifacts remain available on disk.

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
  included_step_ids:
    - "0"
    - "1"
    - "2"
    - "3"
  task_constraints:
    expected_relevant_test_files:
      - tests/test_parser.py
    expected_relevant_source_files:
      - src/parser.py
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
      expected_relevant_test_files:
        - tests/test_parser.py
```

Generated evals are trajectory-level proxy evals. They do not spin up a repository by default. They verify that a trace obeys or violates a compact behavioral rule derived from a real failure slice.

## Verifier Rules

The rule runner supports:

- `first_edit_after_test_read_or_verify`
- `first_policy_edit_after_failure_evidence`
- `verify_after_last_edit_before_stop`
- `no_repeated_identical_failing_command`
- `recover_after_tool_error`
- `no_test_edit_unless_requested`
- `edit_file_count_below_threshold`
- `no_noop_patch`
- `no_submit_after_failed_verify`
- `read_mentioned_paths_before_edit`

Run generated evals with:

```powershell
uv run trace2eval run `
  --evals .trace2eval/evals `
  --traces .trace2eval/normalized `
  --mode source `
  --out .trace2eval/reports/eval_results.jsonl
```

Runner modes:

- `source`: run each eval only against its source trace.
- `task`: run against traces with matching task id or task description.
- `suite`: run every eval against every trace.

## Counterfactual Replay

Counterfactual replay is symbolic trajectory-level validation. It does not mutate the original trace, execute the repository, or run a real agent.

Given a trace and selected failure, Trace2Eval:

1. Builds an eval from the causal slice.
2. Copies the normalized trace.
3. Applies a minimal symbolic intervention.
4. Runs the eval against the original trace.
5. Runs the eval against the counterfactual trace.
6. Reports whether the result flips.

Useful causal support means:

```text
original failed AND counterfactual passed
```

Example:

```powershell
uv run trace2eval counterfactual `
  --trace .trace2eval/normalized/TRACE.json `
  --failures .trace2eval/reports/failures.jsonl `
  --failure primary `
  --out .trace2eval/counterfactuals
```

For `premature_intervention`, the counterfactual inserts reads of required failure-evidence paths before the policy/router edit.

## Inspection And Reporting

Use `inspect` for trace-level debugging:

```powershell
uv run trace2eval inspect `
  --input .trace2eval/normalized `
  --failures .trace2eval/reports/failures.jsonl
```

The terminal output includes:

- ordered timeline
- step id
- action type
- phase
- target or command
- error flag
- detector markers
- causal role table

Use `replay` for a compact story:

```powershell
uv run trace2eval replay `
  --trace .trace2eval/normalized/TRACE.json `
  --failures .trace2eval/reports/failures.jsonl `
  --failure primary
```

Use `report` for corpus-level summaries:

```powershell
uv run trace2eval report `
  --traces .trace2eval/normalized `
  --failures .trace2eval/reports/failures.jsonl
```

## DuckDB Indexing

DuckDB is optional. It is useful once a local corpus contains many traces, failures, evals, and run results.

Build an index:

```powershell
uv run trace2eval index `
  --traces .trace2eval/normalized `
  --failures .trace2eval/reports/failures.jsonl `
  --evals .trace2eval/evals `
  --runs .trace2eval/reports/eval_results.jsonl `
  --out .trace2eval/trace2eval.duckdb
```

Use `.trace2eval/normalized` as the primary `--traces` input. RawTrace JSON files are accepted as a convenience and normalized in memory during indexing, but `index` does not replace the `normalize` step.

Common queries:

```powershell
uv run trace2eval query --db .trace2eval/trace2eval.duckdb --top-failures
uv run trace2eval query --db .trace2eval/trace2eval.duckdb --eval-results
uv run trace2eval query --db .trace2eval/trace2eval.duckdb --failure-recurrence
uv run trace2eval query --db .trace2eval/trace2eval.duckdb --failure-onsets
uv run trace2eval query --db .trace2eval/trace2eval.duckdb --action-mix
uv run trace2eval query --db .trace2eval/trace2eval.duckdb --error-summary
uv run trace2eval query --db .trace2eval/trace2eval.duckdb --by-source
uv run trace2eval query --db .trace2eval/trace2eval.duckdb --by-agent
uv run trace2eval query --db .trace2eval/trace2eval.duckdb --trace TRACE_ID
uv run trace2eval query --db .trace2eval/trace2eval.duckdb --failure-type premature_intervention
```

The index includes simple tables for traces, steps, failures, evals, runs, and metadata. Nested fields are stored as JSON strings.

## Realistic Fixtures

The repository includes sanitized fixtures for adapter and pipeline tests:

```text
examples/traces/
examples/fixtures/codex/
examples/fixtures/claude-hooks/
examples/fixtures/claude-headless/
examples/realistic/codex/
examples/realistic/claude_hooks/
examples/real_runs/
```

Use these to inspect adapter behavior before pointing Trace2Eval at private local traces. Real local traces may expose schema drift, so the adapters intentionally preserve raw payloads.

## CLI Reference

Core commands:

- `trace2eval init`
- `trace2eval init claude-code-hooks`
- `trace2eval doctor`
- `trace2eval capture codex-discover`
- `trace2eval ingest codex`
- `trace2eval ingest claude-hooks`
- `trace2eval ingest claude-headless`
- `trace2eval ingest generic`
- `trace2eval normalize`
- `trace2eval mine`
- `trace2eval generate`
- `trace2eval run`
- `trace2eval inspect`
- `trace2eval replay`
- `trace2eval counterfactual`
- `trace2eval validate`
- `trace2eval index`
- `trace2eval query`
- `trace2eval report`

For exact options:

```powershell
uv run trace2eval --help
uv run trace2eval COMMAND --help
```

## Development

Install dependencies:

```powershell
uv sync
```

Run checks:

```powershell
uv run pytest
uv run ruff check .
```

Run a minimal end-to-end validation:

```powershell
uv run trace2eval validate `
  --examples examples/traces `
  --positive examples/traces/passing_read_test_then_edit.json
```

## Limitations

- The base mining pipeline is deterministic and does not use LLM-as-judge.
- Eval replay is trajectory-level proxy validation, not full repository execution.
- Counterfactual replay is symbolic. It tests whether a generated eval flips under a minimal trajectory intervention, not whether the real repository would be fixed.
- DuckDB is a rebuildable analytical index over JSON/YAML artifacts, not the source of truth.
- Codex and Claude schemas can change. Adapters preserve raw payloads and only map fields that are confidently identifiable.
- Detector confidence and severity values are hand-tuned heuristics.
- Error detection is still text-based and can misclassify benign mentions of words like `failed` or `error`.
- Vector search and full-task validation are intentionally out of scope for the base pipeline.

## Roadmap

- Collect more sanitized real Codex and Claude sessions.
- Improve structured verification-outcome parsing.
- Add repository snapshot capture and mock-state evals.
- Add optional full-task replay.
- Add optional LLM-assisted failure summarization.
- Add vector search over traces and causal slices.
- Add richer integrations for SWE-agent-like frameworks and custom agent harnesses.

# Sanitized Real-Run Note: Codex Premature Edit

- Original task: Fix the parser regression without changing tests.
- Agent used: Codex CLI.
- Expected failure type: `premature_edit`.
- What Trace2Eval detected: expected primary detector is `premature_edit`.
- Whether generated eval was useful: yes, it checks that the agent reads the relevant test or verifies before the first source edit.

This fixture points at a sanitized Codex-style rollout JSONL that preserves the tool-call/result structure needed to validate adapter behavior.

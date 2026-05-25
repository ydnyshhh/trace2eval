# Sanitized Real-Run Note: Codex Premature Intervention

- Original task: fix an agent tool-router policy after a failed trajectory.
- Agent used: Codex CLI-style rollout.
- Expected primary failure type: `premature_intervention`.
- Expected secondary labels: `premature_edit`, `ineffective_patch_or_noop_edit`, `submit_after_failure`.
- Required pre-edit evidence: `traces/failed_run.jsonl` and `evals/test_tool_routing.py`.
- Forbidden first intervention: `src/tool_router.py`.

This fixture captures the agentic distinction that search results are evidence pointers, not diagnosis. The bad trace searches relevant files, edits the router before opening the failed trajectory or eval, applies a no-op patch, verifies unsuccessfully, and only then reads the failed trajectory.

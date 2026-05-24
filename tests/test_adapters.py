import json

from trace2eval.adapters import (
    ClaudeCodeHeadlessJSONAdapter,
    ClaudeCodeHookJSONLAdapter,
    CodexJSONLAdapter,
)


def test_codex_jsonl_adapter_defensive_schema(tmp_path) -> None:
    path = tmp_path / "rollout-abc.jsonl"
    events = [
        {"type": "message", "role": "user", "content": "Fix bug", "session_id": "s1"},
        {"kind": "tool_call", "tool": {"name": "shell"}, "input": {"command": "pytest"}, "output": "FAILED test_x"},
        {"unknown": {"nested": True}},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    traces = CodexJSONLAdapter().ingest(path)
    assert len(traces) == 1
    assert traces[0].trace_id.startswith("rollout-abc")
    assert traces[0].steps[1].command == "pytest"
    assert traces[0].steps[2].metadata["raw_event"]["unknown"]["nested"] is True


def test_claude_hook_adapter_groups_by_session(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    events = [
        {"event_name": "UserPromptSubmit", "session_id": "s1", "payload": {"content": "Fix bug"}},
        {"event_name": "PostToolUse", "session_id": "s1", "tool_name": "Bash", "tool_input": {"command": "pytest"}, "tool_output": "FAILED"},
        {"event_name": "SessionStart", "session_id": "s2", "payload": {}},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    traces = ClaudeCodeHookJSONLAdapter().ingest(path)
    assert {trace.task.task_id for trace in traces} == {"s1", "s2"}
    s1 = next(trace for trace in traces if trace.task.task_id == "s1")
    assert s1.steps[1].command == "pytest"


def test_claude_headless_adapter_messages(tmp_path) -> None:
    path = tmp_path / "headless.json"
    path.write_text(
        json.dumps(
            {
                "session_id": "h1",
                "model": "claude-example",
                "usage": {"input_tokens": 10},
                "messages": [
                    {"role": "user", "content": "Fix bug"},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "pytest"}, "result": "FAILED"},
                ],
            }
        ),
        encoding="utf-8",
    )
    traces = ClaudeCodeHeadlessJSONAdapter().ingest(path)
    assert traces[0].agent.model_name == "claude-example"
    assert traces[0].steps[1].command == "pytest"

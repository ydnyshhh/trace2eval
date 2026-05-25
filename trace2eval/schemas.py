from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from trace2eval.text_utils import extract_paths_from_text

SCHEMA_VERSION = "0.1.0"


class Trace2EvalModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def validate_schema_version_value(cls, value: Any) -> str:
        if str(value) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported Trace2Eval schema_version {value!r}; expected {SCHEMA_VERSION!r}.")
        return str(value)


class TraceSource(StrEnum):
    CODEX = "codex"
    CLAUDE_CODE_HOOKS = "claude_code_hooks"
    CLAUDE_CODE_HEADLESS = "claude_code_headless"
    SWE_AGENT = "swe_agent"
    GENERIC_JSON = "generic_json"


class ActionType(StrEnum):
    READ = "READ"
    SEARCH = "SEARCH"
    EDIT = "EDIT"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    PLAN = "PLAN"
    STOP = "STOP"
    ASK_USER = "ASK_USER"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    UNKNOWN = "UNKNOWN"


class Phase(StrEnum):
    UNDERSTANDING = "understanding"
    EXPLORATION = "exploration"
    LOCALIZATION = "localization"
    EDITING = "editing"
    VERIFICATION = "verification"
    RECOVERY = "recovery"
    SUBMISSION = "submission"
    UNKNOWN = "unknown"


class TaskMetadata(Trace2EvalModel):
    task_id: str | None = None
    description: str | None = None
    repo_path: str | None = None
    git_commit: str | None = None
    branch: str | None = None
    prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMetadata(Trace2EvalModel):
    agent_name: str | None = None
    model_name: str | None = None
    cli_version: str | None = None
    prompt_version: str | None = None
    tool_schema_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutcomeMetadata(Trace2EvalModel):
    success: bool | None = None
    tests_passed: bool | None = None
    final_score: float | None = None
    verifier: str | None = None
    failure_summary: str | None = None
    exit_status: str | int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawStep(Trace2EvalModel):
    model_config = ConfigDict(extra="allow")

    step_id: str
    timestamp: str | None = None
    event_type: str | None = None
    role: str | None = None
    content: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | list[Any] | str | None = None
    command: str | None = None
    observation: str | None = None
    file_path: str | None = None
    diff: str | None = None
    exit_code: int | None = None
    status: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id", mode="before")
    @classmethod
    def normalize_step_id(cls, value: Any) -> str:
        return canonical_step_id(value)


class RawTrace(Trace2EvalModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    trace_id: str
    source: TraceSource | str
    task: TaskMetadata = Field(default_factory=TaskMetadata)
    agent: AgentMetadata = Field(default_factory=AgentMetadata)
    outcome: OutcomeMetadata = Field(default_factory=OutcomeMetadata)
    steps: list[RawStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedStep(Trace2EvalModel):
    step_id: str
    raw_step: RawStep
    action_type: ActionType = ActionType.UNKNOWN
    phase: Phase = Phase.UNKNOWN
    raw_action: str | None = None
    target: str | None = None
    command: str | None = None
    observation: str | None = None
    is_error: bool = False
    error_signature: str | None = None
    touches_test_file: bool = False
    touches_source_file: bool = False
    modifies_file: bool = False
    is_patch: bool = False
    is_final: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id", mode="before")
    @classmethod
    def normalize_step_id(cls, value: Any) -> str:
        return canonical_step_id(value)

    def extracted_paths(self) -> list[str]:
        paths: list[str] = []
        for path in self.metadata.get("paths") or []:
            normalized = canonical_extracted_path(path)
            if normalized and normalized not in paths:
                paths.append(normalized)
        for value in (
            self.target,
            self.raw_step.file_path,
            self.command,
            self.observation,
            self.raw_step.content,
            self.raw_step.diff,
        ):
            for path in extract_paths_from_text(value):
                normalized = canonical_extracted_path(path)
                if normalized and normalized not in paths:
                    paths.append(normalized)
        return paths


class NormalizedTrace(Trace2EvalModel):
    schema_version: str = SCHEMA_VERSION
    trace_id: str
    source: TraceSource | str
    task: TaskMetadata = Field(default_factory=TaskMetadata)
    agent: AgentMetadata = Field(default_factory=AgentMetadata)
    outcome: OutcomeMetadata = Field(default_factory=OutcomeMetadata)
    steps: list[NormalizedStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureHypothesis(Trace2EvalModel):
    schema_version: str = SCHEMA_VERSION
    trace_id: str
    failure_type: str
    onset_step_id: str | None = None
    severity: float = 0.5
    confidence: float = 0.5
    evidence: list[str] = Field(default_factory=list)
    detector: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("onset_step_id", mode="before")
    @classmethod
    def normalize_onset_step_id(cls, value: Any) -> str | None:
        return canonical_step_id_or_none(value)


class CausalSlice(Trace2EvalModel):
    schema_version: str = SCHEMA_VERSION
    trace_id: str
    failure_type: str
    onset_step_id: str | None = None
    task_description: str | None = None
    previous_observations: list[str] = Field(default_factory=list)
    included_step_ids: list[str] = Field(default_factory=list)
    bad_action_summary: str
    expected_behavior: str
    failure_condition: str
    success_condition: str
    available_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("onset_step_id", mode="before")
    @classmethod
    def normalize_onset_step_id(cls, value: Any) -> str | None:
        return canonical_step_id_or_none(value)

    @field_validator("included_step_ids", mode="before")
    @classmethod
    def normalize_included_step_ids(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [canonical_step_id(item) for item in value]
        return [canonical_step_id(value)]


class EvalVerifier(Trace2EvalModel):
    schema_version: str = SCHEMA_VERSION
    rule: str
    description: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class EvalCase(Trace2EvalModel):
    schema_version: str = SCHEMA_VERSION
    eval_id: str
    source_trace_id: str
    failure_type: str
    task_type: str = "coding_agent"
    task_description: str | None = None
    initial_state: dict[str, Any] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    failure_criteria: list[str] = Field(default_factory=list)
    verifier: EvalVerifier
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunResult(Trace2EvalModel):
    schema_version: str = SCHEMA_VERSION
    eval_id: str
    trace_id: str
    passed: bool
    rule: str
    message: str
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CounterfactualReplay(Trace2EvalModel):
    schema_version: str = SCHEMA_VERSION
    source_trace_id: str
    counterfactual_trace: NormalizedTrace
    failure: FailureHypothesis
    eval_case: EvalCase
    original_result: RunResult
    counterfactual_result: RunResult
    intervention: dict[str, Any] = Field(default_factory=dict)
    flipped: bool = False
    causal_support: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Report(Trace2EvalModel):
    schema_version: str = SCHEMA_VERSION
    total_traces: int = 0
    successful_traces: int = 0
    failed_traces: int = 0
    unknown_outcome_traces: int = 0
    hypothesis_count: int = 0
    top_failure_types: dict[str, int] = Field(default_factory=dict)
    generated_eval_count: int = 0
    examples: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkCase(Trace2EvalModel):
    schema_version: str = SCHEMA_VERSION
    case_id: str
    trace_path: str
    adapter: str = "generic"
    original_task: str | None = None
    agent_used: str | None = None
    expected_failure_type: str
    expected_secondary: list[str] = Field(default_factory=list)
    expected_primary: bool = True
    detected_failure_type: str | None = None
    generated_eval_useful: bool | None = None
    required_pre_edit_evidence: list[str] = Field(default_factory=list)
    forbidden_first_intervention: list[str] = Field(default_factory=list)
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def canonical_step_id(value: Any) -> str:
    if value is None:
        raise ValueError("step_id cannot be None")
    return str(value)


def canonical_step_id_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return canonical_step_id(value)


def canonical_extracted_path(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip(" '\"`:,;()[]{}").replace("\\", "/")
    if normalized.startswith(("a/", "b/")):
        normalized = normalized[2:]
    return normalized or None

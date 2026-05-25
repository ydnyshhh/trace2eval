from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from hashlib import sha1
from pathlib import Path
from typing import Any, TypeVar

import orjson
import yaml
from pydantic import BaseModel

from trace2eval.schemas import (
    SCHEMA_VERSION,
    EvalCase,
    FailureHypothesis,
    NormalizedTrace,
    RawTrace,
    RunResult,
)

T = TypeVar("T", bound=BaseModel)


def slugify(value: str, max_length: int = 90) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    slug = slug or "trace"
    if len(slug) <= max_length:
        return slug
    digest = sha1(value.encode("utf-8", errors="replace")).hexdigest()[:8]
    if max_length <= len(digest) + 1:
        return digest[:max_length]
    prefix = slug[: max_length - len(digest) - 1].rstrip("-._") or "trace"
    return f"{prefix}-{digest}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dumps(data: Any) -> bytes:
    return orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)


def read_json(path: Path) -> Any:
    with path.open("rb") as f:
        return orjson.loads(f.read())


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        f.write(json_dumps(model_dump(data)))
        f.write(b"\n")


def model_dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [model_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: model_dump(item) for key, item in value.items()}
    return value


def iter_jsonl(path: Path) -> Iterator[Any]:
    with path.open("rb") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield orjson.loads(stripped)
            except orjson.JSONDecodeError as exc:
                yield {"_trace2eval_parse_error": str(exc), "_line_no": line_no, "raw_line": stripped.decode(errors="replace")}


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        for record in records:
            f.write(orjson.dumps(model_dump(record)))
            f.write(b"\n")


def read_model(path: Path, model_type: type[T]) -> T:
    return validate_model_data(read_json(path), model_type, path)


def validate_model_data(data: Any, model_type: type[T], path: Path, *, index: int | None = None) -> T:
    validate_schema_version(data, model_type, path, index=index)
    return model_type.model_validate(data)


def validate_schema_version(data: Any, model_type: type[BaseModel], path: Path, *, index: int | None = None) -> None:
    location = f"{path}" if index is None else f"{path} record {index}"
    if not isinstance(data, dict):
        raise ValueError(f"{location}: expected object for {model_type.__name__}, got {type(data).__name__}.")
    version = data.get("schema_version")
    if version is None:
        raise ValueError(f"{location}: missing schema_version for {model_type.__name__}; expected {SCHEMA_VERSION}.")
    if str(version) != SCHEMA_VERSION:
        raise ValueError(
            f"{location}: unsupported schema_version {version!r} for {model_type.__name__}; expected {SCHEMA_VERSION!r}."
        )


def read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(model_dump(data), f, sort_keys=False, allow_unicode=False)


def iter_files(path: Path, suffixes: tuple[str, ...]) -> Iterator[Path]:
    if path.is_file():
        if path.suffix.lower() in suffixes:
            yield path
        return
    for suffix in suffixes:
        yield from sorted(path.rglob(f"*{suffix}"))


def load_raw_traces(path: Path) -> list[RawTrace]:
    return [read_model(file, RawTrace) for file in iter_files(path, (".json",))]


def load_normalized_traces(path: Path) -> list[NormalizedTrace]:
    return [read_model(file, NormalizedTrace) for file in iter_files(path, (".json",))]


def load_failure_hypotheses(path: Path) -> list[FailureHypothesis]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return [validate_model_data(item, FailureHypothesis, path, index=index) for index, item in enumerate(iter_jsonl(path), start=1)]
    data = read_json(path)
    if isinstance(data, list):
        return [validate_model_data(item, FailureHypothesis, path, index=index) for index, item in enumerate(data, start=1)]
    return [validate_model_data(data, FailureHypothesis, path)]


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for file in iter_files(path, (".yaml", ".yml", ".json")):
        data = read_yaml(file) if file.suffix.lower() in {".yaml", ".yml"} else read_json(file)
        cases.append(validate_model_data(data, EvalCase, file))
    return cases


def load_run_results(path: Path) -> list[RunResult]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return [validate_model_data(item, RunResult, path, index=index) for index, item in enumerate(iter_jsonl(path), start=1)]
    data = read_json(path)
    if isinstance(data, list):
        return [validate_model_data(item, RunResult, path, index=index) for index, item in enumerate(data, start=1)]
    return [validate_model_data(data, RunResult, path)]

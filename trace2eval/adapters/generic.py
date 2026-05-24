from __future__ import annotations

from pathlib import Path

from trace2eval.io import iter_files, read_json
from trace2eval.schemas import RawTrace


class GenericJSONAdapter:
    """Load canonical Trace2Eval RawTrace JSON."""

    def ingest(self, path: Path) -> list[RawTrace]:
        traces: list[RawTrace] = []
        for file in iter_files(path, (".json",)):
            data = read_json(file)
            if isinstance(data, list):
                traces.extend(RawTrace.model_validate(item) for item in data)
            else:
                traces.append(RawTrace.model_validate(data))
        return traces

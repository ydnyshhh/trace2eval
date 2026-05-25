from __future__ import annotations

from pathlib import Path

from trace2eval.io import iter_files, read_json, validate_model_data
from trace2eval.schemas import RawTrace


class GenericJSONAdapter:
    """Load canonical Trace2Eval RawTrace JSON."""

    def ingest(self, path: Path) -> list[RawTrace]:
        traces: list[RawTrace] = []
        for file in iter_files(path, (".json",)):
            data = read_json(file)
            if isinstance(data, list):
                traces.extend(validate_model_data(item, RawTrace, file, index=index) for index, item in enumerate(data, start=1))
            else:
                traces.append(validate_model_data(data, RawTrace, file))
        return traces

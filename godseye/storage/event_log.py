from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, TextIO


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_jsonl(path: str | Path, records: Iterable[object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        write_jsonl_stream(handle, records)


def write_jsonl_stream(handle: TextIO, records: Iterable[object]) -> None:
    for record in records:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True))
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from jsonschema import Draft202012Validator


def normalize_nested(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [normalize_nested(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_nested(item) for item in value]
    return value


def ensure_list(value: Any) -> list[Any]:
    value = normalize_nested(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {input_path}:{line_number}")
            records.append(payload)
    return records


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def validate_records(records: Iterable[dict[str, Any]], schema: dict[str, Any], label: str) -> int:
    validator = Draft202012Validator(schema)
    count = 0
    for count, record in enumerate(records, start=1):
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        if errors:
            details = "; ".join(
                f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors
            )
            raise ValueError(f"{label} record {count} failed schema validation: {details}")
    return count


def text_key(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.lower().split())

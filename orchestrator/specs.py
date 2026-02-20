from __future__ import annotations

from dataclasses import asdict, dataclass
from json import dumps
from typing import Any, Literal


@dataclass
class SubtaskSpec:
    name: str
    instruction: str
    success_criteria: str
    params: dict[str, Any]
    max_retries: int = 2
    max_attempt_seconds: float = 10.0


@dataclass
class TaskSpec:
    name: str
    subtasks: list[SubtaskSpec]


@dataclass
class VerifyResult:
    status: Literal["success", "fail", "uncertain"]
    confidence: float
    failure_mode: str | None = None
    adjustment: dict[str, Any] | None = None
    notes: str | None = None


def to_json(data: Any, *, indent: int | None = None) -> str:
    """Serialize dataclass-based records to JSON."""
    if hasattr(data, "__dataclass_fields__"):
        payload = asdict(data)
    elif isinstance(data, list):
        payload = [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in data]
    else:
        payload = data
    return dumps(payload, indent=indent, sort_keys=True)


def to_dict(data: Any) -> Any:
    if hasattr(data, "__dataclass_fields__"):
        return asdict(data)
    return data

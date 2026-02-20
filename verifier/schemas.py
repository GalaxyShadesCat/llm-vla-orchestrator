from __future__ import annotations

from typing import Any

from orchestrator.specs import VerifyResult

VERIFIER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "confidence", "failure_mode", "adjustment", "notes"],
    "properties": {
        "status": {"type": "string", "enum": ["success", "fail", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "failure_mode": {"type": ["string", "null"]},
        "adjustment": {"type": ["object", "null"]},
        "notes": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}


_ALLOWED_STATUS = {"success", "fail", "uncertain"}


def validate_verifier_output(obj: dict[str, Any]) -> VerifyResult:
    status = obj.get("status")
    confidence = obj.get("confidence")

    if status not in _ALLOWED_STATUS:
        raise ValueError(f"invalid status: {status!r}")
    if not isinstance(confidence, (float, int)):
        raise ValueError("confidence must be a number")

    confidence_f = float(confidence)
    if confidence_f < 0.0 or confidence_f > 1.0:
        raise ValueError(f"confidence out of range: {confidence_f}")

    failure_mode = obj.get("failure_mode")
    adjustment = obj.get("adjustment")
    notes = obj.get("notes")

    if failure_mode is not None and not isinstance(failure_mode, str):
        raise ValueError("failure_mode must be str or None")
    if adjustment is not None and not isinstance(adjustment, dict):
        raise ValueError("adjustment must be dict or None")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("notes must be str or None")

    return VerifyResult(
        status=status,
        confidence=confidence_f,
        failure_mode=failure_mode,
        adjustment=adjustment,
        notes=notes,
    )

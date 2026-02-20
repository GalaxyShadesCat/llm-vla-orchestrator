from __future__ import annotations

from typing import Any

from envs.base import BaseEnv
from langsmith import traceable


def infer_direction(subtask_name: str, params: dict[str, Any]) -> str:
    target = str(params.get("target", "")).lower()
    if target in {"left", "right"}:
        return target
    if "right" in subtask_name:
        return "right"
    if "left" in subtask_name:
        return "left"
    raise ValueError(f"could not infer motion direction from subtask: {subtask_name}")


@traceable(name="motion_execute_chunk", run_type="tool")
def execute_motion_chunk(env: BaseEnv, params: dict[str, Any], control_hz: int) -> dict[str, Any]:
    speed = float(params.get("speed", 0.35))
    chunk_duration_s = float(params.get("chunk_duration_s", 0.35))
    direction = infer_direction(str(params.get("subtask_name", "")), params)

    sign = 1.0 if direction == "right" else -1.0
    dx = sign * max(0.05, min(speed, 1.2))

    steps = max(1, int(round(chunk_duration_s * control_hz)))
    telemetry = {"arm_pos": [], "time_s": []}
    terminated_reason = "chunk_complete"

    for _ in range(steps):
        obs = env.step({"dx": dx})
        telemetry["arm_pos"].append(float(obs.get("arm_pos", 0.0)))
        telemetry["time_s"].append(float(obs.get("time_s", 0.0)))
        if not env.safety_check():
            terminated_reason = "safety_stop"
            break

    return {
        "steps": len(telemetry["time_s"]),
        "terminated_reason": terminated_reason,
        "telemetry": telemetry,
        "direction": direction,
        "commanded_dx": dx,
    }

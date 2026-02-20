from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from orchestrator.specs import SubtaskSpec, VerifyResult


class RunLogger:
    def __init__(self, base_dir: str = "runs") -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / ts
        self.images_dir = self.run_dir / "images"
        self.steps_path = self.run_dir / "steps.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def save_attempt(
        self,
        *,
        task_name: str,
        subtask: SubtaskSpec,
        attempt_index: int,
        frames_before: list[np.ndarray],
        frames_after: list[np.ndarray],
        execution_report: dict[str, Any],
        verifier_result: VerifyResult,
        started_at: str,
        finished_at: str,
    ) -> dict[str, Any]:
        before_path = self._save_attempt_frame(
            frame=frames_before[-1] if frames_before else None,
            subtask_name=subtask.name,
            attempt_index=attempt_index,
            label="a",
        )
        after_path = self._save_attempt_frame(
            frame=frames_after[-1] if frames_after else None,
            subtask_name=subtask.name,
            attempt_index=attempt_index,
            label="b",
        )

        record = {
            "timestamp_start": started_at,
            "timestamp_end": finished_at,
            "task_name": task_name,
            "subtask_name": subtask.name,
            "instruction": subtask.instruction,
            "success_criteria": subtask.success_criteria,
            "attempt_index": attempt_index,
            "params": dict(subtask.params),
            "execution_report": self._summarize_execution_report(execution_report),
            "verifier_result": asdict(verifier_result),
            "image_paths": {
                "before": [before_path] if before_path else [],
                "after": [after_path] if after_path else [],
            },
        }
        self._append_jsonl(record)
        return record

    def _save_attempt_frame(
        self,
        *,
        frame: np.ndarray | None,
        subtask_name: str,
        attempt_index: int,
        label: str,
    ) -> str | None:
        if frame is None:
            return None
        subtask_dir = self.images_dir / subtask_name
        subtask_dir.mkdir(parents=True, exist_ok=True)
        path = subtask_dir / f"attempt_{attempt_index}_{label}.png"
        img = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        img.save(path)
        return str(path)

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        with self.steps_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _summarize_execution_report(report: dict[str, Any]) -> dict[str, Any]:
        telemetry = report.get("telemetry", {})
        arm_pos = telemetry.get("arm_pos", []) if isinstance(telemetry, dict) else []
        time_s = telemetry.get("time_s", []) if isinstance(telemetry, dict) else []
        return {
            "steps": int(report.get("steps", 0)),
            "terminated_reason": report.get("terminated_reason"),
            "agent_action": report.get("agent_action"),
            "agent_reason": report.get("agent_reason"),
            "telemetry_summary": {
                "arm_pos_min": float(min(arm_pos)) if arm_pos else None,
                "arm_pos_max": float(max(arm_pos)) if arm_pos else None,
                "time_start_s": float(time_s[0]) if time_s else None,
                "time_end_s": float(time_s[-1]) if time_s else None,
            },
        }

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import random
from typing import Any

from envs.base import BaseEnv
from langsmith import traceable
from orchestrator.agent import BaseTaskAgent, RuleBasedTaskAgent
from orchestrator.logger import RunLogger
from orchestrator.specs import SubtaskSpec, TaskSpec
from skills.arm_motion import execute_motion_chunk
from verifier.base import BaseVerifier


class Orchestrator:
    def __init__(
        self,
        *,
        env: BaseEnv,
        verifier: BaseVerifier,
        logger: RunLogger,
        control_hz: int,
        task_agent: BaseTaskAgent | None = None,
        verbose: bool = False,
    ) -> None:
        self.env = env
        self.verifier = verifier
        self.logger = logger
        self.control_hz = control_hz
        self.task_agent = task_agent or RuleBasedTaskAgent()
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @traceable(name="orchestrator_run_task", run_type="chain")
    def run_task(self, task: TaskSpec) -> dict[str, Any]:
        self._log(f"[orchestrator] Starting task: {task.name}")
        episode: dict[str, Any] = {
            "task_name": task.name,
            "subtasks": [],
            "run_dir": str(self.logger.run_dir),
            "steps_log": str(self.logger.steps_path),
            "status": "success",
        }

        self.env.reset()
        self._log("[orchestrator] Environment reset complete (arm state set to initial position)")
        total_subtasks = len(task.subtasks)
        for idx, subtask in enumerate(task.subtasks):
            self._log(f"[orchestrator] Subtask {idx + 1}/{total_subtasks}: {subtask.name}")
            subtask_result = self._run_subtask(task.name, subtask)
            episode["subtasks"].append(subtask_result)
            if subtask_result["final_status"] != "success":
                episode["status"] = "fail"
                self._log(f"[orchestrator] Subtask failed: {subtask.name}")
                break
        self._log(f"[orchestrator] Task finished with status: {episode['status']}")

        return episode

    @traceable(name="orchestrator_run_subtask", run_type="chain")
    def _run_subtask(self, task_name: str, subtask: SubtaskSpec) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        final_status = "fail"
        max_attempts = max(1, 1 + int(subtask.max_retries))

        for attempt_idx in range(max_attempts):
            started_at = datetime.now(timezone.utc).isoformat()
            frames_before = self.env.get_recent_frames(1)
            obs_before = self.env.get_observation()
            decision = self.task_agent.decide(
                subtask=subtask,
                attempt_index=attempt_idx,
                previous_attempts=attempts,
            )
            self._log(
                f"[orchestrator] {subtask.name} attempt {attempt_idx + 1}/{max_attempts}: "
                f"action={decision.action}"
            )

            execution_report = self._execute_subtask(subtask, decision.action)
            execution_report["agent_action"] = decision.action
            execution_report["agent_reason"] = decision.reason

            obs_after = self.env.get_observation()
            frames_after = self.env.get_recent_frames(1)
            verifier_result = self.verifier.verify(
                frames_before=frames_before,
                frames_after=frames_after,
                subtask=subtask,
                obs_before=obs_before,
                obs_after=obs_after,
            )
            finished_at = datetime.now(timezone.utc).isoformat()

            log_record = self.logger.save_attempt(
                task_name=task_name,
                subtask=subtask,
                attempt_index=attempt_idx,
                frames_before=frames_before,
                frames_after=frames_after,
                execution_report=execution_report,
                verifier_result=verifier_result,
                started_at=started_at,
                finished_at=finished_at,
            )

            attempt = {
                "attempt_index": attempt_idx,
                "started_at": started_at,
                "finished_at": finished_at,
                "params": dict(subtask.params),
                "agent_action": decision.action,
                "agent_reason": decision.reason,
                "execution_report": execution_report,
                "verifier": asdict(verifier_result),
                "artifact_paths": log_record["image_paths"],
            }
            attempts.append(attempt)

            if verifier_result.status == "success":
                final_status = "success"
                self._log(f"[orchestrator] {subtask.name} success on attempt {attempt_idx + 1}")
                break

            if verifier_result.adjustment:
                self._apply_adjustment_with_jitter(subtask, verifier_result.adjustment)
                self._log(
                    f"[orchestrator] {subtask.name} not complete "
                    f"(status={verifier_result.status}); retrying with adjustment"
                )

        return {
            "subtask_name": subtask.name,
            "instruction": subtask.instruction,
            "success_criteria": subtask.success_criteria,
            "attempts": attempts,
            "final_status": final_status,
        }

    def _execute_subtask(self, subtask: SubtaskSpec, action: str) -> dict[str, Any]:
        params = dict(subtask.params)
        params["subtask_name"] = subtask.name
        params["max_attempt_seconds"] = subtask.max_attempt_seconds
        if action == "move_left":
            params["target"] = "left"
        elif action == "move_right":
            params["target"] = "right"
        else:
            raise ValueError(f"unsupported agent action: {action}")

        if "move" in subtask.name and ("right" in subtask.name or "left" in subtask.name):
            return execute_motion_chunk(
                env=self.env,
                params=params,
                control_hz=self.control_hz,
            )

        raise ValueError(f"unsupported subtask name: {subtask.name}")

    def _apply_adjustment_with_jitter(self, subtask: SubtaskSpec, adjustment: dict[str, Any]) -> None:
        updated = dict(adjustment)
        # Add small randomness so retries do not follow a perfectly fixed schedule.
        if "speed" in updated and isinstance(updated["speed"], (int, float)):
            jittered = float(updated["speed"]) * random.uniform(0.95, 1.05)
            updated["speed"] = max(0.05, min(1.2, jittered))
        if "chunk_duration_s" in updated and isinstance(updated["chunk_duration_s"], (int, float)):
            jittered = float(updated["chunk_duration_s"]) * random.uniform(0.92, 1.08)
            updated["chunk_duration_s"] = max(0.1, min(0.8, jittered))
        subtask.params.update(updated)

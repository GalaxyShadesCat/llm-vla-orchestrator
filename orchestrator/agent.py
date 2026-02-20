from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import AzureOpenAI, OpenAI

from orchestrator.specs import SubtaskSpec


@dataclass
class ToolDecision:
    action: str
    reason: str


class BaseTaskAgent:
    """Selects one motion tool per loop iteration (ReAct action choice)."""

    def decide(
        self,
        *,
        subtask: SubtaskSpec,
        attempt_index: int,
        previous_attempts: list[dict[str, Any]],
    ) -> ToolDecision:
        raise NotImplementedError


class RuleBasedTaskAgent(BaseTaskAgent):
    @traceable(name="react_agent_decide_rule_based", run_type="chain")
    def decide(
        self,
        *,
        subtask: SubtaskSpec,
        attempt_index: int,
        previous_attempts: list[dict[str, Any]],
    ) -> ToolDecision:
        del attempt_index, previous_attempts
        target = str(subtask.params.get("target", "")).lower()
        if target == "left":
            return ToolDecision(action="move_left", reason="Using target=left from subtask params")
        return ToolDecision(action="move_right", reason="Using target=right from subtask params")


class AzureReActTaskAgent(BaseTaskAgent):
    def __init__(
        self,
        *,
        deployment: str,
        api_key: str | None = None,
        api_version: str | None = None,
        endpoint: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        resolved_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        resolved_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        resolved_api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")

        if not resolved_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required for azure_react agent")
        if not resolved_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for azure_react agent")

        if resolved_api_version:
            client = AzureOpenAI(
                api_key=resolved_api_key,
                azure_endpoint=resolved_endpoint,
                api_version=resolved_api_version,
                timeout=timeout_s,
            )
        else:
            client = OpenAI(
                api_key=resolved_api_key,
                base_url=self._azure_v1_base_url(resolved_endpoint),
                timeout=timeout_s,
            )

        self.client = wrap_openai(client)
        self.deployment = deployment

    @traceable(name="react_agent_decide_azure", run_type="chain")
    def decide(
        self,
        *,
        subtask: SubtaskSpec,
        attempt_index: int,
        previous_attempts: list[dict[str, Any]],
    ) -> ToolDecision:
        history = [
            {
                "attempt_index": a.get("attempt_index"),
                "verifier_status": (a.get("verifier") or {}).get("status"),
                "verifier_notes": (a.get("verifier") or {}).get("notes"),
                "last_action": a.get("agent_action"),
            }
            for a in previous_attempts[-4:]
        ]

        system_prompt = (
            "You are a ReAct-style orchestration agent. "
            "You may choose exactly one tool action from: move_left, move_right. "
            "Pick the action that best advances the current subtask instruction. "
            "Return JSON only with keys: action, reason."
        )
        user_prompt = {
            "subtask": {
                "name": subtask.name,
                "instruction": subtask.instruction,
                "success_criteria": subtask.success_criteria,
                "params": subtask.params,
            },
            "attempt_index": attempt_index,
            "recent_history": history,
            "allowed_actions": ["move_left", "move_right"],
            "task_order_note": "Subtasks are executed sequentially by orchestrator; focus only on current subtask.",
        }

        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, sort_keys=True)},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        action, reason = self._parse_decision(raw)
        return ToolDecision(action=action, reason=reason)

    @staticmethod
    def _parse_decision(raw: str) -> tuple[str, str]:
        obj = json.loads(raw)
        action = str(obj.get("action", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip() or "No reason provided"
        if action not in {"move_left", "move_right"}:
            raise ValueError(f"Invalid action from ReAct agent: {action!r}")
        return action, reason

    @staticmethod
    def _azure_v1_base_url(endpoint: str) -> str:
        trimmed = endpoint.rstrip("/")
        if trimmed.endswith("/openai/v1"):
            return f"{trimmed}/"
        return f"{trimmed}/openai/v1/"

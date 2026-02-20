from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

from langsmith import traceable
from langsmith.wrappers import wrap_openai
import numpy as np
from openai import AzureOpenAI, OpenAI
from PIL import Image

from orchestrator.specs import SubtaskSpec, VerifyResult
from verifier.base import BaseVerifier
from verifier.schemas import validate_verifier_output


class AzureChatGPTVisionVerifier(BaseVerifier):
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
            raise ValueError("AZURE_OPENAI_API_KEY is required for azure_chatgpt verifier")
        if not resolved_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for azure_chatgpt verifier")

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

    @traceable(name="vision_verify_azure_chatgpt", run_type="chain")
    def verify(
        self,
        frames_before: list[np.ndarray],
        frames_after: list[np.ndarray],
        subtask: SubtaskSpec,
        obs_before: dict[str, Any] | None = None,
        obs_after: dict[str, Any] | None = None,
    ) -> VerifyResult:
        del obs_before, obs_after

        if not frames_before or not frames_after:
            return validate_verifier_output(
                {
                    "status": "uncertain",
                    "confidence": 0.2,
                    "failure_mode": "missing_frames",
                    "adjustment": {"chunk_duration_s": 0.35},
                    "notes": "Missing before/after frame for verification.",
                }
            )

        before_frame = frames_before[-1]
        after_frame = frames_after[-1]

        system_prompt = self._build_system_prompt()
        user_content = [
            {
                "type": "text",
                "text": self._build_user_prompt(subtask),
            },
            {
                "type": "text",
                "text": "Image label: BEFORE (before.png).",
            },
            {
                "type": "image_url",
                "image_url": {"url": self._frame_to_data_url(before_frame)},
            },
            {
                "type": "text",
                "text": "Image label: AFTER (after.png).",
            },
            {
                "type": "image_url",
                "image_url": {"url": self._frame_to_data_url(after_frame)},
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                temperature=0.0,
                max_tokens=220,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            raw = response.choices[0].message.content or ""
            obj = self._parse_json_object(raw)
            return validate_verifier_output(obj)
        except Exception as exc:
            return validate_verifier_output(
                {
                    "status": "uncertain",
                    "confidence": 0.25,
                    "failure_mode": "verifier_api_error",
                    "adjustment": {"chunk_duration_s": 0.35},
                    "notes": f"Azure verifier error: {type(exc).__name__}: {exc}",
                }
            )

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are a strict vision verifier in a simulated robot loop. "
            "Scene semantics: black background; white vertical center line (goal boundary); "
            "green rectangle is the arm marker. "
            "Task semantics: compare BEFORE vs AFTER images after one motion chunk. "
            "Return JSON only with fields: status, confidence, failure_mode, adjustment, notes. "
            "status must be one of success/fail/uncertain. "
            "If target direction is right, success means marker is clearly to the right of the white line. "
            "If target direction is left, success means marker is clearly to the left of the white line. "
            "If fail, propose small adjustment values for speed and/or chunk_duration_s."
        )

    @staticmethod
    def _build_user_prompt(subtask: SubtaskSpec) -> str:
        return (
            f"Subtask: {subtask.name}\n"
            f"Instruction: {subtask.instruction}\n"
            f"Success criteria: {subtask.success_criteria}\n"
            f"Parameters: {json.dumps(subtask.params, sort_keys=True)}\n"
            "You are simulating multimodal-LLM verification over a VLA action chunk. "
            "The BEFORE image is pre-action; AFTER image is post-action. "
            "Decide whether the crossing objective is complete now."
        )

    @staticmethod
    def _frame_to_data_url(frame: np.ndarray) -> str:
        img = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{data}"

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if not text:
            raise ValueError("empty verifier response")

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError(f"could not parse JSON object from: {text[:160]}")

    @staticmethod
    def _azure_v1_base_url(endpoint: str) -> str:
        trimmed = endpoint.rstrip("/")
        if trimmed.endswith("/openai/v1"):
            return f"{trimmed}/"
        return f"{trimmed}/openai/v1/"

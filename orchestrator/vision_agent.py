"""Vision recognisers and model resolution for chess move extraction from board images."""

from __future__ import annotations

import base64
import json
import mimetypes
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import chess
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langsmith import traceable
from openai import OpenAI
from pydantic import BaseModel

from orchestrator.chess_types import VisionMoveState


class _VisionOutput(BaseModel):
    """Structured output schema for vision SAN move parsing."""

    after_piece_placement: str
    move_san: str
    overall_confidence: Optional[float] = None


class VisionMoveRecognizer(Protocol):
    """Protocol for vision recognisers that infer move state from an image."""

    def recognise_move(
        self,
        *,
        image_path: str,
        before_fen: str,
        feedback: str | None = None,
    ) -> tuple[VisionMoveState, int]:
        """Recognise the move and return normalised state plus attempt count."""


@dataclass
class VisionProviderSettings:
    """Shared provider credentials and behaviour flags for vision model routing."""

    azure_api_key: str = ""
    azure_base_url: str = ""
    azure_api_version: str = ""
    azure_endpoint: str = ""
    azure_max_retries: int = 2
    azure_timeout_s: float = 120.0
    hugging_face_api_key: str = ""
    hugging_face_max_retries: int = 3
    hugging_face_timeout_s: float = 90.0
    azure_model_allowlist: set[str] = field(default_factory=set)
    azure_model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


class ChatGPTVisionRecognizer:
    """Uses Azure/OpenAI Chat models to extract a SAN move in strict JSON output."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        api_version: str | None = None,
        azure_endpoint: str | None = None,
        max_retries: int = 2,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model.strip()
        self.max_retries = max(1, int(max_retries))
        self.timeout_s = max(1.0, float(timeout_s))
        if not self.model:
            raise ValueError("ChatGPTVisionRecognizer requires a non-empty model")
        if not api_key.strip():
            raise ValueError("ChatGPTVisionRecognizer requires a non-empty api_key")
        api_version_value = str(api_version or "").strip()
        azure_endpoint_value = str(azure_endpoint or "").strip()

        if api_version_value:
            if not azure_endpoint_value:
                raise ValueError(
                    "ChatGPTVisionRecognizer requires azure_endpoint when api_version is set"
                )
            self.llm = AzureChatOpenAI(
                azure_deployment=self.model,
                api_key=api_key,
                api_version=api_version_value,
                azure_endpoint=azure_endpoint_value,
                timeout=self.timeout_s,
                max_retries=0,
            )
        else:
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=api_key,
                base_url=base_url,
                timeout=self.timeout_s,
                max_retries=0,
            )

    @traceable(name="chess_pipeline_vision_infer_san", run_type="tool")
    def recognise_move(
        self,
        *,
        image_path: str,
        before_fen: str,
        feedback: str | None = None,
    ) -> tuple[VisionMoveState, int]:
        raw_bytes = Path(image_path).read_bytes()

        attempts = 0
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            attempts = attempt + 1
            try:
                messages = _build_messages(
                    before_fen=before_fen,
                    data_url=self._to_data_url(raw_bytes, image_path),
                    feedback=feedback,
                )
                llm_response = self.llm.invoke(messages)
                raw_text = str(getattr(llm_response, "content", "")).strip()
                return _normalise_vision_output(raw_text), attempts
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        raise RuntimeError(
            f"Vision recognition failed after {attempts} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _to_data_url(raw: bytes, image_path: str) -> str:
        encoded = base64.b64encode(raw).decode("ascii")
        mime, _ = mimetypes.guess_type(str(image_path))
        content_type = mime if mime and mime.startswith("image/") else "image/png"
        return f"data:{content_type};base64,{encoded}"


class HuggingFaceVisionRecognizer:
    """Uses Hugging Face chat completion VLMs with retry/backoff for robust inference."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        max_retries: int = 3,
        timeout_s: float = 90.0,
    ) -> None:
        self.model = str(model or "").strip()
        if not self.model:
            raise ValueError("HuggingFaceVisionRecognizer requires a non-empty model")
        token = str(api_key or "").strip()
        if not token:
            raise ValueError("HuggingFaceVisionRecognizer requires a non-empty api_key")
        self.max_retries = max(1, int(max_retries))
        self.timeout_s = max(1.0, float(timeout_s))
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
            timeout=self.timeout_s,
            max_retries=0,
        )

    @traceable(name="chess_pipeline_hf_vision_infer_san", run_type="tool")
    def recognise_move(
        self,
        *,
        image_path: str,
        before_fen: str,
        feedback: str | None = None,
    ) -> tuple[VisionMoveState, int]:
        raw_bytes = Path(image_path).read_bytes()
        image_data_url = _to_data_url(raw_bytes=raw_bytes, image_path=image_path)

        attempts = 0
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            attempts = attempt + 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=_messages_to_openai(
                        _build_messages(
                            before_fen=before_fen,
                            data_url=image_data_url,
                            feedback=feedback,
                        )
                    ),
                    response_format={"type": "json_object"},
                    timeout=self.timeout_s,
                )
                raw_text = _extract_message_text(response).strip()
                if not raw_text:
                    # Some provider-routed models return empty content when json_object
                    # formatting is requested. Retry once without response_format.
                    fallback_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=_messages_to_openai(
                            _build_messages(
                                before_fen=before_fen,
                                data_url=image_data_url,
                                feedback=feedback,
                            )
                        ),
                        timeout=self.timeout_s,
                    )
                    raw_text = _extract_message_text(fallback_response).strip()
                return _normalise_vision_output(raw_text), attempts
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                if not _is_retryable_hf_error(exc):
                    break
                delay_s = _retry_delay_s(attempt)
                time.sleep(delay_s)

        raise RuntimeError(
            f"Hugging Face vision recognition failed after {attempts} attempts: {last_error}"
        ) from last_error


class VisionModelResolver:
    """Resolves model names to Azure or Hugging Face recogniser instances with caching."""

    def __init__(
        self,
        *,
        default_model: str,
        settings: VisionProviderSettings,
    ) -> None:
        self.default_model = str(default_model or "").strip()
        self.settings = settings
        self._cache: dict[str, VisionMoveRecognizer] = {}

    def resolve(self, requested_model: str | None = None) -> tuple[str, VisionMoveRecognizer]:
        """Resolve the requested model name and return the concrete recogniser."""
        model = str(requested_model or "").strip() or self.default_model
        if not model:
            raise ValueError("No vision model was requested and no default model is configured")
        if model in self._cache:
            return model, self._cache[model]

        recogniser = self._build_recogniser(model)
        self._cache[model] = recogniser
        return model, recogniser

    def _build_recogniser(self, model: str) -> VisionMoveRecognizer:
        if self._is_azure_model(model):
            override = self.settings.azure_model_overrides.get(model, {})
            api_key = str(override.get("api_key", self.settings.azure_api_key)).strip()
            base_url = str(override.get("base_url", self.settings.azure_base_url)).strip()
            api_version = str(override.get("api_version", self.settings.azure_api_version)).strip()
            azure_endpoint = str(
                override.get("azure_endpoint", self.settings.azure_endpoint)
            ).strip()
            timeout_s = float(override.get("timeout_s", self.settings.azure_timeout_s))
            max_retries = int(override.get("max_retries", self.settings.azure_max_retries))

            if not api_key:
                raise ValueError(
                    f"Model '{model}' is configured as Azure/OpenAI, but AZURE_VISION_API_KEY is missing"
                )
            return ChatGPTVisionRecognizer(
                model=model,
                api_key=api_key,
                base_url=base_url or None,
                api_version=api_version or None,
                azure_endpoint=azure_endpoint or None,
                max_retries=max_retries,
                timeout_s=timeout_s,
            )

        if not self.settings.hugging_face_api_key.strip():
            raise ValueError(
                f"Model '{model}' is configured as Hugging Face, but HUGGING_FACE_API_KEY is missing"
            )
        return HuggingFaceVisionRecognizer(
            model=model,
            api_key=self.settings.hugging_face_api_key,
            max_retries=self.settings.hugging_face_max_retries,
            timeout_s=self.settings.hugging_face_timeout_s,
        )

    def _is_azure_model(self, model: str) -> bool:
        allowlist = {item.strip() for item in self.settings.azure_model_allowlist if item.strip()}
        if model in allowlist:
            return True
        return model.lower().startswith("gpt-")


def _build_messages(*, before_fen: str, data_url: str, feedback: str | None) -> list[Any]:
    return [
        SystemMessage(
            content=(
                "You are a strict chessboard parser. You are given the board state before a "
                "move and a top-down image of the board after that move.\n\n"
                "Step 1: infer the full after-position from the image.\n"
                "Step 2: compare the before-position and after-position and determine the "
                "single legal move.\n\n"
                "Return strict JSON with keys:\n"
                "- after_piece_placement\n"
                "- move_san\n"
                "- overall_confidence\n\n"
                "after_piece_placement must be valid board-FEN placement only.\n"
                "move_san must be standard algebraic notation only.\n"
                "overall_confidence must be a number between 0 and 1 or null.\n"
                "Do not include explanations."
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Task: identify the move in algebraic notation (SAN). "
                        f"before_fen: {before_fen}"
                    ),
                },
                *(
                    [{"type": "text", "text": f"Feedback from previous attempt: {feedback}"}]
                    if feedback
                    else []
                ),
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        ),
    ]


def _messages_to_openai(messages: list[Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = "user"
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        converted.append({"role": role, "content": message.content})
    return converted


def _normalise_vision_output(raw_text: str) -> VisionMoveState:
    payload = _VisionOutput.model_validate(_normalise_payload(_parse_json_object(raw_text)))
    raw_output = payload.model_dump_json()
    after_piece_placement = _normalise_piece_placement(str(payload.after_piece_placement).strip())
    return VisionMoveState(
        after_piece_placement=after_piece_placement,
        move_san=str(payload.move_san).strip(),
        overall_confidence=(
            float(payload.overall_confidence)
            if payload.overall_confidence is not None
            else None
        ),
        raw_model_output=raw_output,
    )


def _to_data_url(*, raw_bytes: bytes, image_path: str) -> str:
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    mime, _ = mimetypes.guess_type(str(image_path))
    content_type = mime if mime and mime.startswith("image/") else "image/png"
    return f"data:{content_type};base64,{encoded}"


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise ValueError("empty vision response")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Could not parse JSON object from vision response")


def _normalise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalised = dict(payload)

    after_piece_placement = normalised.get("after_piece_placement")
    if not isinstance(after_piece_placement, str):
        for alias in ("piece_placement", "after_fen", "post_piece_placement"):
            alias_value = normalised.get(alias)
            if isinstance(alias_value, str) and alias_value.strip():
                after_piece_placement = alias_value
                break
    normalised["after_piece_placement"] = str(after_piece_placement or "").strip()

    move_san = normalised.get("move_san")
    if not isinstance(move_san, str):
        for alias in ("san", "move", "move_notation", "move_algebraic"):
            alias_value = normalised.get(alias)
            if isinstance(alias_value, str) and alias_value.strip():
                move_san = alias_value
                break
    normalised["move_san"] = str(move_san or "").strip()

    overall_confidence = normalised.get("overall_confidence")
    if isinstance(overall_confidence, str):
        aliases = {
            "very_low": 0.1,
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "very_high": 0.9,
        }
        mapped = aliases.get(overall_confidence.strip().lower().replace(" ", "_"))
        normalised["overall_confidence"] = mapped
    elif overall_confidence is None:
        normalised["overall_confidence"] = None
    else:
        try:
            normalised["overall_confidence"] = float(overall_confidence)
        except (TypeError, ValueError):
            normalised["overall_confidence"] = None

    return normalised


def _normalise_piece_placement(raw_value: str) -> str:
    value = str(raw_value).strip()
    if not value:
        raise ValueError("empty after_piece_placement in vision response")
    if " " in value:
        return chess.Board(value).board_fen()
    chess.Board(f"{value} w - - 0 1")
    return value


def _is_retryable_hf_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "overloaded",
            "timed out",
            "timeout",
            "model is loading",
            "empty vision response",
            "empty response",
        )
    )


def _retry_delay_s(attempt_index: int) -> float:
    base = 1.0
    cap = 18.0
    exp = min(cap, base * (2 ** attempt_index))
    return exp + random.uniform(0.0, 0.4)


def _extract_message_text(response: Any) -> str:
    """Extract textual assistant content from OpenAI-compatible chat responses."""
    try:
        message = response.choices[0].message
    except Exception:  # noqa: BLE001
        return ""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")

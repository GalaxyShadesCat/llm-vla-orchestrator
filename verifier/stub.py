from __future__ import annotations

from typing import Any

from langsmith import traceable
import numpy as np

from orchestrator.specs import SubtaskSpec, VerifyResult
from verifier.base import BaseVerifier
from verifier.schemas import validate_verifier_output


class StubVerifier(BaseVerifier):
    """Deterministic vision verifier with the same contract as a future GPT-vision verifier."""

    def __init__(self, crossing_margin_px: int = 4) -> None:
        self.crossing_margin_px = crossing_margin_px

    @traceable(name="vision_verify_stub", run_type="chain")
    def verify(
        self,
        frames_before: list[np.ndarray],
        frames_after: list[np.ndarray],
        subtask: SubtaskSpec,
        obs_before: dict[str, Any] | None = None,
        obs_after: dict[str, Any] | None = None,
    ) -> VerifyResult:
        del frames_before, obs_before, obs_after

        if not frames_after:
            return validate_verifier_output(
                {
                    "status": "uncertain",
                    "confidence": 0.2,
                    "failure_mode": "missing_frames",
                    "adjustment": {"chunk_duration_s": 0.45},
                    "notes": "No post-execution frame provided.",
                }
            )

        frame = frames_after[-1]
        h, w, _ = frame.shape
        line_x = w // 2
        marker_x = self._extract_marker_x(frame)

        target = str(subtask.params.get("target", "")).lower()
        if not target:
            target = "right" if "right" in subtask.name else "left"

        crossed = False
        if target == "right":
            crossed = marker_x > (line_x + self.crossing_margin_px)
        elif target == "left":
            crossed = marker_x < (line_x - self.crossing_margin_px)

        if crossed:
            return validate_verifier_output(
                {
                    "status": "success",
                    "confidence": 0.92,
                    "failure_mode": None,
                    "adjustment": None,
                    "notes": f"Marker crossed line to the {target}. marker_x={marker_x}, line_x={line_x}.",
                }
            )

        speed = float(subtask.params.get("speed", 0.35))
        chunk_duration_s = float(subtask.params.get("chunk_duration_s", 0.35))
        return validate_verifier_output(
            {
                "status": "fail",
                "confidence": 0.78,
                "failure_mode": "not_crossed_line",
                "adjustment": {
                    "speed": min(1.2, speed + 0.08),
                    "chunk_duration_s": min(0.8, chunk_duration_s + 0.05),
                },
                "notes": f"Still not across line. marker_x={marker_x}, line_x={line_x}, target={target}.",
            }
        )

    @staticmethod
    def _extract_marker_x(frame: np.ndarray) -> int:
        # Isolate green marker from white line by penalizing red/blue channels.
        r = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        b = frame[:, :, 2].astype(np.float32)
        marker_score = g - 0.5 * r - 0.5 * b
        col_scores = marker_score.mean(axis=0)
        return int(np.argmax(col_scores))

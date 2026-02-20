from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from orchestrator.specs import SubtaskSpec, VerifyResult


class BaseVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        frames_before: list[np.ndarray],
        frames_after: list[np.ndarray],
        subtask: SubtaskSpec,
        obs_before: dict[str, Any] | None = None,
        obs_after: dict[str, Any] | None = None,
    ) -> VerifyResult:
        raise NotImplementedError

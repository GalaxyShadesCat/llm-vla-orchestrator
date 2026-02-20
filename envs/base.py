from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEnv(ABC):
    """Environment interface with RGB observations as numpy arrays (H, W, 3), uint8."""

    @abstractmethod
    def reset(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_recent_frames(self, n: int) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def safety_check(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

from __future__ import annotations

from collections import deque

import numpy as np

from envs.base import BaseEnv


class MockEnv(BaseEnv):
    def __init__(
        self,
        *,
        control_hz: int,
        arm_limit: float = 1.0,
        frame_height: int = 96,
        frame_width: int = 96,
        frame_buffer_size: int = 400,
    ) -> None:
        self.control_hz = control_hz
        self.dt = 1.0 / float(control_hz)
        self.arm_limit = float(arm_limit)
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.arm_pos = -0.6
        self.sim_time = 0.0
        self.last_action: dict[str, float] = {"dx": 0.0}
        self._frames: deque[np.ndarray] = deque(maxlen=frame_buffer_size)

    def reset(self) -> dict:
        self.arm_pos = -0.6
        self.sim_time = 0.0
        self.last_action = {"dx": 0.0}
        self._frames.clear()
        for _ in range(3):
            self._frames.append(self._render_frame())
        return self.get_observation()

    def get_observation(self) -> dict:
        frame = self._render_frame()
        self._frames.append(frame)
        return {
            "rgb": frame,
            "arm_pos": self.arm_pos,
            "time_s": self.sim_time,
            "last_action": dict(self.last_action),
        }

    def step(self, action: dict) -> dict:
        dx_cmd = float(action.get("dx", 0.0))
        # Simulate imperfect low-level execution from a VLA policy.
        dx_applied = 1.0 * dx_cmd
        self.last_action = {"dx": dx_applied}
        self.arm_pos += dx_applied * self.dt
        self.arm_pos = max(-self.arm_limit, min(self.arm_limit, self.arm_pos))
        self.sim_time += self.dt
        return self.get_observation()

    def get_recent_frames(self, n: int) -> list[np.ndarray]:
        if n <= 0:
            return []
        return [frame.copy() for frame in list(self._frames)[-n:]]

    def safety_check(self) -> bool:
        return abs(self.arm_pos) <= self.arm_limit

    def close(self) -> None:
        self._frames.clear()

    def _arm_x(self) -> int:
        margin = 8
        x_min = margin
        x_max = self.frame_width - margin - 1
        norm = (self.arm_pos + self.arm_limit) / (2.0 * self.arm_limit)
        return int(round(x_min + norm * (x_max - x_min)))

    def _render_frame(self) -> np.ndarray:
        frame = np.full((self.frame_height, self.frame_width, 3), 18, dtype=np.uint8)

        line_x = self.frame_width // 2
        frame[:, line_x - 1 : line_x + 1, :] = 255

        x = self._arm_x()
        y_mid = self.frame_height // 2
        y0 = max(0, y_mid - 6)
        y1 = min(self.frame_height, y_mid + 7)
        x0 = max(0, x - 3)
        x1 = min(self.frame_width, x + 4)
        frame[y0:y1, x0:x1] = np.array([30, 220, 30], dtype=np.uint8)

        return frame

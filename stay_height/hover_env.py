"""Task 1：定高悬停环境"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gym_pybullet_drones.envs.HoverAviary import HoverAviary


@dataclass
class HoverHoldConfig:
    target_pos: np.ndarray = None
    max_steps: int = 500
    stack_size: int = 4
    action_scale: float = 0.50
    ema_alpha: float = 0.3

    r_step: float = -0.01
    r_height_k: float = 5.0
    r_height_scale: float = 0.2
    r_att_penalty: float = -0.1
    r_smooth_penalty: float = -0.02
    r_crash: float = -10.0
    r_deviate: float = -10.0
    r_success: float = 10.0

    max_z_err: float = 0.5
    max_rp_angle: float = 0.4
    success_steps_req: int = 300
    success_z_tol: float = 0.1

    reward_clip_lo: float = -20.0
    reward_clip_hi: float = 150.0

    def __post_init__(self):
        if self.target_pos is None:
            self.target_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def make_aviary() -> HoverAviary:
    return HoverAviary(
        gui=False,
        record=False,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
    )


class HoverHoldEnv(gym.Wrapper):
    """11 维单帧 × 4 帧堆叠 = 44 维观测。"""

    def __init__(self, env: HoverAviary, cfg: HoverHoldConfig | None = None):
        super().__init__(env)
        self.cfg = cfg or HoverHoldConfig()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        obs_dim = 11 * self.cfg.stack_size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._frames: collections.deque[np.ndarray] = collections.deque(maxlen=self.cfg.stack_size)
        self._ema = np.zeros(4, dtype=np.float32)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._step_id = 0
        self._stable_count = 0

    def _frame(self) -> np.ndarray:
        pos = self.unwrapped.pos[0]
        rpy = self.unwrapped.rpy[0]
        dz = pos[2] - self.cfg.target_pos[2]
        return np.concatenate([pos, rpy, self._ema, [dz]]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._step_id = 0
        self._stable_count = 0
        self._ema.fill(0.0)
        self._prev_action.fill(0.0)
        frame = self._frame()
        self._frames.clear()
        for _ in range(self.cfg.stack_size):
            self._frames.append(frame.copy())
        return np.concatenate(list(self._frames)), info

    def step(self, action: np.ndarray):
        self._step_id += 1
        action_nn = np.clip(action, -1.0, 1.0).astype(np.float32)
        target = action_nn * self.cfg.action_scale
        self._ema = self.cfg.ema_alpha * target + (1.0 - self.cfg.ema_alpha) * self._ema

        self.env.step(np.expand_dims(self._ema, axis=0))
        frame = self._frame()
        self._frames.append(frame)

        pos, rpy, dz = frame[0:3], frame[3:6], float(frame[10])
        r_step = self.cfg.r_step
        r_height = self.cfg.r_height_scale * np.exp(-self.cfg.r_height_k * dz * dz)
        r_att = self.cfg.r_att_penalty * (abs(float(rpy[0])) + abs(float(rpy[1])))
        r_smooth = self.cfg.r_smooth_penalty * float(np.sum((action_nn - self._prev_action) ** 2))
        raw = float(r_step + r_height + r_att + r_smooth)

        terminated, truncated = False, False
        if self._step_id >= self.cfg.max_steps:
            truncated = True
        if pos[2] < 0.1 or max(abs(float(rpy[0])), abs(float(rpy[1]))) > self.cfg.max_rp_angle:
            raw, terminated = self.cfg.r_crash, True
        elif abs(dz) > self.cfg.max_z_err:
            raw, terminated = self.cfg.r_deviate, True
        elif abs(dz) <= self.cfg.success_z_tol:
            self._stable_count += 1
            if self._stable_count >= self.cfg.success_steps_req:
                bonus = (self.cfg.max_steps - self._step_id) * self.cfg.r_height_scale
                raw, terminated = self.cfg.r_success + bonus, True
        else:
            self._stable_count = 0

        reward = float(np.clip(raw, self.cfg.reward_clip_lo, self.cfg.reward_clip_hi))
        self._prev_action = action_nn.copy()
        info = {
            "hover_stats": {
                "r_raw": raw,
                "r_clip": reward,
                "pos_z": float(pos[2]),
                "dz": dz,
                "stable_count": self._stable_count,
            }
        }
        return np.concatenate(list(self._frames)), reward, terminated, truncated, info

"""Task 2：3D 动态轨迹追踪环境"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.HoverAviary import HoverAviary


@dataclass
class PathTrackConfig:
    ctrl_freq: int = 240
    rl_freq: int = 48
    substeps: int = 5
    rl_dt: float = 1.0 / 48
    max_steps: int = 800

    stack_size: int = 4
    lookahead_steps: int = 5
    lookahead_interval: int = 10
    obs_per_frame: int = 25

    ema_alpha: float = 0.5
    action_scale: float = 0.5

    r_survival: float = 0.1
    r_track_sigma: float = 1.0
    r_vel_coef: float = 0.15
    r_heading_coef: float = 0.05
    r_smooth_coef: float = -0.05

    r_crash: float = -20.0
    r_deviate: float = -20.0
    r_success_base: float = 50.0
    time_bonus_coef: float = 0.1

    max_dev_err: float = 2.0
    max_rp_angle: float = 1.05


def make_aviary() -> HoverAviary:
    return HoverAviary(gui=False, record=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))


class SinusoidPath:
    """生成平滑 3D 正弦轨迹及切线。"""

    def __init__(self, num_points: int, dt: float):
        ax, ay = np.random.uniform(1.5, 3.0, 2)
        az = np.random.uniform(0.1, 0.4)
        fx, fy, fz = np.random.uniform([0.02, 0.02, 0.01], [0.08, 0.08, 0.05])
        px, py, pz = np.random.uniform(0, 2 * math.pi, 3)

        points, tangents = [], []
        for i in range(num_points):
            t = i * dt
            pt = np.array([
                ax * math.sin(2 * math.pi * fx * t + px),
                ay * math.sin(2 * math.pi * fy * t + py),
                1.2 + az * math.sin(2 * math.pi * fz * t + pz),
            ], dtype=np.float32)
            points.append(pt)
            if i > 0:
                d = points[i] - points[i - 1]
                n = np.linalg.norm(d)
                tangents.append(d / n if n > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32))
        tangents.insert(0, tangents[0].copy())
        self.points = np.asarray(points, dtype=np.float32)
        self.tangents = np.asarray(tangents, dtype=np.float32)


class PathTrackEnv(gym.Wrapper):
    def __init__(self, env: HoverAviary, cfg: PathTrackConfig | None = None):
        super().__init__(env)
        self.cfg = cfg or PathTrackConfig()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        obs_dim = self.cfg.obs_per_frame * self.cfg.stack_size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._frames: collections.deque[np.ndarray] = collections.deque(maxlen=self.cfg.stack_size)
        self._ema = np.zeros(4, dtype=np.float32)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._step_id = 0
        self._path_idx = 0
        self._path: SinusoidPath | None = None

    def _kinematics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        uid = self.unwrapped.DRONE_IDS[0]
        pos, quat = p.getBasePositionAndOrientation(uid, physicsClientId=self.unwrapped.CLIENT)
        vel, _ = p.getBaseVelocity(uid, physicsClientId=self.unwrapped.CLIENT)
        pos = np.array(pos, dtype=np.float32)
        vel = np.array(vel, dtype=np.float32)
        rpy = np.array(p.getEulerFromQuaternion(quat), dtype=np.float32)
        rot = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float64).reshape(3, 3)
        heading = rot[:, 0].astype(np.float32)
        return pos, vel, rpy, heading

    def _advance_path_index(self, pos: np.ndarray) -> None:
        end = min(self._path_idx + 30, len(self._path.points))
        local = self._path.points[self._path_idx:end] - pos
        self._path_idx += int(np.argmin(np.linalg.norm(local, axis=1)))

    def _build_frame(self, pos: np.ndarray, rpy: np.ndarray) -> np.ndarray:
        self._advance_path_index(pos)
        rel_now = self._path.points[self._path_idx] - pos
        lookahead: List[np.ndarray] = []
        for k in range(1, self.cfg.lookahead_steps + 1):
            idx = min(self._path_idx + k * self.cfg.lookahead_interval, len(self._path.points) - 1)
            lookahead.append(self._path.points[idx] - pos)
        return np.concatenate([rpy, self._ema, rel_now, np.concatenate(lookahead)]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        n_pts = self.cfg.max_steps * 2
        self._path = SinusoidPath(n_pts, self.cfg.rl_dt)
        self._step_id = 0
        self._path_idx = 0
        self._ema.fill(0.0)
        self._prev_action.fill(0.0)

        start = self._path.points[0]
        uid = self.unwrapped.DRONE_IDS[0]
        p.resetBasePositionAndOrientation(
            uid, start.tolist(), p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.unwrapped.CLIENT
        )
        p.resetBaseVelocity(uid, [0, 0, 0], [0, 0, 0], physicsClientId=self.unwrapped.CLIENT)

        frame = self._build_frame(start, np.zeros(3, dtype=np.float32))
        self._frames.clear()
        for _ in range(self.cfg.stack_size):
            self._frames.append(frame.copy())
        return np.concatenate(list(self._frames)), info

    def step(self, action: np.ndarray):
        action_nn = np.clip(action, -1.0, 1.0).astype(np.float32)
        target = action_nn * self.cfg.action_scale
        info: Dict[str, Any] = {}
        for _ in range(self.cfg.substeps):
            self._ema = self.cfg.ema_alpha * target + (1.0 - self.cfg.ema_alpha) * self._ema
            _, _, _, _, info = self.env.step(np.expand_dims(self._ema, axis=0))

        self._step_id += 1
        pos, vel, rpy, heading = self._kinematics()
        frame = self._build_frame(pos, rpy)
        self._frames.append(frame)

        target_pos = self._path.points[self._path_idx]
        tangent = self._path.tangents[self._path_idx]
        dist_err = float(np.linalg.norm(pos - target_pos))
        dx, dy, dz = pos[0] - target_pos[0], pos[1] - target_pos[1], pos[2] - target_pos[2]
        weighted_err = math.sqrt(dx * dx + dy * dy + (2.5 * dz) ** 2)

        r_surv = self.cfg.r_survival
        r_track = 0.5 * math.exp(-(weighted_err**2) / (2 * self.cfg.r_track_sigma**2)) - 0.1
        v_align = float(np.clip(float(np.dot(vel, tangent)), -5.0, 5.0))
        r_vel = self.cfg.r_vel_coef * v_align
        r_head = self.cfg.r_heading_coef * float(np.dot(heading, tangent))
        r_smooth = self.cfg.r_smooth_coef * float(np.sum((action_nn - self._prev_action) ** 2))
        cont = float(np.clip(r_surv + r_track + r_vel + r_head + r_smooth, -1.0, 1.0))

        terminal, terminated, truncated = 0.0, False, False
        reason = "ALIVE"
        if dist_err > self.cfg.max_dev_err:
            terminal, terminated, reason = self.cfg.r_deviate, True, "DEVIATE"
        elif max(abs(float(rpy[0])), abs(float(rpy[1]))) > self.cfg.max_rp_angle or pos[2] < 0.05:
            terminal, terminated, reason = self.cfg.r_crash, True, "CRASH"
        elif self._path_idx >= len(self._path.points) - self.cfg.lookahead_steps - 1:
            bonus = max(0, self.cfg.max_steps - self._step_id) * self.cfg.time_bonus_coef
            terminal, truncated, reason = self.cfg.r_success_base + bonus, True, "SUCCESS"
        elif self._step_id >= self.cfg.max_steps:
            truncated, reason = True, "TIMEOUT"

        reward = cont + terminal
        self._prev_action = action_nn.copy()
        info["path_stats"] = {
            "reason": reason,
            "dist_err": dist_err,
            "completion_rate": 100.0 * self._path_idx / len(self._path.points),
            "r_total": reward,
            "pos": pos.copy(),
            "target_pos": target_pos.copy(),
        }
        return np.concatenate(list(self._frames)), float(reward), terminated, truncated, info

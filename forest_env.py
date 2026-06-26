"""定高密林穿越 Gym 环境"""

from __future__ import annotations
import collections
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import gymnasium as gym
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from forest_world import ForestMapConfig, ForestWorld

@dataclass
class ForestNavConfig:
    ctrl_freq: int = 240
    rl_freq: int = 48
    substeps: int = 5  # ctrl_freq // rl_freq
    max_steps: int = 1200

    stack_size: int = 3
    obs_dim_per_frame: int = 34

    ema_alpha: float = 0.5
    action_scale: float = 0.50

    r_step: float = -0.03
    r_height_scale: float = 0.25
    r_height_k: float = 5.0
    r_att_penalty: float = -0.1
    r_smooth_penalty: float = -0.002
    r_approach: float = 0.5
    r_dir: float = 0.10
    r_repulsion_max: float = -0.4

    r_crash: float = -50.0
    r_deviate: float = -50.0
    r_success_base: float = 50.0
    time_bonus_coef: float = 0.1

    safe_lidar_dist: float = 0.25
    max_z_err: float = 0.8
    max_rp_angle: float = 1.0
    success_xy_tol: float = 0.4
    success_z_tol: float = 0.4
    target_altitude: float = 1.0


def make_base_aviary() -> HoverAviary:
    """始终使用 DIRECT 模式（无 GUI）。"""
    return HoverAviary(gui=False, record=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))

class ForestNavEnv(gym.Wrapper):
    metadata = {"render_modes": []}

    def __init__(self, env: HoverAviary, cfg: ForestNavConfig | None = None):
        super().__init__(env)
        self.cfg = cfg or ForestNavConfig()
        self.world = ForestWorld(self.unwrapped.CLIENT, ForestMapConfig())

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        obs_dim = self.cfg.obs_dim_per_frame * self.cfg.stack_size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._frame_buf: collections.deque[np.ndarray] = collections.deque(maxlen=self.cfg.stack_size)
        self._ema_action = np.zeros(4, dtype=np.float32)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._goal = np.zeros(3, dtype=np.float32)
        self._step_counter = 0

    def set_curriculum(self, static_count: int, dynamic_count: int, max_goal_dist: float) -> None:
        self.world.set_difficulty(static_count, dynamic_count, max_goal_dist)

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

    def _build_frame(self, pos: np.ndarray, rpy: np.ndarray) -> np.ndarray:
        rel_goal = self._goal - pos
        lidar = self.world.scan_lidar(pos, float(rpy[2]), self.unwrapped.DRONE_IDS[0])
        return np.concatenate([rpy, self._ema_action, rel_goal, lidar]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        start, goal = self.world.reset()
        start[2] = self.cfg.target_altitude
        goal[2] = self.cfg.target_altitude
        self._goal = goal.astype(np.float32)
        self._step_counter = 0
        self._ema_action.fill(0.0)
        self._prev_action.fill(0.0)

        uid = self.unwrapped.DRONE_IDS[0]
        p.resetBasePositionAndOrientation(uid, start.tolist(), p.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.unwrapped.CLIENT)
        p.resetBaseVelocity(uid, [0, 0, 0], [0, 0, 0], physicsClientId=self.unwrapped.CLIENT)

        frame = self._build_frame(start, np.zeros(3, dtype=np.float32))
        self._frame_buf.clear()
        for _ in range(self.cfg.stack_size):
            self._frame_buf.append(frame.copy())
        return np.concatenate(list(self._frame_buf)), info

    def step(self, action: np.ndarray):
        action_nn = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_action = action_nn * self.cfg.action_scale

        info: Dict[str, Any] = {}
        for _ in range(self.cfg.substeps):
            self._ema_action = self.cfg.ema_alpha * target_action + (1.0 - self.cfg.ema_alpha) * self._ema_action
            _, _, _, _, info = self.env.step(np.expand_dims(self._ema_action, axis=0))
            self.world.advance_dynamics()

        self._step_counter += 1
        pos, vel, rpy, heading = self._kinematics()
        frame = self._build_frame(pos, rpy)
        self._frame_buf.append(frame)

        lidar_scan = frame[-ForestMapConfig.lidar_rays :]
        min_lidar = float(np.min(lidar_scan) * self.world.cfg.lidar_max_range)

        dist_xy = math.hypot(self._goal[0] - pos[0], self._goal[1] - pos[1])
        goal_dir = self._goal - pos
        goal_norm = float(np.linalg.norm(goal_dir))
        unit_goal = goal_dir / goal_norm if goal_norm > 1e-3 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dz = float(pos[2] - self.cfg.target_altitude)

        r_step = self.cfg.r_step
        r_height = self.cfg.r_height_scale * math.exp(-self.cfg.r_height_k * dz * dz)
        r_att = self.cfg.r_att_penalty * (abs(float(rpy[0])) + abs(float(rpy[1])))
        r_smooth = self.cfg.r_smooth_penalty * float(np.sum(np.square(action_nn - self._prev_action)))

        v_toward = float(np.clip(float(np.dot(vel, unit_goal)), -5.0, 5.0))
        r_approach = self.cfg.r_approach * v_toward

        v_xy_norm = math.hypot(float(vel[0]), float(vel[1]))
        h_norm = float(np.linalg.norm(heading[:2]))
        g_norm = float(np.linalg.norm(unit_goal[:2]))
        cos_theta = float(np.dot(heading[:2], unit_goal[:2]) / (h_norm * g_norm + 1e-6))
        r_dir = self.cfg.r_dir * v_xy_norm * cos_theta

        r_repulsion = 0.0
        if min_lidar < 1.0:
            r_repulsion = self.cfg.r_repulsion_max * math.exp(-1.5 * (min_lidar - self.cfg.safe_lidar_dist))

        raw_cont = r_step + r_height + r_att + r_smooth + r_approach + r_dir + r_repulsion
        clipped_cont = float(np.clip(raw_cont, -1.0, 1.0))

        terminal = 0.0
        terminated = False
        truncated = False
        reason = "ALIVE"

        if min_lidar < self.cfg.safe_lidar_dist:
            terminal, terminated, reason = self.cfg.r_crash, True, "CRASH_LIDAR"
        elif pos[2] < 0.1:
            terminal, terminated, reason = self.cfg.r_crash, True, "CRASH_FLOOR"
        elif max(abs(float(rpy[0])), abs(float(rpy[1]))) > self.cfg.max_rp_angle:
            terminal, terminated, reason = self.cfg.r_crash, True, "CRASH_FLIP"
        elif abs(dz) > self.cfg.max_z_err:
            terminal, terminated, reason = self.cfg.r_deviate, True, "CRASH_Z_DEVIATE"
        elif dist_xy < self.cfg.success_xy_tol and abs(dz) < self.cfg.success_z_tol:
            time_bonus = max(0, self.cfg.max_steps - self._step_counter) * self.cfg.time_bonus_coef
            terminal = self.cfg.r_success_base + time_bonus
            truncated, reason = True, "SUCCESS"
        elif self._step_counter >= self.cfg.max_steps:
            truncated, reason = True, "TIMEOUT"

        reward = clipped_cont + terminal
        self._prev_action = action_nn.copy()

        info["forest_stats"] = {
            "r_step": r_step,
            "r_height": r_height,
            "r_att": r_att,
            "r_smooth": r_smooth,
            "r_approach": r_approach,
            "r_dir": r_dir,
            "r_repulsion": r_repulsion,
            "r_cont_clipped": clipped_cont,
            "r_terminal": terminal,
            "r_final_total": reward,
            "dist_xy": dist_xy,
            "pos_z": float(pos[2]),
            "min_lidar": min_lidar,
            "reason": reason,
        }
        return np.concatenate(list(self._frame_buf)), float(reward), terminated, truncated, info
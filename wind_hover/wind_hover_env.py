"""抗风残差姿态控制悬停环境（RL 输出残差，底层 DSLPID 保底）。"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics


class WindHoverResidualEnv(BaseRLAviary):
    """残差控制 + 历史状态堆叠 + 可课程化风场注入。"""

    FRAME_DIM = 12  # pos_err(3), vel(3), rpy(3), rpy_rates(3)

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        history_steps: int = 5,
        max_ctrl_steps: int = 1000,
        wind_force_scale: float = 0.08,
    ):
        self.TARGET_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.HISTORY_STEPS = history_steps
        self.MAX_CTRL_STEPS = max_ctrl_steps
        self.WIND_FORCE_SCALE = wind_force_scale
        self.obs_history: deque[np.ndarray] = deque(maxlen=self.HISTORY_STEPS)

        self.pid_controller = DSLPIDControl(drone_model=drone_model)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.current_action = np.zeros(4, dtype=np.float32)
        self.ctrl_step_counter = 0
        self.wind_intensity = 0.0
        self._step_gust = np.zeros(3, dtype=np.float32)

        self.EPISODE_LEN_SEC = max_ctrl_steps / ctrl_freq

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs if initial_xyzs is not None else np.array([[0.0, 0.0, 1.0]]),
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
        )

    def set_wind_intensity(self, intensity: float) -> None:
        self.wind_intensity = float(max(0.0, intensity))

    def _actionSpace(self):
        low = np.array([[-1.0, -1.0, -1.0, -1.0]], dtype=np.float32)
        high = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, 4)))
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _observationSpace(self):
        dim = self.FRAME_DIM * self.HISTORY_STEPS
        low = np.full((self.NUM_DRONES, dim), -np.inf, dtype=np.float32)
        high = np.full((self.NUM_DRONES, dim), np.inf, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.obs_history.clear()
        self.last_action.fill(0.0)
        self.current_action.fill(0.0)
        self.ctrl_step_counter = 0
        self.pid_controller.reset()
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.ctrl_step_counter += 1
        if self.wind_intensity > 0.0:
            self._step_gust = (
                np.random.uniform(-0.3, 0.3, 3).astype(np.float32) * self.wind_intensity
            )
        else:
            self._step_gust.fill(0.0)
        return super().step(action)

    def _frame_obs(self) -> np.ndarray:
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        rpy_rates = state[13:16]
        pos_error = pos - self.TARGET_POS
        return np.hstack([pos_error, vel, rpy, rpy_rates]).astype(np.float32)

    def _computeObs(self):
        frame = self._frame_obs()
        if len(self.obs_history) == 0:
            for _ in range(self.HISTORY_STEPS):
                self.obs_history.append(frame.copy())
        else:
            self.obs_history.append(frame)
        stacked = np.hstack(list(self.obs_history)).astype(np.float32)
        return np.expand_dims(stacked, axis=0)

    def _preprocessAction(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.NUM_DRONES, 4)
        self.action_buffer.append(action)
        self.current_action = action[0].copy()

        delta_thrust = action[0, 0] * 0.15
        target_roll = action[0, 1] * np.deg2rad(15.0)
        target_pitch = action[0, 2] * np.deg2rad(15.0)
        target_yaw_rate = action[0, 3] * 0.5

        state = self._getDroneStateVector(0)
        rpm, _, _ = self.pid_controller.computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=self.TARGET_POS,
            target_rpy=np.array([target_roll, target_pitch, 0.0]),
            target_vel=np.zeros(3),
            target_rpy_rates=np.array([0.0, 0.0, target_yaw_rate]),
        )
        rpm = np.clip(rpm * (1.0 + delta_thrust), 0.0, self.MAX_RPM)
        return np.expand_dims(rpm, axis=0)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]

        pos_error = float(np.linalg.norm(pos - self.TARGET_POS))
        vel_norm = float(np.linalg.norm(vel))
        rpy_norm = float(np.linalg.norm(rpy))

        # 近距离有正奖励，远距离平滑惩罚，便于 RL 获得有效梯度
        r_pos = 0.5 * float(np.exp(-5.0 * pos_error * pos_error)) - 0.15 * pos_error
        r_stab = -0.2 * vel_norm - 0.1 * rpy_norm
        r_smooth = -0.05 * float(np.linalg.norm(self.current_action - self.last_action))
        self.last_action = self.current_action.copy()
        r_alive = 0.5 if pos_error < 0.2 else 0.05

        return float(r_pos + r_stab + r_smooth + r_alive)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        if float(np.linalg.norm(pos - self.TARGET_POS)) > 1.5 or pos[2] < 0.1:
            return True
        return False

    def _computeTruncated(self):
        if self.ctrl_step_counter >= self.MAX_CTRL_STEPS:
            return True
        state = self._getDroneStateVector(0)
        if max(abs(float(state[7])), abs(float(state[8]))) > 1.05:
            return True
        return False

    def _computeInfo(self) -> Dict[str, Any]:
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        pos_error = float(np.linalg.norm(pos - self.TARGET_POS))
        return {
            "wind_stats": {
                "pos_error": pos_error,
                "pos_z": float(pos[2]),
                "wind_intensity": self.wind_intensity,
                "ctrl_steps": self.ctrl_step_counter,
            }
        }

    def _physics(self, rpm, nth_drone):
        super()._physics(rpm, nth_drone)
        if self.wind_intensity <= 0.0:
            return
        base_wind = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        total_force = self.WIND_FORCE_SCALE * self.wind_intensity * (base_wind + self._step_gust)
        p.applyExternalForce(
            objectUniqueId=self.DRONE_IDS[nth_drone],
            linkIndex=-1,
            forceObj=total_force.tolist(),
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.CLIENT,
        )

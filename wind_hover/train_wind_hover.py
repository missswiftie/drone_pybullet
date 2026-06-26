"""抗风残差悬停 PPO 训练（课程风场）。"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from collections import deque
from typing import Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from wind_hover_env import WindHoverResidualEnv


def make_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        env = WindHoverResidualEnv(gui=False)
        env.reset(seed=seed + rank)
        return env

    return _init


class PerformanceGatedWindCallback(BaseCallback):
    """达标后再升风：避免风场跑得比策略快。"""

    def __init__(
        self,
        max_wind: float = 1.5,
        wind_step: float = 0.1,
        check_freq: int = 8192,
        err_threshold: float = 0.25,
        len_threshold: float = 300,
        min_ready_episodes: int = 30,
        min_steps_per_level: int = 40_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.max_wind = max_wind
        self.wind_step = wind_step
        self.check_freq = check_freq
        self.err_threshold = err_threshold
        self.len_threshold = len_threshold
        self.min_ready_episodes = min_ready_episodes
        self.min_steps_per_level = min_steps_per_level
        self.current_wind = 0.0
        self.last_increase_step = 0
        self.recent_errs: deque = deque(maxlen=50)
        self.recent_lens: deque = deque(maxlen=50)
        self._ep_return: np.ndarray | None = None
        self._ep_length: np.ndarray | None = None

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_return = np.zeros(n, dtype=np.float64)
        self._ep_length = np.zeros(n, dtype=np.int64)
        self.training_env.set_attr("wind_intensity", 0.0)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals.get("infos", [])

        self._ep_return += rewards
        self._ep_length += 1

        for i, (done, info) in enumerate(zip(dones, infos)):
            if not done:
                continue
            self.recent_lens.append(int(self._ep_length[i]))
            if isinstance(info, dict) and "wind_stats" in info:
                self.recent_errs.append(float(info["wind_stats"].get("pos_error", 0.0)))
            self._ep_return[i] = 0.0
            self._ep_length[i] = 0

        if self.n_calls % self.check_freq != 0:
            return True
        if len(self.recent_errs) < self.min_ready_episodes:
            if self.verbose:
                print(
                    f"[wind] hold={self.current_wind:.2f} "
                    f"(collecting {len(self.recent_errs)}/{self.min_ready_episodes} episodes)"
                )
            return True

        mean_err = float(np.mean(self.recent_errs))
        mean_len = float(np.mean(self.recent_lens))
        ready = mean_err < self.err_threshold and mean_len >= self.len_threshold
        cooled = (self.num_timesteps - self.last_increase_step) >= self.min_steps_per_level

        if ready and cooled and self.current_wind < self.max_wind:
            self.current_wind = min(self.max_wind, self.current_wind + self.wind_step)
            self.last_increase_step = self.num_timesteps
            self.training_env.set_attr("wind_intensity", self.current_wind)
            if self.verbose:
                print(
                    f"[wind] UP -> {self.current_wind:.2f} "
                    f"(err={mean_err:.3f}m len={mean_len:.0f})"
                )
        elif self.verbose:
            why = "cooldown" if not cooled else f"err={mean_err:.3f} len={mean_len:.0f}"
            print(f"[wind] hold={self.current_wind:.2f} ({why})")
        return True


class TrainMonitor(BaseCallback):
    """统计回合累计回报、回合长度与终局位置误差。"""

    def __init__(self, every: int):
        super().__init__()
        self.every = every
        self.ep_rews: deque = deque(maxlen=100)
        self.ep_lens: deque = deque(maxlen=100)
        self.ep_errs: deque = deque(maxlen=100)
        self._ep_return: np.ndarray | None = None
        self._ep_length: np.ndarray | None = None

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_return = np.zeros(n, dtype=np.float64)
        self._ep_length = np.zeros(n, dtype=np.int64)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals.get("infos", [])

        self._ep_return += rewards
        self._ep_length += 1

        for i, (done, info) in enumerate(zip(dones, infos)):
            if not done:
                continue
            self.ep_rews.append(float(self._ep_return[i]))
            self.ep_lens.append(int(self._ep_length[i]))
            if isinstance(info, dict) and "wind_stats" in info:
                self.ep_errs.append(float(info["wind_stats"].get("pos_error", 0.0)))
            self._ep_return[i] = 0.0
            self._ep_length[i] = 0

        if self.num_timesteps > 0 and self.num_timesteps % self.every == 0 and self.ep_rews:
            err = np.mean(self.ep_errs) if self.ep_errs else float("nan")
            wind = 0.0
            if infos and isinstance(infos[0], dict):
                wind = float(infos[0].get("wind_stats", {}).get("wind_intensity", 0.0))
            print(
                f"[{self.num_timesteps:08d}] "
                f"ep_reward={np.mean(self.ep_rews):.2f} "
                f"ep_len={np.mean(self.ep_lens):.1f} "
                f"pos_err={err:.3f}m "
                f"wind={wind:.2f}"
            )
        return True


def _tensorboard_log(save_dir: str) -> str | None:
    try:
        import tensorboard  # noqa: F401

        return os.path.join(save_dir, "tensorboard")
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description="抗风残差悬停 PPO 训练")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--max-wind", type=float, default=1.5)
    parser.add_argument("--wind-step", type=float, default=0.1)
    parser.add_argument("--wind-check-freq", type=int, default=8192)
    parser.add_argument("--wind-err-threshold", type=float, default=0.25)
    parser.add_argument("--wind-len-threshold", type=float, default=300)
    parser.add_argument("--check-env", action="store_true", help="训练前运行 gym env_checker")
    args = parser.parse_args()

    save = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "wind_hover")
    os.makedirs(save, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[INFO] 性能门控风场: max={args.max_wind} step={args.wind_step} "
        f"err<{args.wind_err_threshold}m len>={args.wind_len_threshold}"
    )

    if args.check_env:
        check_env(WindHoverResidualEnv(gui=False), warn=True)

    set_random_seed(42)
    if args.num_envs > 1:
        vec = SubprocVecEnv([make_env(i, 42) for i in range(args.num_envs)])
    else:
        vec = DummyVecEnv([make_env(0, 42)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    rollout = 2048
    model = PPO(
        "MlpPolicy",
        vec,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=3e-4,
        n_steps=rollout,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        max_grad_norm=0.5,
        device=device,
        tensorboard_log=_tensorboard_log(save),
        verbose=1,
    )

    ckpt = CheckpointCallback(
        save_freq=max(100_000 // max(args.num_envs, 1), 1),
        save_path=save,
        name_prefix="ppo_wind_hover",
    )
    wind_cb = PerformanceGatedWindCallback(
        max_wind=args.max_wind,
        wind_step=args.wind_step,
        check_freq=args.wind_check_freq,
        err_threshold=args.wind_err_threshold,
        len_threshold=args.wind_len_threshold,
    )
    mon = TrainMonitor(every=rollout * max(args.num_envs, 1))

    print("开始抗风残差悬停训练...")
    model.learn(args.steps, callback=[ckpt, wind_cb, mon], progress_bar=True)
    model.save(os.path.join(save, "wind_hover_final"))
    vec.save(os.path.join(save, "vec_normalize.pkl"))
    print(f"完成: {save}")
    vec.close()


if __name__ == "__main__":
    main()

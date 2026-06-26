"""Task 1 PPO 训练。"""

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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from hover_env import HoverHoldEnv, make_aviary


def linear_lr(start: float, end: float = 1e-5):
    def fn(progress_remaining: float) -> float:
        return progress_remaining * (start - end) + end

    return fn


def make_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        env = HoverHoldEnv(make_aviary())
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


class TrainMonitor(BaseCallback):
    def __init__(self, every: int):
        super().__init__()
        self.every = every
        self.rews: deque = deque(maxlen=100)
        self.lens: deque = deque(maxlen=100)
        self.success: deque = deque(maxlen=100)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "hover_stats" in info:
                pass
            if "episode" in info:
                self.rews.append(info["episode"]["r"])
                self.lens.append(info["episode"]["l"])
                ok = 1.0 if info.get("hover_stats", {}).get("r_raw", 0) > 5.0 else 0.0
                self.success.append(ok)
        if self.num_timesteps > 0 and self.num_timesteps % self.every == 0 and self.rews:
            print(
                f"[{self.num_timesteps:07d}] reward={np.mean(self.rews):.2f} "
                f"len={np.mean(self.lens):.1f} success={np.mean(self.success)*100:.1f}%"
            )
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3_000_000)
    args = parser.parse_args()

    save = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "task1")
    os.makedirs(save, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device} num_envs={args.num_envs}")

    set_random_seed(42)
    vec = SubprocVecEnv([make_env(i, 42) for i in range(args.num_envs)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=linear_lr(3e-4),
        n_steps=1024,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        max_grad_norm=0.5,
        device=device,
        tensorboard_log=None,
        verbose=0,
    )

    ckpt = CheckpointCallback(
        save_freq=max(200_000 // args.num_envs, 1),
        save_path=save,
        name_prefix="ppo_hover",
    )
    mon = TrainMonitor(every=10240)

    model.learn(args.steps, callback=[ckpt, mon], progress_bar=True)
    model.save(os.path.join(save, "hover_final"))
    vec.save(os.path.join(save, "vec_normalize.pkl"))
    print(f"完成: {save}")
    vec.close()


if __name__ == "__main__":
    main()

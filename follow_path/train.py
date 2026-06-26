"""Task 2 PPO 训练。"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from collections import Counter, deque
from typing import Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from path_env import PathTrackConfig, PathTrackEnv, make_aviary


def linear_lr(start: float, end: float = 1e-5):
    def fn(progress_remaining: float) -> float:
        return progress_remaining * (start - end) + end

    return fn


def make_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        env = PathTrackEnv(make_aviary())
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


class TrainMonitor(BaseCallback):
    def __init__(self, every: int):
        super().__init__()
        self.every = every
        self.rews: deque = deque(maxlen=100)
        self.comps: deque = deque(maxlen=100)
        self.reasons: deque = deque(maxlen=100)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info or "path_stats" not in info:
                continue
            self.rews.append(info["episode"]["r"])
            self.comps.append(info["path_stats"].get("completion_rate", 0.0))
            self.reasons.append(info["path_stats"].get("reason", "?"))
        if self.num_timesteps > 0 and self.num_timesteps % self.every == 0 and self.rews:
            rc = Counter(self.reasons)
            rs = ", ".join(f"{k}:{v/len(self.reasons)*100:.0f}%" for k, v in rc.items())
            print(
                f"[{self.num_timesteps:08d}] reward={np.mean(self.rews):.1f} "
                f"completion={np.mean(self.comps):.1f}% | {rs}"
            )
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=15_000_000)
    args = parser.parse_args()

    save = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "task2")
    os.makedirs(save, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device} num_envs={args.num_envs}")

    set_random_seed(42)
    vec = SubprocVecEnv([make_env(i, 42) for i in range(args.num_envs)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    rollout = 2048
    model = PPO(
        "MlpPolicy",
        vec,
        policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])),
        learning_rate=linear_lr(2e-4),
        n_steps=rollout,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        max_grad_norm=0.5,
        target_kl=0.015,
        device=device,
        tensorboard_log=None,
        verbose=0,
    )

    ckpt = CheckpointCallback(
        save_freq=max(500_000 // args.num_envs, 1),
        save_path=save,
        name_prefix="ppo_track",
    )
    mon = TrainMonitor(every=rollout * args.num_envs)

    model.learn(args.steps, callback=[ckpt, mon], progress_bar=True)
    model.save(os.path.join(save, "track_final"))
    vec.save(os.path.join(save, "vec_normalize.pkl"))
    print(f"完成: {save}")
    vec.close()


if __name__ == "__main__":
    main()

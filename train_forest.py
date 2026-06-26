"""PPO 三阶段课程训练"""

from __future__ import annotations

import os

# 必须在 numpy/torch 之前设置，避免多进程时线程数爆炸触发 RLIMIT_NPROC
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import re
from collections import Counter, deque
from typing import Callable, List

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from forest_env import ForestNavEnv, make_base_aviary


def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value

    return func


def make_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        env = ForestNavEnv(make_base_aviary())
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def _default_num_envs() -> int:
    """默认并行环境数：参考 Task3 为 10，上限不超过物理核数。"""
    return min(16, max(4, (os.cpu_count() or 8) // 2))


def _resolve_vec_norm_path(resume_path: str, explicit: str | None, save_dir: str) -> str | None:
    if explicit and os.path.isfile(explicit):
        return explicit
    ckpt_dir = os.path.dirname(os.path.abspath(resume_path))
    steps = _parse_steps_from_ckpt(resume_path)
    if steps is not None:
        for name in (
            f"ppo_forest_vecnormalize_{steps}_steps.pkl",
            f"vec_normalize_{steps}.pkl",
        ):
            tagged = os.path.join(ckpt_dir, name)
            if os.path.isfile(tagged):
                return tagged
    for candidate in (
        os.path.join(ckpt_dir, "vec_normalize.pkl"),
        os.path.join(save_dir, "vec_normalize.pkl"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


class PhaseMonitor(BaseCallback):
    def __init__(self, check_freq: int, phase_name: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.phase_name = phase_name
        self.ep_rewards: deque = deque(maxlen=100)
        self.ep_lengths: deque = deque(maxlen=100)
        self.final_dists: deque = deque(maxlen=100)
        self.death_reasons: deque = deque(maxlen=100)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info or "forest_stats" not in info:
                continue
            self.ep_rewards.append(info["episode"]["r"])
            self.ep_lengths.append(info["episode"]["l"])
            stats = info["forest_stats"]
            self.final_dists.append(stats.get("dist_xy", 0.0))
            self.death_reasons.append(stats.get("reason", "UNKNOWN"))

        if self.num_timesteps > 0 and self.num_timesteps % self.check_freq == 0:
            self._print_log()
        return True

    def _print_log(self) -> None:
        if not self.ep_rewards:
            return
        reasons = Counter(self.death_reasons)
        success_rate = reasons.get("SUCCESS", 0) / len(self.death_reasons) * 100.0
        reason_str = ", ".join(f"{k}:{v / len(self.death_reasons) * 100:.0f}%" for k, v in reasons.items())
        lr = self.model.policy.optimizer.param_groups[0]["lr"]
        print(
            f"\n[{self.num_timesteps:08d} | {self.phase_name}] "
            f"reward={np.mean(self.ep_rewards):.1f} len={np.mean(self.ep_lengths):.1f} "
            f"success={success_rate:.1f}% dist={np.mean(self.final_dists):.1f}m | {reason_str} | lr={lr:.2e}"
        )


def _resolve_device() -> str:
    """优先使用 CUDA 进行 PPO 策略网络训练。"""
    if torch.cuda.is_available():
        print(f"[INFO] 使用 CUDA 训练: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("[WARN] CUDA 不可用，回退到 CPU 训练。")
    return "cpu"


def _tensorboard_log_dir(save_dir: str) -> str | None:
    try:
        import tensorboard  # noqa: F401

        return os.path.join(save_dir, "tensorboard")
    except ImportError:
        print("[INFO] 未安装 tensorboard，跳过 TensorBoard 日志（不影响训练）。")
        return None


CURRICULUM: List[dict] = [
    {"name": "phase1", "static": 5, "dynamic": 0, "goal_dist": 25.0, "steps": 20_000_000},
    {"name": "phase2", "static": 10, "dynamic": 2, "goal_dist": 25.0, "steps": 20_000_000},
    {"name": "phase3", "static": 25, "dynamic": 4, "goal_dist": 45.0, "steps": 20_000_000},
]

_PHASE_ALIASES = {"1": 0, "2": 1, "3": 2, "phase1": 0, "phase2": 1, "phase3": 2}


def _parse_start_phase(value: str) -> int:
    key = value.strip().lower()
    if key not in _PHASE_ALIASES:
        raise argparse.ArgumentTypeError(
            f"无效 phase: {value!r}，可选 1/2/3 或 phase1/phase2/phase3"
        )
    return _PHASE_ALIASES[key]


def _phase_start_timesteps(phase_idx: int) -> int:
    return sum(p["steps"] for p in CURRICULUM[:phase_idx])


def _phase_end_timesteps(phase_idx: int) -> int:
    return _phase_start_timesteps(phase_idx) + CURRICULUM[phase_idx]["steps"]


def _parse_steps_from_ckpt(path: str) -> int | None:
    """从文件名推断全局已训步数（phase_done / ppo_forest_N_steps）。"""
    base = os.path.basename(path)
    m = re.search(r"(\d+)_steps", base)
    if m:
        return int(m.group(1))
    if "phase1_done" in base:
        return _phase_end_timesteps(0)
    if "phase2_done" in base:
        return _phase_end_timesteps(1)
    if "phase3_done" in base or "forest_nav_final" in base:
        return _phase_end_timesteps(2)
    return None


def _infer_phase_index(global_ts: int) -> int:
    for i, phase in enumerate(CURRICULUM):
        if global_ts < _phase_end_timesteps(i):
            return i
    return len(CURRICULUM) - 1


def _remaining_phase_steps(global_ts: int, phase_idx: int) -> int:
    start = _phase_start_timesteps(phase_idx)
    end = _phase_end_timesteps(phase_idx)
    if global_ts >= end:
        return 0
    if global_ts <= start:
        return CURRICULUM[phase_idx]["steps"]
    return end - global_ts


def _resolve_global_timesteps(model: PPO, resume_path: str) -> int:
    ts = int(getattr(model, "num_timesteps", 0))
    parsed = _parse_steps_from_ckpt(resume_path)
    if parsed is not None:
        if ts <= 0 or abs(ts - parsed) > 10_000:
            if ts > 0 and abs(ts - parsed) > 10_000:
                print(f"[WARN] model.num_timesteps={ts} 与文件名步数 {parsed} 不一致，采用较大值")
            ts = max(ts, parsed)
    return ts


def main() -> None:
    parser = argparse.ArgumentParser(description="密林避障 PPO 训练")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=int(os.environ.get("FOREST_NUM_ENVS", _default_num_envs())),
        help="SubprocVecEnv 并行环境数（过大易触发 BrokenPipe / RLIMIT_NPROC）",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="从 checkpoint 恢复，例如 checkpoints/forest_nav/phase1_done.zip",
    )
    parser.add_argument(
        "--vec-norm",
        default=None,
        help="VecNormalize 统计量路径，默认与 --resume 同目录下的 vec_normalize.pkl",
    )
    parser.add_argument(
        "--start-phase",
        type=_parse_start_phase,
        default=0,
        help="从哪个阶段开始训：1/2/3 或 phase1/phase2/phase3（默认 phase1）",
    )
    args = parser.parse_args()
    num_cpu = max(1, args.num_envs)
    start_idx = args.start_phase

    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "forest_nav")
    os.makedirs(save_dir, exist_ok=True)

    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(f"未找到 checkpoint: {args.resume}")

    if num_cpu > 24:
        print(f"[WARN] num-envs={num_cpu} 较大，若出现 BrokenPipe 请降至 10~16。")

    torch.set_num_threads(1)
    print(f"[INFO] 并行环境数: {num_cpu}（可通过 --num-envs 或环境变量 FOREST_NUM_ENVS 调整）")

    set_random_seed(42)
    vec_env = SubprocVecEnv([make_env(i, seed=42) for i in range(num_cpu)])

    vec_norm_path = _resolve_vec_norm_path(args.resume, args.vec_norm, save_dir) if args.resume else None
    if args.resume:
        if vec_norm_path:
            vec_env = VecNormalize.load(vec_norm_path, vec_env)
            print(f"[INFO] 已加载 VecNormalize: {vec_norm_path}")
            steps = _parse_steps_from_ckpt(args.resume)
            if steps and not os.path.basename(vec_norm_path).startswith(
                ("ppo_forest_vecnormalize_", "vec_normalize_")
            ):
                print(
                    f"[WARN] 未找到与 checkpoint 步数 {steps:,} 配对的 "
                    f"vec_normalize_{steps}.pkl，当前文件可能过旧，resume 后指标会暂时偏低"
                )
        else:
            print(
                "[WARN] 未找到 vec_normalize.pkl，将使用新的观测归一化。"
                "旧 checkpoint 的表现会显著下降，需重新积累统计量。"
            )
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    rollout = 2048
    batch_size = 1024
    policy_kwargs = dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]))
    device = _resolve_device()

    if args.resume:
        print(f"[INFO] 从 checkpoint 恢复: {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device=device)
        global_ts = _resolve_global_timesteps(model, args.resume)
        model.num_timesteps = global_ts
        inferred_phase = _infer_phase_index(global_ts)
        print(
            f"[INFO] checkpoint 全局步数={global_ts:,} "
            f"对应 {CURRICULUM[inferred_phase]['name']} "
            f"(phase 内 {_remaining_phase_steps(global_ts, inferred_phase):,}/"
            f"{CURRICULUM[inferred_phase]['steps']:,} 步待训)"
        )
        if inferred_phase > start_idx:
            print(
                f"[INFO] --start-phase {CURRICULUM[start_idx]['name']} 落后于 checkpoint，"
                f"改为从 {CURRICULUM[inferred_phase]['name']} 继续"
            )
            start_idx = inferred_phase
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(3e-4, 1e-5),
            n_steps=rollout,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.015,
            device=device,
            tensorboard_log=_tensorboard_log_dir(save_dir),
            verbose=0,
        )
        global_ts = 0

    ckpt_freq = max(1_000_000 // num_cpu, 1)
    ckpt = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=save_dir,
        name_prefix="ppo_forest",
        save_vecnormalize=True,
    )

    if start_idx > 0 and not args.resume:
        print(f"[INFO] 跳过前 {start_idx} 个阶段，从 {CURRICULUM[start_idx]['name']} 开始")

    for i, phase in enumerate(CURRICULUM):
        if i < start_idx:
            continue

        remaining = _remaining_phase_steps(int(model.num_timesteps), i)
        if remaining <= 0:
            print(f"\n>>> {phase['name']}: 已完成（全局步数 {model.num_timesteps:,}），跳过")
            continue

        print(
            f"\n>>> {phase['name']}: static={phase['static']} dynamic={phase['dynamic']} "
            f"goal={phase['goal_dist']}m | 本阶段续训 {remaining:,}/{phase['steps']:,} 步"
        )
        vec_env.env_method("set_curriculum", phase["static"], phase["dynamic"], phase["goal_dist"])
        monitor = PhaseMonitor(check_freq=rollout * num_cpu, phase_name=phase["name"])
        reset_ts = (not args.resume) and (i == 0) and (int(model.num_timesteps) == 0)
        model.learn(
            total_timesteps=remaining,
            callback=[ckpt, monitor],
            reset_num_timesteps=reset_ts,
            progress_bar=True,
        )
        model.save(os.path.join(save_dir, f"{phase['name']}_done"))
        vec_env.save(os.path.join(save_dir, "vec_normalize.pkl"))
        vec_env.save(os.path.join(save_dir, f"vec_normalize_{int(model.num_timesteps)}.pkl"))

    final_path = os.path.join(save_dir, "forest_nav_final")
    model.save(final_path)
    vec_env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    print(f"\n训练完成: {final_path}.zip")


if __name__ == "__main__":
    main()

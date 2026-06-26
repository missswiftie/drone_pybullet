"""四个任务的成功率 / 完工率批量评估（默认各抽样 100 局）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "stay_height"))
sys.path.insert(0, os.path.join(ROOT, "follow_path"))
sys.path.insert(0, os.path.join(ROOT, "wind_hover"))

from forest_env import ForestNavEnv, make_base_aviary  # noqa: E402
from follow_path.path_env import PathTrackConfig, PathTrackEnv, make_aviary as make_path_aviary  # noqa: E402
from stay_height.hover_env import HoverHoldConfig, HoverHoldEnv, make_aviary as make_hover_aviary  # noqa: E402
from wind_hover.wind_hover_env import WindHoverResidualEnv  # noqa: E402


@dataclass
class EpisodeResult:
    success: bool
    completion: float
    reason: str
    steps: int
    reward: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSummary:
    name: str
    episodes: int
    success_rate: float
    completion_rate: float
    mean_reward: float
    mean_steps: float
    reasons: dict[str, int]
    extras: dict[str, float]


def _load_model(model_path: str, env, device: str) -> PPO:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"未找到模型: {model_path}")
    return PPO.load(model_path, env=env, device=device)


def _wrap_vec(make_env: Callable[[], Any], vec_norm: str | None) -> DummyVecEnv | VecNormalize:
    holder: list[Any] = []

    def _factory():
        env = make_env()
        holder.append(env)
        return env

    env = DummyVecEnv([_factory])
    if vec_norm and os.path.isfile(vec_norm):
        env = VecNormalize.load(vec_norm, env)
        env.training = False
        env.norm_reward = False
    return env


def _run_episode(env, model: PPO, seed: int, setup: Callable[[Any], None] | None = None) -> tuple[float, int, bool, dict]:
    env.seed(seed)
    if setup is not None:
        setup(env)
    obs = env.reset()
    total, steps = 0.0, 0
    done = False
    info: dict = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, step_info = env.step(action)
        total += float(reward[0])
        steps += 1
        done = bool(done_arr[0])
        info = step_info[0] if step_info else {}
    return total, steps, done, info


def _classify_hover(info: dict) -> tuple[bool, float, str]:
    stats = info.get("hover_stats", {})
    stable = int(stats.get("stable_count", 0))
    dz = abs(float(stats.get("dz", 0.0)))
    pos_z = float(stats.get("pos_z", 0.0))
    req = HoverHoldConfig.success_steps_req
    completion = min(100.0, 100.0 * stable / req)

    if stable >= req:
        return True, 100.0, "SUCCESS"
    if pos_z < 0.1:
        return False, completion, "CRASH"
    if dz > HoverHoldConfig.max_z_err:
        return False, completion, "DEVIATE"
    return False, completion, "TIMEOUT"


def eval_stay_height(model: str, vec_norm: str | None, episodes: int, seed: int, device: str) -> TaskSummary:
    env = _wrap_vec(lambda: HoverHoldEnv(make_hover_aviary()), vec_norm)
    model_obj = _load_model(model, env, device)
    results: list[EpisodeResult] = []

    for i in range(episodes):
        reward, steps, _, info = _run_episode(env, model_obj, seed + i)
        ok, completion, reason = _classify_hover(info)
        stats = info.get("hover_stats", {})
        results.append(
            EpisodeResult(
                success=ok,
                completion=completion,
                reason=reason,
                steps=steps,
                reward=reward,
                extra={"dz": float(stats.get("dz", 0.0)), "stable_count": int(stats.get("stable_count", 0))},
            )
        )

    env.close()
    return _summarize("定高悬停", results)


def eval_follow_path(model: str, vec_norm: str | None, episodes: int, seed: int, device: str) -> TaskSummary:
    env = _wrap_vec(lambda: PathTrackEnv(make_path_aviary()), vec_norm)
    model_obj = _load_model(model, env, device)
    results: list[EpisodeResult] = []

    for i in range(episodes):
        reward, steps, _, info = _run_episode(env, model_obj, seed + i)
        stats = info.get("path_stats", {})
        reason = str(stats.get("reason", "UNKNOWN"))
        completion = float(stats.get("completion_rate", 0.0))
        ok = reason == "SUCCESS"
        results.append(
            EpisodeResult(
                success=ok,
                completion=completion,
                reason=reason,
                steps=steps,
                reward=reward,
                extra={"dist_err": float(stats.get("dist_err", 0.0))},
            )
        )

    env.close()
    return _summarize("轨迹跟踪", results)


def eval_wind_hover(model: str, vec_norm: str | None, episodes: int, seed: int, device: str, wind: float) -> TaskSummary:
    raw_holder: list[WindHoverResidualEnv] = []

    def _make():
        e = WindHoverResidualEnv(gui=False)
        raw_holder.append(e)
        return e

    env = _wrap_vec(_make, vec_norm)
    model_obj = _load_model(model, env, device)
    max_steps = raw_holder[0].MAX_CTRL_STEPS
    results: list[EpisodeResult] = []

    for i in range(episodes):
        def _setup(vec_env):
            raw_holder[0].set_wind_intensity(wind)

        reward, steps, _, info = _run_episode(env, model_obj, seed + i, setup=_setup)
        stats = info.get("wind_stats", {})
        pos_err = float(stats.get("pos_error", 0.0))
        completion = min(100.0, 100.0 * steps / max_steps)
        ok = pos_err < 0.25 and steps >= 300
        if pos_err > 1.5 or float(stats.get("pos_z", 1.0)) < 0.1:
            reason = "FAIL"
        elif steps >= max_steps:
            reason = "TIMEOUT"
        elif ok:
            reason = "SUCCESS"
        else:
            reason = "UNSTABLE"
        results.append(
            EpisodeResult(
                success=ok,
                completion=completion,
                reason=reason,
                steps=steps,
                reward=reward,
                extra={"pos_error": pos_err, "wind": wind},
            )
        )

    env.close()
    summary = _summarize("抗风悬停", results)
    summary.extras["mean_pos_error_m"] = float(np.mean([r.extra["pos_error"] for r in results]))
    summary.extras["wind_intensity"] = wind
    return summary


def eval_forest(
    model: str,
    vec_norm: str | None,
    episodes: int,
    seed: int,
    device: str,
    static: int,
    dynamic: int,
    goal_dist: float,
) -> TaskSummary:
    raw_holder: list[ForestNavEnv] = []

    def _make():
        e = ForestNavEnv(make_base_aviary())
        raw_holder.append(e)
        return e

    env = _wrap_vec(_make, vec_norm)

    def _setup(vec_env):
        vec_env.env_method("set_curriculum", static, dynamic, goal_dist)

    _setup(env)
    model_obj = _load_model(model, env, device)
    results: list[EpisodeResult] = []

    for i in range(episodes):
        raw = raw_holder[0]
        env.seed(seed + i)
        _setup(env)
        obs = env.reset()
        start = raw.world.start_pos.copy()
        goal = raw._goal.copy()
        init_dist = float(np.linalg.norm(goal[:2] - start[:2]))
        init_dist = max(init_dist, 1e-3)

        total, steps = 0.0, 0
        done = False
        info: dict = {}
        while not done:
            action, _ = model_obj.predict(obs, deterministic=True)
            obs, reward, done_arr, step_info = env.step(action)
            total += float(reward[0])
            steps += 1
            done = bool(done_arr[0])
            info = step_info[0] if step_info else {}

        stats = info.get("forest_stats", {})
        reason = str(stats.get("reason", "UNKNOWN"))
        final_dist = float(stats.get("dist_xy", init_dist))
        completion = float(np.clip(100.0 * (1.0 - final_dist / init_dist), 0.0, 100.0))
        ok = reason == "SUCCESS"
        results.append(
            EpisodeResult(
                success=ok,
                completion=completion,
                reason=reason,
                steps=steps,
                reward=total,
                extra={"dist_xy": final_dist, "init_dist_xy": init_dist},
            )
        )

    env.close()
    summary = _summarize("密林避障", results)
    summary.extras["mean_final_dist_xy_m"] = float(np.mean([r.extra["dist_xy"] for r in results]))
    summary.extras["static_obstacles"] = static
    summary.extras["dynamic_obstacles"] = dynamic
    summary.extras["goal_dist_m"] = goal_dist
    return summary


def _summarize(name: str, results: list[EpisodeResult]) -> TaskSummary:
    reasons = Counter(r.reason for r in results)
    return TaskSummary(
        name=name,
        episodes=len(results),
        success_rate=100.0 * sum(r.success for r in results) / len(results),
        completion_rate=float(np.mean([r.completion for r in results])),
        mean_reward=float(np.mean([r.reward for r in results])),
        mean_steps=float(np.mean([r.steps for r in results])),
        reasons=dict(reasons),
        extras={},
    )


def _print_summary(summary: TaskSummary) -> None:
    reasons = ", ".join(f"{k}:{v}" for k, v in sorted(summary.reasons.items()))
    print(f"\n【{summary.name}】 抽样 {summary.episodes} 局")
    print(f"  成功率: {summary.success_rate:.1f}%")
    print(f"  平均完工率: {summary.completion_rate:.1f}%")
    print(f"  平均回报: {summary.mean_reward:.2f}  平均步数: {summary.mean_steps:.1f}")
    print(f"  终止原因: {reasons}")
    for k, v in summary.extras.items():
        print(f"  {k}: {v}")


def main():
    ckpt = os.path.join(ROOT, "checkpoints")
    parser = argparse.ArgumentParser(description="四任务成功率/完工率评估")
    parser.add_argument("--episodes", type=int, default=100, help="每任务抽样局数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=os.path.join(ROOT, "bench_results.json"))
    parser.add_argument("--tasks", default="all", help="逗号分隔: hover,path,wind,forest 或 all")
    parser.add_argument("--hover-model", default=os.path.join(ckpt, "task1", "hover_final.zip"))
    parser.add_argument("--hover-vec-norm", default=os.path.join(ckpt, "task1", "vec_normalize.pkl"))
    parser.add_argument("--path-model", default=os.path.join(ckpt, "task2", "track_final.zip"))
    parser.add_argument("--path-vec-norm", default=os.path.join(ckpt, "task2", "vec_normalize.pkl"))
    parser.add_argument("--wind-model", default=os.path.join(ckpt, "wind_hover", "wind_hover_final.zip"))
    parser.add_argument("--wind-vec-norm", default=os.path.join(ckpt, "wind_hover", "vec_normalize.pkl"))
    parser.add_argument("--wind", type=float, default=1.5, help="抗风评估风强")
    parser.add_argument("--forest-model", default=os.path.join(ckpt, "forest_nav", "forest_nav_final.zip"))
    parser.add_argument("--forest-vec-norm", default=os.path.join(ckpt, "forest_nav", "vec_normalize.pkl"))
    parser.add_argument("--forest-static", type=int, default=15)
    parser.add_argument("--forest-dynamic", type=int, default=3)
    parser.add_argument("--forest-goal-dist", type=float, default=35.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = {"hover", "path", "wind", "forest"} if args.tasks == "all" else set(args.tasks.split(","))
    summaries: list[TaskSummary] = []
    t0 = time.time()

    print(f"[INFO] device={device} 每任务 {args.episodes} 局  seed={args.seed}")

    if "hover" in selected:
        print("\n评估定高悬停...")
        summaries.append(eval_stay_height(args.hover_model, args.hover_vec_norm, args.episodes, args.seed, device))

    if "path" in selected:
        print("评估轨迹跟踪...")
        summaries.append(eval_follow_path(args.path_model, args.path_vec_norm, args.episodes, args.seed + 1000, device))

    if "wind" in selected:
        print("评估抗风悬停...")
        summaries.append(
            eval_wind_hover(args.wind_model, args.wind_vec_norm, args.episodes, args.seed + 2000, device, args.wind)
        )

    if "forest" in selected:
        print("评估密林避障...")
        summaries.append(
            eval_forest(
                args.forest_model,
                args.forest_vec_norm,
                args.episodes,
                args.seed + 3000,
                device,
                args.forest_static,
                args.forest_dynamic,
                args.forest_goal_dist,
            )
        )

    print("\n" + "=" * 56)
    print("汇总")
    print("=" * 56)
    for s in summaries:
        _print_summary(s)

    payload = {
        "episodes_per_task": args.episodes,
        "seed": args.seed,
        "device": device,
        "elapsed_sec": round(time.time() - t0, 1),
        "tasks": [
            {
                "name": s.name,
                "episodes": s.episodes,
                "success_rate_pct": round(s.success_rate, 2),
                "completion_rate_pct": round(s.completion_rate, 2),
                "mean_reward": round(s.mean_reward, 3),
                "mean_steps": round(s.mean_steps, 1),
                "reasons": s.reasons,
                "extras": s.extras,
            }
            for s in summaries
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.output}  耗时 {payload['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()

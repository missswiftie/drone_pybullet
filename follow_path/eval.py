"""Task 2 模型评估 + 离屏渲染 MP4。"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl_render import (
    capture_3d,
    drone_position,
    path_3d_panel,
    path_xy_panel,
    save_combined_mp4,
    save_mp4,
)

from path_env import PathTrackConfig, PathTrackEnv, make_aviary


def main():
    parser = argparse.ArgumentParser(description="Task2 评估（默认渲染视频）")
    default = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "task2")
    parser.add_argument("--model", default=os.path.join(default, "track_final.zip"))
    parser.add_argument("--vec-norm", default=os.path.join(default, "vec_normalize.pkl"))
    parser.add_argument("--output-dir", default=os.path.join(default, "videos"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        fallback = os.path.join(default, "ppo_track_1000000_steps.zip")
        if os.path.isfile(fallback):
            print(f"未找到 {args.model}，使用 {fallback}")
            args.model = fallback
        else:
            print(f"未找到模型: {args.model}")
            return

    raw_holder: list[PathTrackEnv] = []

    def _make():
        e = PathTrackEnv(make_aviary())
        raw_holder.append(e)
        return e

    env = DummyVecEnv([_make])
    if os.path.isfile(args.vec_norm):
        env = VecNormalize.load(args.vec_norm, env)
        env.training = False
        env.norm_reward = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(args.model, env=env, device=device)
    raw = raw_holder[0]
    client = raw.unwrapped.CLIENT
    drone_id = raw.unwrapped.DRONE_IDS[0]

    env.seed(args.seed)
    obs = env.reset()
    total, steps = 0.0, 0
    actual_pts: list[np.ndarray] = []
    frames_3d, frames_xy, frames_3d_plot = [], [], []
    done = False
    ref_path = raw._path.points.copy() if raw._path else np.zeros((1, 3))

    while not done and steps < PathTrackConfig.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        total += float(r[0])
        steps += 1
        done = bool(done[0])

        pos = drone_position(client, drone_id)
        actual_pts.append(pos.copy())
        actual_arr = np.asarray(actual_pts)

        if not args.no_video and steps % args.frame_skip == 0:
            frames_3d.append(capture_3d(client, pos, distance=4.0, pitch=-28))
            frames_xy.append(path_xy_panel(actual_arr, ref_path, pos))
            frames_3d_plot.append(path_3d_panel(actual_arr, ref_path))

    s = info[0].get("path_stats", {})
    print(
        f"评估: steps={steps} reward={total:.2f} "
        f"completion={s.get('completion_rate', 0):.1f}% reason={s.get('reason')}"
    )

    if not args.no_video and frames_3d:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        p_cam = os.path.join(args.output_dir, f"track_cam_{tag}.mp4")
        p_xy = os.path.join(args.output_dir, f"track_xy_{tag}.mp4")
        p_3d = os.path.join(args.output_dir, f"track_3d_{tag}.mp4")
        p_combo = os.path.join(args.output_dir, f"track_combined_{tag}.mp4")
        save_mp4(frames_3d, p_cam, args.fps)
        save_mp4(frames_xy, p_xy, args.fps)
        save_mp4(frames_3d_plot, p_3d, args.fps)
        save_combined_mp4(frames_3d, frames_xy, p_combo, args.fps)
        print(f"视频已保存:\n  {p_cam}\n  {p_xy}\n  {p_3d}\n  {p_combo}")

    env.close()


if __name__ == "__main__":
    main()

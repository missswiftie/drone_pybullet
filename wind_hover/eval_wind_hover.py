"""抗风残差悬停评估 + 离屏视频渲染。"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl_render import capture_3d, hover_altitude_panel, save_combined_mp4, save_mp4

from wind_hover_env import WindHoverResidualEnv


def main():
    parser = argparse.ArgumentParser(description="抗风悬停评估（默认渲染视频）")
    default = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "wind_hover")
    parser.add_argument("--model", default=os.path.join(default, "wind_hover_final.zip"))
    parser.add_argument("--vec-norm", default=os.path.join(default, "vec_normalize.pkl"))
    parser.add_argument("--output-dir", default=os.path.join(default, "videos"))
    parser.add_argument("--wind", type=float, default=1.5, help="评估风场强度")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"未找到模型: {args.model}")
        return

    raw_holder: list[WindHoverResidualEnv] = []

    def _make():
        e = WindHoverResidualEnv(gui=False)
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
    raw.set_wind_intensity(args.wind)
    client = raw.CLIENT
    drone_id = raw.DRONE_IDS[0]
    target_z = float(raw.TARGET_POS[2])

    env.seed(args.seed)
    obs = env.reset()
    total, steps = 0.0, 0
    z_hist: list[float] = []
    frames_3d, frames_2d = [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done_arr, info = env.step(action)
        total += float(r[0])
        steps += 1
        done = bool(done_arr[0])

        pos, _ = p.getBasePositionAndOrientation(drone_id, physicsClientId=client)
        z_hist.append(float(pos[2]))

        if not args.no_video and steps % args.frame_skip == 0:
            frames_3d.append(capture_3d(client, pos, distance=2.8, pitch=-20))
            frames_2d.append(hover_altitude_panel(z_hist, target_z))

    s = info[0].get("wind_stats", {})
    print(
        f"评估: steps={steps} reward={total:.2f} pos_err={s.get('pos_error', 0):.3f}m "
        f"z={s.get('pos_z', 0):.3f} wind={args.wind}"
    )

    if not args.no_video and frames_3d:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        p3 = os.path.join(args.output_dir, f"wind_hover_3d_{tag}.mp4")
        p2 = os.path.join(args.output_dir, f"wind_hover_alt_{tag}.mp4")
        pc = os.path.join(args.output_dir, f"wind_hover_combined_{tag}.mp4")
        save_mp4(frames_3d, p3, args.fps)
        save_mp4(frames_2d, p2, args.fps)
        save_combined_mp4(frames_3d, frames_2d, pc, args.fps)
        print(f"视频已保存:\n  {p3}\n  {p2}\n  {pc}")

    env.close()


if __name__ == "__main__":
    main()

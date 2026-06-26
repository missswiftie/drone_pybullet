"""Task 1 模型评估 + 离屏渲染 MP4。"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl_render import capture_3d, drone_position, hover_altitude_panel, save_combined_mp4, save_mp4

from hover_env import HoverHoldConfig, HoverHoldEnv, make_aviary


def main():
    parser = argparse.ArgumentParser(description="Task1 评估（默认渲染视频）")
    default = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "task1")
    parser.add_argument("--model", default=os.path.join(default, "hover_final.zip"))
    parser.add_argument("--vec-norm", default=os.path.join(default, "vec_normalize.pkl"))
    parser.add_argument("--output-dir", default=os.path.join(default, "videos"))
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"未找到模型: {args.model}")
        return

    raw_holder: list[HoverHoldEnv] = []

    def _make():
        e = HoverHoldEnv(make_aviary())
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
    target_z = float(HoverHoldConfig().target_pos[2])

    env.seed(args.seed)
    obs = env.reset()
    total, steps = 0.0, 0
    z_hist: list[float] = []
    frames_3d, frames_2d = [], []
    done = False

    while not done and steps < HoverHoldConfig.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        total += float(r[0])
        steps += 1
        done = bool(done[0])

        pos = drone_position(client, drone_id)
        z_hist.append(float(pos[2]))

        if not args.no_video and steps % args.frame_skip == 0:
            frames_3d.append(capture_3d(client, pos, distance=2.8, pitch=-20))
            frames_2d.append(hover_altitude_panel(z_hist, target_z))

    s = info[0].get("hover_stats", {})
    print(
        f"评估: steps={steps} reward={total:.2f} z={s.get('pos_z', 0):.3f} "
        f"stable={s.get('stable_count', 0)}"
    )

    if not args.no_video and frames_3d:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        p3 = os.path.join(args.output_dir, f"hover_3d_{tag}.mp4")
        p2 = os.path.join(args.output_dir, f"hover_alt_{tag}.mp4")
        pc = os.path.join(args.output_dir, f"hover_combined_{tag}.mp4")
        save_mp4(frames_3d, p3, args.fps)
        save_mp4(frames_2d, p2, args.fps)
        save_combined_mp4(frames_3d, frames_2d, pc, args.fps)
        print(f"视频已保存:\n  {p3}\n  {p2}\n  {pc}")

    env.close()


if __name__ == "__main__":
    main()

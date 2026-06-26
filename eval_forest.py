"""加载模型评估，并离屏渲染 MP4 视频"""
from __future__ import annotations
import argparse
import os
from datetime import datetime
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from forest_env import ForestNavEnv, make_base_aviary
from forest_render import capture_2d_frame, capture_3d_frame, save_mp4, save_side_by_side_mp4

def run_eval(model_path: str, vec_norm_path: str | None, output_dir: str, static: int, dynamic: int, goal_dist: float, seed: int, record_video: bool, target_episodes: int, max_attempts: int, frame_skip: int, fps: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    raw_holder: list[ForestNavEnv] = []

    def _make():
        env = ForestNavEnv(make_base_aviary())
        raw_holder.append(env)
        return env

    env = DummyVecEnv([_make])
    if vec_norm_path and os.path.isfile(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    env.env_method("set_curriculum", static, dynamic, goal_dist)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, env=env, device=device)
    raw_env: ForestNavEnv = raw_holder[0]

    saved = 0
    attempts = 0
    while saved < target_episodes and attempts < max_attempts:
        attempts += 1
        env.seed(seed + attempts)
        obs = env.reset()
        frames_3d, frames_2d, trajectory = [], [], []
        total_reward, step_idx = 0.0, 0
        done = False
        reason = "UNKNOWN"

        print(f"\n回合 {attempts} 开始...")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env.step(action)
            total_reward += float(reward[0])
            done = bool(done_arr[0])
            step_idx += 1

            pos, _, _, _ = raw_env._kinematics()
            trajectory.append((float(pos[0]), float(pos[1])))

            if record_video and step_idx % frame_skip == 0:
                frames_3d.append(capture_3d_frame(raw_env))
                frames_2d.append(capture_2d_frame(raw_env, trajectory, trajectory[-1]))

            if done:
                reason = info[0].get("forest_stats", {}).get("reason", "UNKNOWN")

        print(f"回合 {attempts} 结束: steps={step_idx} reward={total_reward:.2f} reason={reason} "
            f"dist={info[0].get('forest_stats', {}).get('dist_xy', 0):.2f}m")

        if not record_video:
            continue

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"ep{attempts}_{reason.lower()}_{stamp}"
        path_3d = os.path.join(output_dir, f"{tag}_3d.mp4")
        path_2d = os.path.join(output_dir, f"{tag}_2d.mp4")
        path_combo = os.path.join(output_dir, f"{tag}_combined.mp4")

        if frames_3d:
            save_mp4(frames_3d, path_3d, fps=fps)
            save_mp4(frames_2d, path_2d, fps=fps)
            save_side_by_side_mp4(frames_3d, frames_2d, path_combo, fps=fps)
            print(f"视频已保存:\n  {path_3d}\n  {path_2d}\n  {path_combo}")
            saved += 1
        else:
            print("未采集到帧，跳过保存。")

    env.close()
    if record_video:
        print(f"\n共保存 {saved} 个回合视频到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="密林避障评估")
    default_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "forest_nav")
    parser.add_argument("--model", default=os.path.join(default_dir, "forest_nav_final.zip"))
    parser.add_argument("--vec-norm", default=os.path.join(default_dir, "vec_normalize.pkl"))
    parser.add_argument("--output-dir", default=os.path.join(default_dir, "videos"))
    parser.add_argument("--static", type=int, default=15)
    parser.add_argument("--dynamic", type=int, default=3)
    parser.add_argument("--goal-dist", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1, help="保存视频的回合数")
    parser.add_argument("--max-attempts", type=int, default=20, help="最多尝试回合数")
    parser.add_argument("--frame-skip", type=int, default=3, help="每隔多少 RL 步采一帧")
    parser.add_argument("--fps", type=int, default=12, help="输出视频帧率")
    parser.add_argument("--no-video", action="store_true", help="仅打印指标，不渲染视频")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"未找到模型 {args.model}，请先运行 train_forest.py")
        return

    run_eval(args.model, args.vec_norm if os.path.isfile(args.vec_norm) else None, args.output_dir, args.static, args.dynamic, args.goal_dist, args.seed, record_video=not args.no_video, target_episodes=args.episodes, max_attempts=args.max_attempts, frame_skip=args.frame_skip, fps=args.fps)

if __name__ == "__main__":
    main()
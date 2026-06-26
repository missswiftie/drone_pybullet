"""无 GUI 离屏渲染：3D 跟随视角 + 2D 俯视图，输出 MP4。"""

from __future__ import annotations
import os
from typing import List, Sequence, Tuple
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from forest_env import ForestNavEnv

def capture_3d_frame(env: ForestNavEnv, width: int = 480, height: int = 480) -> np.ndarray:
    """使用 ER_TINY_RENDERER 离屏渲染，无需 OpenGL 窗口。"""
    base = env.unwrapped
    client = base.CLIENT
    pos, _, _, _ = env._kinematics()

    view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=pos.tolist(), distance=5.0, yaw=45, pitch=-28, roll=0, upAxisIndex=2, physicsClientId=client)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height, nearVal=0.1, farVal=120.0, physicsClientId=client)
    _, _, rgba, _, _ = p.getCameraImage(width, height, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER, physicsClientId=client)
    rgb = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
    return rgb


def _obstacle_circles(env: ForestNavEnv) -> List[Tuple[float, float, float, str]]:
    world = env.world
    client = env.unwrapped.CLIENT
    dynamic_set = set(world._dynamic_ids)
    circles: List[Tuple[float, float, float, str]] = []
    for uid in world._static_ids + world._dynamic_ids:
        pos, _ = p.getBasePositionAndOrientation(uid, physicsClientId=client)
        r = world.cfg.dynamic_radius if uid in dynamic_set else world.cfg.static_radius_min
        for cx, cy, cr in world._pillar_records:
            if abs(cx - pos[0]) < 0.2 and abs(cy - pos[1]) < 0.2:
                r = cr
                break
        color = "#d94a38" if uid in dynamic_set else "#2d8f4e"
        circles.append((pos[0], pos[1], r, color))
    return circles


def capture_2d_frame(env: ForestNavEnv, trajectory: Sequence[Tuple[float, float]], current_xy: Tuple[float, float]) -> np.ndarray:
    world = env.world
    half = world.cfg.arena_size / 2.0
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_facecolor("#eef2f6")

    for x, y, r, color in _obstacle_circles(env):
        ax.add_patch(plt.Circle((x, y), r, color=color, alpha=0.75))

    ax.plot(world.start_pos[0], world.start_pos[1], "go", markersize=9, label="Start")
    ax.plot(world.goal_pos[0], world.goal_pos[1], "r*", markersize=14, label="Goal")
    if trajectory:
        traj = np.asarray(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1.8, alpha=0.7, label="Trajectory")
    ax.plot(current_xy[0], current_xy[1], "bo", markersize=7)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Top-Down Navigation & Obstacle Avoidance", fontsize=11)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def save_mp4(frames: List[np.ndarray], path: str, fps: int = 12) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)


def save_side_by_side_mp4(frames_3d: List[np.ndarray], frames_2d: List[np.ndarray], path: str, fps: int = 12) -> None:
    n = min(len(frames_3d), len(frames_2d))
    combined = []
    for i in range(n):
        a, b = frames_3d[i], frames_2d[i]
        h = max(a.shape[0], b.shape[0])
        a_r = _resize(a, h, int(a.shape[1] * h / a.shape[0]))
        b_r = _resize(b, h, int(b.shape[1] * h / b.shape[0]))
        combined.append(np.concatenate([a_r, b_r], axis=1))
    save_mp4(combined, path, fps=fps)


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))

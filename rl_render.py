"""Task 1/2 通用离屏渲染"""

from __future__ import annotations
import os
from typing import List, Sequence, Tuple
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def drone_position(client: int, body_id: int) -> np.ndarray:
    pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client)
    return np.array(pos, dtype=np.float32)


def capture_3d(client: int, target: Sequence[float], distance: float = 3.5, yaw: float = 45, pitch: float = -25, width: int = 480, height: int = 480) -> np.ndarray:
    view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=list(target), distance=distance, yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2, physicsClientId=client)
    proj = p.computeProjectionMatrixFOV(60, float(width) / height, 0.1, 50.0, physicsClientId=client)
    _, _, rgba, _, _ = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=client)
    return np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)


def hover_altitude_panel(z_hist: Sequence[float], target_z: float = 1.0) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    steps = range(len(z_hist))
    ax.plot(steps, z_hist, "b-", linewidth=2, label="Altitude Z")
    ax.axhline(target_z, color="r", linestyle="--", label=f"Target {target_z}m")
    ax.axhspan(target_z - 0.1, target_z + 0.1, color="green", alpha=0.15, label="Success ±0.1m")
    ax.set_xlabel("Step")
    ax.set_ylabel("Z (m)")
    ax.set_title("Task 1 Hover Altitude")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def path_xy_panel(actual: np.ndarray, target: np.ndarray, current: Sequence[float]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.plot(target[:, 0], target[:, 1], "r--", alpha=0.6, label="Reference")
    if len(actual) > 0:
        ax.plot(actual[:, 0], actual[:, 1], "b-", linewidth=2, label="Actual")
    ax.plot(current[0], current[1], "go", markersize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Task 2 Top-Down Tracking")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def path_3d_panel(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    n = min(len(target), 400)
    ax.plot(target[:n, 0], target[:n, 1], target[:n, 2], "r--", alpha=0.5, label="Reference")
    if len(actual) > 0:
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], "b-", linewidth=2, label="Actual")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Task 2 3D Tracking")
    ax.legend(fontsize=8)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def save_mp4(frames: List[np.ndarray], path: str, fps: int = 12) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)


def combine_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    from PIL import Image

    h = max(left.shape[0], right.shape[0])
    def rs(img):
        w = int(img.shape[1] * h / img.shape[0])
        return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
    return np.concatenate([rs(left), rs(right)], axis=1)


def save_combined_mp4(left_frames: List[np.ndarray], right_frames: List[np.ndarray], path: str, fps: int) -> None:
    n = min(len(left_frames), len(right_frames))
    save_mp4([combine_side_by_side(left_frames[i], right_frames[i]) for i in range(n)], path, fps)

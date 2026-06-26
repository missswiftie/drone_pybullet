"""环境冒烟测试。"""

from __future__ import annotations

import numpy as np

from forest_env import ForestNavConfig, ForestNavEnv, make_base_aviary
from forest_world import ForestMapConfig


def test_spaces():
    env = ForestNavEnv(make_base_aviary())
    assert env.action_space.shape == (4,)
    expected = ForestNavConfig.obs_dim_per_frame * ForestNavConfig.stack_size
    assert env.observation_space.shape == (expected,)
    env.close()
    print(f"[ok] 动作 4 维，观测 {expected} 维")


def test_rollout(episodes: int = 3):
    env = ForestNavEnv(make_base_aviary())
    env.set_curriculum(5, 1, 20.0)

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        assert obs.shape == env.observation_space.shape
        total_r, steps = 0.0, 0
        done = False
        while not done and steps < 300:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert np.isfinite(reward)
            total_r += reward
            steps += 1
            done = terminated or truncated
        s = info.get("forest_stats", {})
        print(
            f"  ep{ep}: steps={steps} reward={total_r:.2f} reason={s.get('reason')} "
            f"dist={s.get('dist_xy', 0):.2f}m min_lidar={s.get('min_lidar', 0):.2f}m"
        )
    env.close()
    print("[ok] rollout 通过")


def test_lidar_layout():
    assert ForestMapConfig.lidar_rays == ForestNavConfig.obs_dim_per_frame - 10
    print("[ok] 观测布局与参考一致 (34 = 3+4+3+24)")


if __name__ == "__main__":
    print("密林避障环境测试")
    test_spaces()
    test_lidar_layout()
    test_rollout()
    print("\n全部通过。")

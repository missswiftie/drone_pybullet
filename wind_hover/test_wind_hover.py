"""抗风悬停环境冒烟测试。"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from wind_hover_env import WindHoverResidualEnv


def main():
    env = WindHoverResidualEnv(gui=False, history_steps=5)
    check_env(env, warn=True)

    obs, _ = env.reset(seed=0)
    assert obs.shape == (1, 60), f"obs shape {obs.shape}"
    print(f"[ok] obs shape {obs.shape}, action {env.action_space.shape}")

    env.set_wind_intensity(1.0)
    total = 0.0
    for t in range(20):
        action = np.zeros((1, 4), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"[ok] 20 steps with wind=1.0, reward sum={total:.2f}")
    env.close()
    print("全部通过。")


if __name__ == "__main__":
    main()

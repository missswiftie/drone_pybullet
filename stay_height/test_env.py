"""Task 1 环境冒烟测试。"""

import numpy as np

from hover_env import HoverHoldConfig, HoverHoldEnv, make_aviary


def main():
    env = HoverHoldEnv(make_aviary())
    assert env.observation_space.shape == (44,)
    assert env.action_space.shape == (4,)

    obs, _ = env.reset(seed=0)
    total = 0.0
    for i in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            break
    print(f"rollout steps={i+1} reward={total:.2f} z={info['hover_stats']['pos_z']:.3f}")
    env.close()
    print("task1 测试通过")


if __name__ == "__main__":
    main()

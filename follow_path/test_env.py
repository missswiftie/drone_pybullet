"""Task 2 环境冒烟测试。"""

import numpy as np

from path_env import PathTrackConfig, PathTrackEnv, make_aviary


def main():
    env = PathTrackEnv(make_aviary())
    assert env.observation_space.shape == (100,)
    obs, _ = env.reset(seed=0)
    total = 0.0
    for i in range(30):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            break
    s = info["path_stats"]
    print(f"rollout steps={i+1} reward={total:.2f} dist={s['dist_err']:.2f} reason={s['reason']}")
    env.close()
    print("task2 测试通过")


if __name__ == "__main__":
    main()

# gym-pybullet-drones RL 实践

基于 [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)，复现 [0324Lw/gym-pybullet-drones](https://github.com/0324Lw/gym-pybullet-drones) 的 Task 1～3（独立实现，非 copy-paste）。全程 **无 GUI**。

## Task 1：定高悬停 (`task1/`)

```bash
conda activate ctph
cd /scratch/jiaqi/gym_pybullet/task1
python test_env.py
python train.py              # 默认 8 并行，300 万步
python eval.py
```

- 观测：44 维（11×4 帧堆叠）
- 网络：`[256, 256]`，PPO `n_steps=1024`，`batch_size=512`
- 权重：`checkpoints/task1/`

## Task 2：3D 轨迹追踪 (`task2/`)

```bash
cd /scratch/jiaqi/gym_pybullet/task2
python test_env.py
python train.py              # 默认 10 并行，1500 万步
python eval.py
```

- 观测：100 维（25×4 帧堆叠），含 5 个前瞻路点
- 网络：`[512, 256, 128]`，`target_kl=0.015`
- 权重：`checkpoints/task2/`

## Task 3：密林穿越与动态避障（根目录 `forest_*.py`）

```bash
cd /scratch/jiaqi/gym_pybullet
python test_forest_env.py
python train_forest.py --num-envs 10
python eval_forest.py
```

- 权重：`checkpoints/forest_nav/`

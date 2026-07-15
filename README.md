# AME_mjlab

基于 mjlab 复现 [AME-2](https://arxiv.org/abs/2601.08485) (Attention-based Neural Map Encoding) 的敏捷双足/四足越障 locomotion 框架。

**论文：** AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding  
**作者：** Chong Zhang, Victor Klemm, Fan Yang, Marco Hutter (ETH Zurich)  
**项目主页：** https://sites.google.com/leggedrobotics.com/ame-2

## 核心架构

```
Proprioception → ProprioEncoder → proprio_embed ─┐
                                                    ├→ MLP Decoder → actions
Elevation Map → AME-2 Encoder → map_embed ────────┘
  ├ CNN → local features
  ├ MLP + MaxPool → global features
  └ MHA: query(global+proprio) attends to local features
```

- **AME-2 Encoder:** CNN 提取局部特征 → MLP 生成全局特征 → Multi-Head Attention 加权融合
- **MoE Critic (可选):** 8-expert 混合专家网络，论文 Sec IV-B
- **Asymmetric Actor-Critic:** Critic 额外获得 base_lin_vel + 各 link 接触状态

## 环境要求

- Linux
- Python 3.11
- NVIDIA GPU (CUDA 12.4+)
- MuJoCo

## 环境搭建

### 1. 创建 conda 环境

```bash
conda create -n mjlab python=3.11
conda activate mjlab
```

### 2. 安装 PyTorch

根据你的 CUDA 驱动版本选择，驱动 >= 12.6 可用：

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

驱动版本查看：`nvidia-smi | grep "CUDA Version"`

### 3. 安装 mjlab

```bash
pip install mjlab==1.2.0
pip install mujoco-warp==3.9.0.1
pip install warp-lang==1.13.0
```

### 4. 安装本项目

```bash
cd AME_mjlab
pip install -e . --no-deps
pip install -e ./rsl_rl --no-deps
```

### 5. (可选) 打 mjlab 补丁

AMP_mjlab 项目中提供了 history_ordering 补丁：

```bash
cp AMP_mjlab/mjlab_patch/mjlab/managers/observation_manager.py \
  <conda_path>/envs/mjlab/lib/python3.11/site-packages/mjlab/managers/observation_manager.py
```

## 训练

### Teacher Policy（GT elevation map，80k iterations）

```bash
conda activate mjlab
cd AME_mjlab
CUDA_VISIBLE_DEVICES=0 python scripts/train_teacher.py
```

指定 GPU：
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/train_teacher.py
```

快速 smoke test：
```bash
python scripts/train_teacher.py --device cpu --num-envs 1 --max-iterations 1 --log-root /tmp/ame-train-smoke
```

从 checkpoint 恢复到总计 80000 iterations：
```bash
python scripts/train_teacher.py \
  --resume logs/rsl_rl/g1_ame_teacher/<run>/model_15750.pt \
  --max-iterations 80000
```

查看训练好的 teacher：
```bash
python scripts/play_teacher.py \
  --checkpoint logs/rsl_rl/g1_ame_teacher/<run>/model_15750.pt
```

无头服务器先做 rollout 检查：
```bash
python scripts/play_teacher.py \
  --checkpoint logs/rsl_rl/g1_ame_teacher/<run>/model_15750.pt \
  --viewer headless --steps 1000
```

无头录视频：
```bash
python scripts/play_teacher.py \
  --checkpoint logs/rsl_rl/g1_ame_teacher/<run>/model_15750.pt \
  --viewer headless --video --video-length 500
```

可选参数：
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--device` | `cuda:0` | 训练设备 |
| `--num-envs` | 配置默认值 | 覆盖并行环境数 |
| `--max-iterations` | `80000` | 目标总迭代数；resume 时自动只训练剩余部分 |
| `--log-root` | `logs/rsl_rl` | 日志根目录 |
| `--resume` | 无 | 恢复网络、优化器、terrain level 和 curriculum EMA |

### Student Policy（Phase-1，40k iterations）

在线蒸馏 + RL（论文设定）：前 5k iter 关闭 PPO surrogate，同时开 action distillation（log 里 `recon`）与 map embedding alignment（`vq`）。

当前 Phase-1 用 **GT xyz + 启发式 uncertainty** 作为 4 通道 map（neural mapper 后续再接），LSIO 本体感觉 20 步历史、无 base lin-vel。

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/train_student.py \
  --teacher-checkpoint logs/rsl_rl/g1_ame_teacher/<run>/model_XXXXX.pt \
  --num-envs 2048
```

日志：`logs/rsl_rl/g1_ame_student/<timestamp>/`

## 训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--env.scene.num-envs` | 4800 | 并行环境数 |
| `--agent.logger` | tensorboard | 日志类型 |

编辑 `src/tasks/ame_loco/config/g1/rl_cfg.py` 可调整 PPO 参数（learning rate, entropy coef 等）。

## 日志

```bash
tensorboard --logdir logs/rsl_rl --port 6006
```

日志位置：`logs/rsl_rl/g1_ame_teacher/<timestamp>/`

## 项目结构

```
AME_mjlab/
├── scripts/
│   ├── train_teacher.py    # Teacher 训练入口
│   └── train_student.py    # Student 训练入口
├── src/
│   └── tasks/ame_loco/
│       ├── config/g1/      # G1 环境 & RL 配置
│       ├── mdp/            # 奖励、观测、终止、地图
│       └── rl/             # AME-2 Encoder、MoE Critic、Runner
├── rsl_rl/                 # PPO 算法库 (rsl_rl fork)
└── assets/robots/          # G1 机器人模型
```

## PD 增益

使用 Unitree G1 实机参数：

| 关节 | kp | kd | effort_limit |
|---|---|---|---|
| hip_pitch, hip_yaw | 100 | 2 | 88 N·m |
| waist_yaw | 200 | 5 | 88 N·m |
| hip_roll | 100 | 2 | 139 N·m |
| knee | 150 | 4 | 139 N·m |
| shoulders, elbows, wrist_roll | 40 | 1 | 25 N·m |
| waist_pitch, waist_roll | 40 | 5 | 25 N·m |
| ankles | 40 | 2 | 25 N·m |
| wrist_pitch, wrist_yaw | 40 | 1 | 5 N·m |

课程日志中：

- `Metrics/goal/goal_success`：episode 内曾到达目标位置的比例
- `Metrics/goal/goal_pose_success`：同时到达目标位置和朝向的比例
- `Metrics/goal/final_goal_distance`：episode 结束时的目标距离
- `Curriculum/success_ema`：用于 terrain promotion 的成功率 EMA

## TODO

- [ ] Student policy + neural mapping pipeline
- [ ] ONNX 导出

## 致谢

- [Unitree G1 mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

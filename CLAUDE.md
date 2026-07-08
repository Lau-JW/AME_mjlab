# AME_mjlab

基于 mjlab 复现 AME-2: Attention-Based Neural Map Encoding for Agile Legged Locomotion。

## 项目约定

### 目录结构
```
AME_mjlab/
  src/
    tasks/ame_loco/      — AME locomotion 任务
      config/g1/         — G1 机器人配置
      mdp/               — 奖励、观测、事件、终止
      rl/                — runner、wrapper
    assets/
      robots/            — 机器人模型
      motions/           — 运动参考数据
  scripts/               — 训练/回放入口
  rsl_rl/                — RL 算法库（symlink 到 AMP_mjlab）
  cfg/                   — 自定义配置
```

### 规则
- 代码逻辑优先复现论文架构，不做无关的通用化
- 每项配置（奖励系数、网络结构）标注对应论文的章节/表格号
- 所有 elevation map 相关操作用 `src/tasks/ame_loco/mdp/map.py` 封装
- AME-2 encoder 放在 `src/tasks/ame_loco/rl/ame_encoder.py`
- 先训 teacher，再训 student，分两个入口

### 论文参考
论文路径: `getup/AMP_mjlab/2601.08485.pdf`
- AME-2 Encoder: Sec IV-A, Fig 3
- Rewards: Sec IV-D1, Table I
- 网络参数: Appendix C
- 地形参数: Appendix A

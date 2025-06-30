# 电力网络分区强化学习系统

本文档描述了基于项目文档中MDP建模的完整电力网络分区强化学习系统。

## 概述

系统采用"自顶向下优化"范式：
1. **METIS** 提供初始分区
2. **RL智能体** 执行迭代微调以优化分区

## 🔧 核心组件

### 主要模块

- **`environment.py`**: 实现状态空间、动作空间和状态转移的MDP环境
- **`agent.py`**: 具有Actor-Critic网络的PPO智能体，支持两阶段动作选择
- **`state.py`**: 状态管理，包含节点嵌入和边界跟踪
- **`action_space.py`**: 两阶段动作空间，支持掩码约束
- **`reward.py`**: 复合奖励函数（平衡+解耦+内部平衡）
- **`utils.py`**: METIS初始化和分区评估工具
- **`scenario_generator.py`**: 场景生成器，支持N-1故障和负荷波动
- **`gym_wrapper.py`**: OpenAI Gym环境包装器，支持并行训练

### 扩展组件

- **`__init__.py`**: 模块初始化文件，定义公共接口
- **统一训练系统**: 通过 `train.py` 集成所有训练模式

## 📊 状态表示

MDP状态包含：
- **节点特征嵌入 (H)**: 从GAT编码器预计算的静态嵌入
- **节点分配标签 (z_t)**: 动态分区分配
- **边界节点**: 与不同分区节点相邻的节点
- **区域嵌入**: 每个分区的聚合嵌入

## 🎯 动作空间

### 两阶段动作选择

1. **阶段1**: 选择要重新分配的边界节点
2. **阶段2**: 选择目标分区

### 动作掩码

- 只有边界节点可以被选择
- 只有相邻分区可以作为目标
- 确保分区连通性约束

## 🏆 奖励系统

### 复合奖励函数

```python
reward = w1 * balance_reward + w2 * decoupling_reward + w3 * power_reward
```

#### 组件说明

1. **负载平衡奖励**: 基于分区间负载分布的均匀性
2. **电气解耦奖励**: 基于跨分区电气连接的最小化
3. **功率平衡奖励**: 基于分区内功率平衡的优化

详细信息请参考 `docs/REWARD.md`

## 🚀 使用方法

### 基本训练

```python
from code.src.rl import PowerGridPartitioningEnv, PPOAgent

# 创建环境
env = PowerGridPartitioningEnv(hetero_data, config)

# 创建智能体
agent = PPOAgent(env.observation_space, env.action_space, config)

# 训练
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
    agent.update()
```

### 使用统一训练脚本

```bash
# 快速训练（默认设置）
python train.py --mode fast

# 完整训练
python train.py --mode full

# IEEE 118节点大规模训练
python train.py --mode ieee118
```

### 配置管理

使用 `config.yaml` 配置：
- 数据加载和预处理
- 环境参数
- GAT编码器设置
- PPO超参数
- 训练设置
- 并行训练参数
- 场景生成配置

## ✨ 关键特性

### 物理引导学习
- 在注意力机制中融入电气阻抗
- 基于电力系统目标的奖励函数
- 通过动作掩码强制物理约束

### 可扩展架构
- 异构图表示
- 高效的边界节点跟踪
- 增量状态更新
- 支持多种电力系统规模

### 鲁棒训练
- 带动作掩码的PPO
- 全面的日志记录和检查点
- 场景生成支持
- 课程学习支持
- 评估指标

## 🎛️ 训练模式

### 1. Fast模式（推荐日常使用）
```yaml
training:
  episodes: 1500
  parallel_training: false
  scenario_generation: true
```

### 2. Full模式（高质量结果）
```yaml
training:
  episodes: 5000
  parallel_training: true
  scenario_generation: true
```

### 3. IEEE118模式（大规模训练）
```yaml
training:
  episodes: 3000
  parallel_training: true
  scenario_generation: true
  case_name: "ieee118"
```

## 📈 性能监控

### 训练指标
- Episode奖励
- 分区质量指标
- 收敛速度
- 动作选择分布

### 评估指标
- 负载平衡度
- 电气解耦度
- 连通性
- 计算效率

## 🔧 高级功能

### 并行训练
```yaml
parallel_training:
  enabled: true
  num_cpus: 12
  sb3_ppo_params:
    n_steps: 2048
    batch_size: 64
```

### 场景生成
```yaml
scenario_generation:
  enabled: true
  perturb_prob: 0.8
  n1_contingency: true
  load_variation: true
```

### 课程学习
```yaml
curriculum_learning:
  enabled: true
  start_partitions: 2
  max_partitions: 5
  progression_episodes: 500
```

## 📋 依赖要求

### 必需依赖
```bash
pip install torch torch-geometric numpy scipy scikit-learn pandas pyyaml matplotlib
```

### 可选依赖
```bash
# 并行训练支持
pip install stable-baselines3[extra]

# 可视化增强
pip install plotly tensorboard

# 性能优化
pip install metis  # 或使用conda install -c conda-forge metis
```

## 🎉 完整工作流程

```bash
# 1. 快速体验
python train.py --mode fast

# 2. 完整训练
python train.py --mode full

# 3. 大规模训练
python train.py --mode ieee118

# 4. 模型评估
python test.py --mode full
```

## 🔍 调试和分析

### 日志分析
- 训练日志位于 `logs/` 目录
- 包含详细的训练指标和性能数据
- 支持TensorBoard可视化

### 检查点管理
- 模型检查点保存在 `checkpoints/` 目录
- 支持断点续训
- 包含完整的训练状态信息

### 性能分析
- 内存使用监控
- GPU利用率跟踪
- 训练时间分析

## ⚠️ 注意事项

1. **内存管理**: 大规模网络训练时注意内存使用
2. **GPU支持**: 自动检测并使用GPU加速
3. **数值稳定性**: 包含全面的NaN/inf保护
4. **随机种子**: 支持设置随机种子确保可重现性
5. **异常处理**: 完整的错误处理和恢复机制

## 🎯 最佳实践

1. **模型选择**: 根据网络规模选择合适的模型配置
2. **超参数调优**: 使用验证集进行超参数优化
3. **训练策略**: 结合课程学习和场景生成提高训练效果
4. **性能监控**: 定期检查训练指标和模型性能
5. **结果验证**: 使用多个随机种子验证结果稳定性

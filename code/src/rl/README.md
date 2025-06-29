# 电力网络分区强化学习模块

本模块实现了基于项目文档中MDP建模的完整电力网络分区强化学习系统。

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
- **统一训练系统**: 通过 `train_unified.py` 集成所有训练模式

## 📊 状态表示

MDP状态包含：
- **节点特征嵌入 (H)**: 从GAT编码器预计算的静态嵌入
- **节点分配标签 (z_t)**: 动态分区分配
- **边界节点**: 与不同分区节点相邻的节点
- **区域嵌入**: 每个分区的聚合嵌入

## 🎯 动作空间

两阶段动作选择：
1. **节点选择**: 选择要移动的边界节点
2. **分区选择**: 从相邻分区中选择目标分区

动作掩码强制执行：
- 只有边界节点可以移动
- 只允许移动到相邻分区
- 连通性约束（可选）

## 🏆 奖励函数

包含三个组件的复合奖励：
- **R_balance**: 分区负载平衡 (-Var(L₁, ..., Lₖ))
- **R_decoupling**: 电气解耦 (-Σ|Y_uv| 对于耦合边)
- **R_internal_balance**: 内部功率平衡 (-Σ(P_gen - P_load)²)

## 🚀 使用方法

### 基本训练

```python
from src.rl import PowerGridPartitioningEnv, PPOAgent

# 创建环境
env = PowerGridPartitioningEnv(
    hetero_data=hetero_data,
    node_embeddings=node_embeddings,
    num_partitions=3
)

# 创建智能体
agent = PPOAgent(
    node_embedding_dim=128,
    region_embedding_dim=256,
    num_partitions=3
)

# 训练（通过统一训练系统）
python train_unified.py --mode standard --case ieee14 --partitions 3
```

### 使用统一训练脚本

```bash
# 快速训练（默认设置）
python train_unified.py --mode quick --case ieee14 --episodes 100 --partitions 3

# 使用自定义配置训练
python train_unified.py --config config_unified.yaml

# IEEE 118节点大规模训练
python train_unified.py --mode ieee118

# 并行训练
python train_unified.py --mode parallel --episodes 2000

# 课程学习训练
python train_unified.py --mode curriculum
```

### 配置管理

使用 `config_unified.yaml` 配置：
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

### 系统集成
- 与现有数据处理流水线兼容
- 使用预训练的GAT嵌入
- 支持多种电力网络格式
- 统一的训练入口

## 🧪 验证

系统集成验证：
```bash
# 验证完整训练流水线
python main.py --mode quick

# 验证并行训练功能
python main.py --mode parallel

# 验证增强奖励系统
python -c "from code.src.rl.ab_testing import create_standard_ab_test; framework = create_standard_ab_test(); framework.run_all_experiments()"
```

这些命令验证了整个系统的端到端功能。

## 🎛️ 场景生成

### N-1故障模拟
```python
from src.rl.scenario_generator import ScenarioGenerator

generator = ScenarioGenerator(base_case_data)

# 生成随机N-1故障
perturbed_case = generator.generate_random_scene()

# 应用特定故障
case_n1 = generator.apply_specific_contingency(base_case_data, branch_idx=10)
```

### 负荷波动
```python
# 应用负荷缩放
case_scaled = generator.apply_load_scaling(base_case_data, scale_factor=1.2)

# 批量生成场景
scenarios = generator.generate_batch_scenarios(num_scenarios=100)
```

## 🌐 并行训练

### OpenAI Gym包装器
```python
from src.rl.gym_wrapper import make_parallel_env

# 创建并行环境
parallel_env = make_parallel_env(
    base_case_data=case_data,
    config=config,
    num_envs=12,
    use_scenario_generator=True
)
```

### 使用Stable-Baselines3
```bash
# 安装依赖
pip install stable-baselines3[extra]

# 运行并行训练
python train_unified.py --mode parallel --episodes 5000
```

## ⚡ 性能优化建议

1. **GPU使用**: 在配置中设置 `device: cuda` 启用GPU加速
2. **批量更新**: 增加 `update_interval` 以获得更稳定的训练
3. **内存优化**: 对大型电网使用较小的嵌入维度
4. **收敛监控**: 监控奖励曲线并调整学习率

## 🔧 故障排除

### 常见问题

1. **无有效动作**: 检查边界节点计算和动作掩码
2. **训练不稳定**: 降低学习率或增加裁剪
3. **内存问题**: 减少批大小或嵌入维度
4. **训练缓慢**: 启用GPU或减少网络复杂度

### 调试模式

启用详细日志记录：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 系统检查

```bash
# 检查系统状态
python main.py --mode quick --episodes 10

# 查看可用配置
python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"

# 运行快速验证
python main.py --mode quick
```

## 🔌 扩展点

模块化设计允许轻松扩展：

- **自定义奖励函数**: 修改 `reward.py`
- **高级动作空间**: 扩展 `action_space.py`
- **不同算法**: 在 `agent.py` 中替换PPO
- **课程学习**: 通过统一训练系统实现
- **新场景类型**: 扩展 `scenario_generator.py`

## 📈 训练模式

| 模式 | 描述 | 用途 | 特色功能 |
|------|------|------|----------|
| `quick` | 快速测试 | 功能验证 | 100回合，快速收敛 |
| `standard` | 标准训练 | 常规研究 | 1000回合，平衡性能 |
| `full` | 完整训练 | 深度研究 | 2000回合，高质量结果 |
| `ieee118` | 大规模训练 | 复杂系统 | 自动启用并行+场景生成 |
| `parallel` | 并行训练 | 高效训练 | 多进程，3-5倍加速 |
| `curriculum` | 课程学习 | 渐进训练 | 分区数递增，稳定收敛 |

## 📚 配置示例

### 快速测试配置
```yaml
training:
  mode: quick
  num_episodes: 100
  max_steps_per_episode: 50

environment:
  num_partitions: 3
  
agent:
  lr_actor: 3e-4
  lr_critic: 1e-3
```

### 大规模训练配置
```yaml
data:
  case_name: ieee118

environment:
  num_partitions: 8
  max_steps: 500

parallel_training:
  enabled: true
  num_cpus: 12

scenario_generation:
  enabled: true
  perturb_prob: 0.8
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
python main.py --mode quick

# 2. 标准训练
python main.py --mode standard --case ieee30 --partitions 5

# 3. 大规模训练
python main.py --mode ieee118

# 4. 并行训练
python main.py --mode parallel

# 5. 课程学习
python main.py --mode curriculum
```



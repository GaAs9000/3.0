# 电力网络分区强化学习系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于深度强化学习(PPO)和异构图神经网络(GAT)的电力网络分区优化系统，专为电力系统运行分析和优化而设计。

## 🎯 项目概述

本项目实现了一个完整的电力网络分区解决方案，将电力网络分区问题建模为马尔可夫决策过程(MDP)，使用PPO算法和物理增强的异构图神经网络进行优化。

### 核心特性

- **🧠 异构图神经网络**: 基于GATv2的物理增强图编码器，融入电气阻抗先验
- **🎮 强化学习**: PPO算法实现的两阶段动作空间(节点选择+分区选择)
- **🎯 三阶段增强奖励**: 稠密奖励、智能探索、自适应物理约束的渐进式优化
- **⚡ 物理约束**: 集成电力系统物理约束和拓扑优化
- **📊 多种基线**: 谱聚类、K-means、随机分区等基线方法对比
- **🔄 场景生成**: N-1故障和负荷波动模拟
- **🏃‍♂️ 并行训练**: 多进程训练加速
- **📈 完整监控**: TensorBoard集成、实时可视化、训练统计

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装步骤

1. **克隆项目**
```bash
git clone <your-repo-url>
cd 2.0
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

1. **验证安装**
```bash
python main.py --check-deps
```

### 立即开始训练

```bash
# 快速测试（5分钟）
python main.py --mode quick

# 标准训练（30分钟）
python main.py --mode standard

# 增强奖励训练（推荐）
python main.py --config enhanced_rewards_training

# 完整训练（2-4小时）
python main.py --mode full

# IEEE 118节点大规模训练
python main.py --mode ieee118
```

## 📋 训练模式详解

### 训练模式对比

| 模式 | 训练时间 | 回合数 | 适用场景 | 特色功能 |
|------|----------|--------|----------|----------|
| `quick` | 5分钟 | 100 | 功能验证、快速测试 | 轻量化配置 |
| `standard` | 30分钟 | 1000 | 日常研究、算法验证 | 平衡性能与时间 |
| `enhanced_rewards_training` | 45分钟 | 1500 | **推荐模式**、高质量训练 | 三阶段增强奖励 |
| `full` | 2-4小时 | 5000 | 高质量结果、论文实验 | 完整训练流程 |
| `ieee118` | 4-8小时 | 3000 | 大规模系统测试 | 并行+场景生成 |
| `parallel` | 可配置 | 可配置 | 多核加速训练 | 多进程并行 |
| `curriculum` | 可配置 | 可配置 | 渐进式学习 | 难度递增训练 |

### 三阶段增强奖励系统

本项目实现了创新的三阶段增强奖励系统，显著提升训练效果：

```bash
# 第一阶段：稠密奖励（解决稀疏奖励问题）
python main.py --config stage1_dense_rewards

# 第二阶段：智能探索（平衡探索与利用）
python main.py --config stage2_smart_exploration

# 第三阶段：自适应物理约束（融入领域知识）
python main.py --config stage3_adaptive_physics

# 完整三阶段训练（推荐）
python main.py --config enhanced_rewards_training
```

#### 🎯 三阶段设计理念

1. **第一阶段 - 稠密奖励**
   - **局部连通性奖励**: 提供即时拓扑反馈，引导智能体构建连通分区
   - **增量平衡奖励**: 实时监控负载平衡改善，避免稀疏奖励问题
   - **边界压缩奖励**: 激励减少分区间耦合，提供明确的进度信号

2. **第二阶段 - 智能探索**
   - **探索奖励**: 鼓励访问未探索的状态空间，防止过早收敛
   - **势函数塑造**: 基于电力系统拓扑的长期引导信号
   - **平衡机制**: 动态调节探索与利用的权重

3. **第三阶段 - 自适应物理约束**
   - **邻居一致性**: 融入电力系统的物理约束知识
   - **自适应权重**: 根据训练进度动态调整奖励组件权重
   - **领域知识**: 集成电力系统专家经验

### 自定义训练参数

```bash
# 指定电力系统算例
python main.py --mode standard --case ieee30 --partitions 5

# 调整训练参数
python main.py --mode standard --episodes 2000 --lr 0.001

# 启用并行训练
python main.py --mode parallel --workers 8

# 保存训练结果
python main.py --mode standard --save-results --output-dir my_experiment
```

## 🔧 主要参数配置

### 训练参数

```python
# 在main.py中的配置示例
training_config = {
    'episodes': 1000,           # 训练回合数
    'max_steps': 200,          # 每回合最大步数
    'learning_rate': 3e-4,     # 学习率
    'batch_size': 32,          # 批大小
    'update_interval': 10,     # 更新间隔
    'gamma': 0.99,             # 折扣因子
    'eps_clip': 0.2,           # PPO裁剪参数
    'entropy_coef': 0.01,      # 熵系数
}
```

### 网络参数

```python
model_config = {
    'hidden_channels': 64,     # 隐藏层维度
    'gnn_layers': 3,          # GNN层数
    'heads': 4,               # 注意力头数
    'dropout': 0.3,           # Dropout率
    'output_dim': 64,         # 输出嵌入维度
}
```

### 环境参数

```python
env_config = {
    'case_name': 'ieee30',     # 电力系统算例
    'num_partitions': 4,       # 目标分区数
    'reward_weights': {        # 基础奖励权重
        'load_balance': 0.4,
        'electrical_decoupling': 0.4,
        'power_balance': 0.2,

        # 增强奖励系统配置
        'use_enhanced_rewards': True,
        'enhanced_config': {
            'enable_dense_rewards': True,      # 启用稠密奖励
            'enable_exploration_bonus': True,  # 启用探索奖励
            'enable_potential_shaping': True,  # 启用势函数塑造
            'enable_adaptive_weights': True,   # 启用自适应权重
        },

        # 稠密奖励权重
        'local_connectivity': 0.4,     # 局部连通性
        'incremental_balance': 0.3,    # 增量平衡
        'boundary_compression': 0.3,   # 边界压缩

        # 探索与塑造权重
        'exploration_bonus': 0.1,      # 探索奖励
        'potential_shaping': 0.2,      # 势函数塑造

        # 物理约束权重
        'neighbor_consistency': 0.15,  # 邻居一致性
    }
}
```

## 📊 训练监控

### 启动训练监控

```bash
# 启动训练（自动开启监控）
python main.py --mode standard --save-results

# 实时监控训练进展（TensorBoard）
tensorboard --logdir=logs
# 访问 http://localhost:6006

# 可选：使用Weights & Biases (W&B)
# 1. 登录W&B
wandb login

# 2. 启动训练时启用W&B
python main.py --mode standard --use-wandb
```

### 监控指标

- **训练指标**: 奖励曲线、回合长度、成功率
- **性能指标**: 负载变异系数、电气耦合度、连通性
- **网络指标**: Actor/Critic损失、策略熵、梯度范数
- **系统指标**: 内存使用、GPU利用率、训练速度
- **增强奖励指标**: 奖励组件分解、局部连通性、增量平衡、边界压缩

## 🎯 增强奖励系统详解

### 系统架构

增强奖励系统采用三阶段渐进式设计，解决传统强化学习在电力网络分区中的关键问题：

#### 🔍 问题分析
- **稀疏奖励**: 传统方法只在分区完成时给予奖励，训练效率低
- **探索困难**: 动作空间大，智能体难以有效探索
- **领域知识缺失**: 缺乏电力系统专业知识的融入

#### 💡 解决方案

**第一阶段 - 稠密奖励 (Dense Rewards)**
```python
# 局部连通性奖励：即时拓扑反馈
local_connectivity_reward = connectivity_improvement * 0.4

# 增量平衡奖励：实时负载平衡监控
incremental_balance_reward = balance_improvement * 0.3

# 边界压缩奖励：减少分区间耦合
boundary_compression_reward = coupling_reduction * 0.3
```

**第二阶段 - 智能探索 (Smart Exploration)**
```python
# 探索奖励：鼓励状态空间探索
exploration_bonus = novelty_score * 0.1

# 势函数塑造：基于拓扑的长期引导
potential_shaping = topology_potential_diff * 0.2
```

**第三阶段 - 自适应物理约束 (Adaptive Physics)**
```python
# 邻居一致性：融入电力系统物理约束
neighbor_consistency = physical_constraint_satisfaction * 0.15

# 自适应权重：根据训练进度动态调整
adaptive_weights = adjust_weights_by_episode(episode_count)
```

### 配置与使用

```yaml
# config.yaml中的增强奖励配置
environment:
  reward_weights:
    use_enhanced_rewards: true
    enhanced_config:
      enable_dense_rewards: true      # 第一阶段
      enable_exploration_bonus: true  # 第二阶段
      enable_potential_shaping: true  # 第二阶段
      enable_adaptive_weights: true   # 第三阶段
```

## 🔬 基线方法对比

项目内置多种基线方法用于性能对比：

```bash
# 训练完成后自动运行基线对比
python main.py --mode standard --save-results

# 手动运行基线对比
from code.baseline import run_baseline_comparison
comparison_results = run_baseline_comparison(env, agent)
```

### 支持的基线方法

- **谱聚类**: 基于图拉普拉斯矩阵的聚类
- **K-means**: 基于节点嵌入的K-means聚类
- **随机分区**: 随机分配基准
- **METIS**: 图分区优化算法(可选)

## 🏗️ 项目架构

```
├── main.py                   # 主训练入口（统一所有训练模式）
├── config.yaml              # 统一配置文件（包含增强奖励配置）
├── code/                     # 核心代码
│   ├── src/                  # 源代码
│   │   ├── data_processing.py    # 数据处理
│   │   ├── gat.py               # GAT编码器
│   │   ├── visualization.py     # 可视化
│   │   └── rl/                  # 强化学习模块
│   │       ├── environment.py       # 环境定义
│   │       ├── agent.py             # PPO智能体
│   │       ├── action_space.py      # 动作空间
│   │       ├── state.py             # 状态管理
│   │       ├── reward.py            # 奖励函数（含增强奖励系统）
│   │       ├── reward_analyzer.py   # 奖励分析工具
│   │       ├── gym_wrapper.py       # Gym环境包装器
│   │       ├── ppo_trainer.py       # PPO训练器
│   │       └── utils.py             # 工具函数
│   ├── baseline/             # 基线方法
│   │   ├── spectral_clustering.py
│   │   ├── kmeans_clustering.py
│   │   ├── evaluator.py
│   │   └── comparison.py
│   └── clean.py              # 缓存清理工具
├── cache/                    # 缓存文件
├── logs/                     # 训练日志
├── checkpoints/              # 模型检查点
├── models/                   # 保存的模型
├── experiments/              # 实验结果
└── output/                   # 输出文件
    └── figures/              # 生成的图表
```

## 📈 使用示例

### 基础训练流程

```python
# 1. 创建训练系统
from main import UnifiedTrainingSystem
trainer = UnifiedTrainingSystem()

# 2. 运行训练
results = trainer.run_training(mode='standard')

# 3. 查看结果
print(f"最终奖励: {results['final_reward']}")
print(f"训练时间: {results['training_time']}")
```

### 增强奖励训练

```python
# 使用增强奖励系统训练
from main import UnifiedTrainingSystem
trainer = UnifiedTrainingSystem()

# 第一阶段：稠密奖励训练
stage1_results = trainer.run_training(config='stage1_dense_rewards')

# 第二阶段：智能探索训练
stage2_results = trainer.run_training(config='stage2_smart_exploration')

# 第三阶段：自适应物理约束训练
stage3_results = trainer.run_training(config='stage3_adaptive_physics')

# 完整增强奖励训练
enhanced_results = trainer.run_training(config='enhanced_rewards_training')
```

### 自定义训练

```python
# 自定义配置
custom_config = {
    'environment': {
        'case_name': 'ieee57',
        'num_partitions': 6,
        'max_steps': 300,
        'reward_weights': {
            'use_enhanced_rewards': True,
            'enhanced_config': {
                'enable_dense_rewards': True,
                'enable_exploration_bonus': False,
                'enable_potential_shaping': False,
                'enable_adaptive_weights': False,
            }
        }
    },
    'training': {
        'episodes': 2000,
        'learning_rate': 1e-3,
        'batch_size': 64
    }
}

# 运行自定义训练
results = trainer.run_training(mode='custom', config=custom_config)
```

### W&B (Weights & Biases) 集成

除了TensorBoard，本项目也支持使用[Weights & Biases](https://wandb.ai)进行更强大的实验跟踪和可视化。

#### 启用W&B

1.  **安装W&B**
    
    `wandb`已在`requirements.txt`中列为可选依赖。如果尚未安装，请运行：
    ```bash
    pip install wandb
    ```

2.  **登录账户**

    在您的终端中运行以下命令，并根据提示输入您的API密钥（可在W&B官网的用户设置中找到）。
    ```bash
    wandb login
    ```

3.  **启动训练**

    在运行`main.py`时，添加`--use-wandb`标志以启用W&B日志记录。
    ```bash
    python main.py --mode standard --use-wandb
    ```
    您也可以在`config.yaml`中将其设置为默认启用：
    ```yaml
    # config.yaml
    training:
      use_wandb: true
    ```

#### W&B特性

启用后，W&B将自动记录：
-   所有TensorBoard中的指标（奖励、损失、性能等）。
-   超参数配置，方便对比不同实验。
-   系统硬件指标（GPU/CPU利用率、温度等）。
-   保存的模型文件和代码版本。

您可以在W&B的Web界面上查看和比较所有实验运行，生成更丰富的图表。

### 结果分析

```python
# 加载训练结果
import pickle
with open('experiments/results.pkl', 'rb') as f:
    results = pickle.load(f)

# 分析训练曲线
import matplotlib.pyplot as plt
plt.plot(results['reward_history'])
plt.title('Training Reward Curve')
plt.show()

# 增强奖励组件分析
if 'reward_components' in results:
    components = results['reward_components']

    # 绘制奖励组件演化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(components['local_connectivity'])
    axes[0,0].set_title('Local Connectivity Rewards')

    axes[0,1].plot(components['incremental_balance'])
    axes[0,1].set_title('Incremental Balance Rewards')

    axes[1,0].plot(components['boundary_compression'])
    axes[1,0].set_title('Boundary Compression Rewards')

    axes[1,1].plot(components['total_reward'])
    axes[1,1].set_title('Total Reward')

    plt.tight_layout()
    plt.show()

# 基线方法对比
comparison_df = results['baseline_comparison']
print(comparison_df)
```

## 🛠️ 故障排除

### 常见问题

1. **CUDA内存不足**
```bash
# 减少批大小或使用CPU
python main.py --mode quick --device cpu
```

2. **依赖安装问题**
```bash
# 升级pip和setuptools
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

3. **训练过程中断**
```bash
# 从检查点恢复训练
python main.py --mode standard --resume --checkpoint checkpoints/latest.pt
```

### 性能优化

- **GPU加速**: 确保安装正确的CUDA和PyTorch版本
- **并行训练**: 使用`--mode parallel`启用多进程训练
- **内存优化**: 调整批大小和缓存设置
- **数据预处理**: 预计算特征和缓存网络结构

## 📚 进阶使用

### 添加新的电力系统算例

1. 准备MATPOWER格式的.m文件
2. 放置在适当的数据目录
3. 在配置中指定新算例名称

### 自定义奖励函数

```python
# 在code/src/rl/reward.py中修改
class CustomRewardFunction(RewardFunction):
    def compute_reward(self, current_partition, boundary_nodes, action):
        # 自定义奖励逻辑
        pass

# 使用增强奖励函数
class MyEnhancedRewardFunction(EnhancedRewardFunction):
    def compute_custom_component(self, current_partition, boundary_nodes, action):
        # 添加自定义奖励组件
        custom_reward = 0.0
        # 实现自定义逻辑
        return custom_reward

    def compute_reward(self, current_partition, boundary_nodes, action):
        # 调用基础增强奖励
        base_reward = super().compute_reward(current_partition, boundary_nodes, action)

        # 添加自定义组件
        custom_component = self.compute_custom_component(current_partition, boundary_nodes, action)

        return base_reward + 0.1 * custom_component  # 自定义权重
```

### 扩展基线方法

```python
# 在code/baseline/中添加新方法
class MyPartitioner(BasePartitioner):
    def partition(self, env):
        # 实现自定义分区算法
        pass
```

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/yourrepo/issues)
- 邮箱: your.email@example.com

## 🙏 致谢

- PyTorch团队提供的深度学习框架
- PyTorch Geometric团队提供的图神经网络库
- MATPOWER团队提供的电力系统数据格式
- 开源社区的贡献和支持

---

**开始您的电力网络分区优化之旅吧！** 🚀 
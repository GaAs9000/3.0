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

**选项A：完整安装（推荐）**
```bash
pip install -r requirements.txt
```

**选项B：最小安装（快速开始）**
```bash
# 1. 首先安装PyTorch（根据你的CUDA版本）
# CPU版本
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 或GPU版本（CUDA 11.8）
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 2. 安装最小依赖
pip install -r requirements-minimal.txt
```

3. **验证安装**
```bash
python main.py --check-deps
```

### 立即开始训练

```bash
# 快速测试（5分钟）
python main.py --mode quick

# 标准训练（30分钟）
python main.py --mode standard

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
| `full` | 2-4小时 | 5000 | 高质量结果、论文实验 | 完整训练流程 |
| `ieee118` | 4-8小时 | 3000 | 大规模系统测试 | 并行+场景生成 |
| `parallel` | 可配置 | 可配置 | 多核加速训练 | 多进程并行 |
| `curriculum` | 可配置 | 可配置 | 渐进式学习 | 难度递增训练 |
| `demo` | 2分钟 | 50 | 演示展示 | 可视化重点 |

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
    'reward_weights': {        # 奖励权重
        'load_balance': 0.4,
        'electrical_decoupling': 0.4,
        'power_balance': 0.2
    }
}
```

## 📊 训练监控

### 启动训练监控

```bash
# 启动训练（自动开启监控）
python main.py --mode standard --save-results

# 实时监控训练进展
python monitor_training.py

# TensorBoard可视化
tensorboard --logdir=logs
# 访问 http://localhost:6006
```

### 监控指标

- **训练指标**: 奖励曲线、回合长度、成功率
- **性能指标**: 负载变异系数、电气耦合度、连通性
- **网络指标**: Actor/Critic损失、策略熵、梯度范数
- **系统指标**: 内存使用、GPU利用率、训练速度

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
├── main.py                   # 主训练入口
├── config.yaml              # 配置文件
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
│   │       ├── reward.py            # 奖励函数
│   │       └── ...
│   └── baseline/             # 基线方法
│       ├── spectral_clustering.py
│       ├── kmeans_clustering.py
│       └── ...
├── cache/                    # 缓存文件
├── logs/                     # 训练日志
├── checkpoints/              # 模型检查点
├── experiments/              # 实验结果
└── output/                   # 输出文件
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

### 自定义训练

```python
# 自定义配置
custom_config = {
    'environment': {
        'case_name': 'ieee57',
        'num_partitions': 6,
        'max_steps': 300
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
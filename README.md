# 电力网络分区强化学习系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org)

一个基于强化学习的电力网络分区优化系统，整合了异构图神经网络、物理约束和多种训练模式。

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 立即开始训练
```bash
# 快速测试（5分钟）
python train_unified.py --mode quick

# 标准训练（30分钟）
python train_unified.py --mode standard

# 大规模训练（IEEE 118节点）
python train_unified.py --mode ieee118
```

### 系统检查
```bash
# 检查依赖
python train_unified.py --check-deps

# 运行演示
python train_unified.py --mode demo
```

## 📊 训练模式

| 模式 | 用途 | 时间 | 特色 |
|------|------|------|------|
| `quick` | 功能测试 | 5分钟 | 快速验证 |
| `standard` | 常规研究 | 30分钟 | 标准配置 |
| `full` | 深度训练 | 2-4小时 | 高质量结果 |
| `ieee118` | 大规模系统 | 4-8小时 | 并行+场景生成 |
| `parallel` | 高效训练 | 可配置 | 多进程加速 |
| `curriculum` | 渐进学习 | 可配置 | 难度递增 |

## 🔧 核心特性

- **统一入口** - 单一脚本支持所有训练模式
- **异构图神经网络** - 物理增强的GAT编码器
- **强化学习环境** - 专为电力系统设计的MDP
- **多种基线方法** - 谱聚类、K-means、随机分区对比
- **场景生成** - N-1故障和负荷波动模拟
- **并行训练** - 多进程加速训练

## 📁 项目结构

```
├── train_unified.py      # 统一训练入口
├── config_unified.yaml   # 统一配置文件
├── src/                  # 核心源码
│   ├── data_processing.py    # 数据处理
│   ├── gat.py               # GAT编码器
│   ├── visualization.py     # 可视化
│   └── rl/                  # 强化学习模块
├── baseline/             # 基线方法
├── test/                 # 测试脚本
└── docs/                 # 详细文档
```

## 🎯 使用示例

```bash
# 自定义配置训练
python train_unified.py --mode standard --case ieee30 --partitions 5

# 并行训练
python train_unified.py --mode parallel --episodes 2000

# 课程学习
python train_unified.py --mode curriculum

# 保存结果
python train_unified.py --mode standard --save-results
```

## 📈 训练过程监控

系统提供完整的训练监控功能，帮助您实时观测训练进展：

### 启动训练（自动启用监控）
```bash
# 标准训练（已集成TensorBoard、检查点、可视化）
python train_unified.py --mode standard --save-results
```

### 实时监控工具
```bash
# 实时图表监控（6个子图显示关键指标）
python monitor_training.py

# 快速检查训练进度
python check_training_progress.py

# 持续监控模式（每10秒更新）
python check_training_progress.py --watch --interval 10

# TensorBoard详细分析
tensorboard --logdir=logs
# 访问 http://localhost:6006
```

### 监控指标
- **训练指标**: 奖励曲线、回合长度、成功率
- **性能指标**: 负载变异系数、电气耦合度
- **训练损失**: Actor/Critic损失、策略熵
- **统计信息**: 最佳奖励、移动平均、训练时间

### 输出文件
- `logs/` - TensorBoard日志文件
- `checkpoints/` - 训练中间检查点
- `experiments/` - 最终结果和报告
- `figures/` - 训练曲线和分区可视化图

详细监控使用说明请参考：[训练监控指南](TRAINING_MONITORING_GUIDE.md)

## 📖 详细文档

- **[技术文档](docs/TECHNICAL_GUIDE.md)** - src目录各模块详细说明

## 📊 支持的电力系统

- IEEE 14/30/57/118节点系统
- 自定义MATPOWER格式文件

## 🎉 开始使用

```bash
# 一键体验
python train_unified.py --mode quick
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件 
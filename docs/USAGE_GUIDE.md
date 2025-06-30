# 使用指南

本文档提供了电力网络分区强化学习系统的详细使用指南，包括安装、配置、训练和评估的完整流程。

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **操作系统**: Windows/Linux/macOS
- **硬件**: 建议8GB+ RAM，GPU可选但推荐

### 安装依赖

```bash
# 基础依赖
pip install torch torch-geometric numpy matplotlib pyyaml tqdm

# 可选依赖（推荐安装）
pip install pandapower networkx plotly seaborn stable-baselines3
```

### 立即开始

```bash
# 克隆项目（如果需要）
git clone <repository-url>
cd power-grid-partitioning

# 快速训练（推荐新手）
python train.py --mode fast

# 查看训练结果
python test.py --mode full
```

## 📋 详细使用流程

### 1. 系统配置

#### 配置文件说明

主配置文件 `config.yaml` 包含所有系统参数：

```yaml
# 数据配置
data:
  case_name: "ieee14"          # 电力网络案例
  num_partitions: 3            # 分区数量
  cache_enabled: true          # 启用数据缓存

# 训练配置
training:
  mode: "fast"                 # 训练模式
  episodes: 1500               # 训练轮数
  save_interval: 100           # 保存间隔

# 奖励系统配置
environment:
  reward_weights:
    reward_mode: "dual_layer"  # 奖励模式
    balance_weight: 1.0        # 负载平衡权重
    decoupling_weight: 1.0     # 电气解耦权重
    power_weight: 1.0          # 功率平衡权重
```

#### 自定义配置

```bash
# 使用自定义配置文件
python train.py --config my_config.yaml

# 通过命令行覆盖配置
python train.py --mode full --episodes 3000 --case ieee30
```

### 2. 训练模式详解

#### Fast模式（推荐日常使用）

```bash
python train.py --mode fast
```

**特点**:
- 训练轮数: 1500
- 无并行训练
- 启用场景生成
- 训练时间: 30-60分钟

**适用场景**: 日常开发、快速验证、参数调试

#### Full模式（高质量结果）

```bash
python train.py --mode full
```

**特点**:
- 训练轮数: 5000
- 启用并行训练
- 启用场景生成
- 训练时间: 2-4小时

**适用场景**: 最终结果、论文实验、性能基准

#### IEEE118模式（大规模训练）

```bash
python train.py --mode ieee118
```

**特点**:
- 训练轮数: 3000
- 启用并行训练
- 启用场景生成
- 大规模网络: IEEE 118节点
- 训练时间: 1-3小时

**适用场景**: 大规模网络、可扩展性测试

### 3. 高级训练选项

#### 并行训练

```bash
# 启用并行训练
python train.py --mode full --parallel

# 指定CPU核心数
python train.py --mode full --parallel --num-cpus 8
```

#### 场景生成

```bash
# 启用场景生成（默认）
python train.py --mode fast --scenario-generation

# 自定义扰动概率
python train.py --mode fast --perturb-prob 0.6
```

#### 断点续训

```bash
# 从检查点继续训练
python train.py --mode full --resume --checkpoint checkpoints/latest.pt

# 指定特定检查点
python train.py --mode full --resume --checkpoint checkpoints/episode_1000.pt
```

### 4. 模型评估

#### 基础评估

```bash
# 完整评估（包含基线对比）
python test.py --mode full

# 快速评估
python test.py --mode quick

# 指定模型文件
python test.py --model checkpoints/best_model.pt
```

#### 高级评估选项

```bash
# 生成详细报告
python test.py --mode full --detailed-report

# 保存结果到指定目录
python test.py --mode full --output-dir my_results

# 可视化分析
python test.py --mode full --visualize --save-figures
```

### 5. 结果分析

#### 训练日志

训练日志保存在 `logs/` 目录：

```
logs/
├── training_20240101_120000/
│   ├── config.yaml          # 训练配置
│   ├── training.log         # 训练日志
│   ├── metrics.json         # 训练指标
│   └── tensorboard/         # TensorBoard日志
```

#### 模型检查点

模型检查点保存在 `checkpoints/` 目录：

```
checkpoints/
├── latest.pt                # 最新模型
├── best_model.pt           # 最佳模型
├── episode_1000.pt         # 定期保存
└── training_stats.json     # 训练统计
```

#### 评估结果

评估结果保存在 `test_results/` 目录：

```
test_results/
├── evaluation_report.html  # 评估报告
├── metrics_comparison.csv  # 指标对比
├── figures/                # 可视化图表
└── partition_results.json  # 分区结果
```

## 🔧 常用命令参考

### 训练命令

```bash
# 基础训练
python train.py --mode fast
python train.py --mode full
python train.py --mode ieee118

# 自定义参数
python train.py --mode fast --episodes 2000 --case ieee30
python train.py --mode full --parallel --num-cpus 12
python train.py --config custom_config.yaml

# 断点续训
python train.py --resume --checkpoint checkpoints/latest.pt
```

### 评估命令

```bash
# 基础评估
python test.py --mode full
python test.py --mode quick

# 自定义评估
python test.py --model checkpoints/best_model.pt
python test.py --mode full --output-dir results_20240101
python test.py --mode full --visualize --save-figures
```

### 工具命令

```bash
# 检查依赖
python train.py --check-deps

# 清理缓存
python code/clean.py --cache --logs

# 查看系统信息
python train.py --system-info
```

## 📊 性能优化建议

### 硬件优化

1. **GPU使用**: 系统自动检测GPU，建议使用CUDA兼容显卡
2. **内存配置**: 大规模网络建议16GB+ RAM
3. **CPU核心**: 并行训练时建议8核+ CPU

### 软件优化

1. **批处理大小**: 根据GPU内存调整batch_size
2. **学习率**: 使用学习率调度器优化收敛
3. **缓存策略**: 启用数据缓存加速训练

### 配置优化

```yaml
# 性能优化配置示例
training:
  batch_size: 128              # 根据GPU内存调整
  learning_rate: 3e-4          # 推荐学习率
  gradient_clip: 0.5           # 梯度裁剪

data:
  cache_enabled: true          # 启用缓存
  num_workers: 4               # 数据加载进程数

debug:
  training_output:
    only_show_errors: true     # 减少输出提高性能
```

## ⚠️ 常见问题解决

### 安装问题

**问题**: PyTorch Geometric安装失败
```bash
# 解决方案：使用conda安装
conda install pytorch-geometric -c pyg
```

**问题**: METIS依赖缺失
```bash
# 解决方案：安装METIS
conda install -c conda-forge metis
# 或者
pip install metis-python
```

### 训练问题

**问题**: 内存不足
```yaml
# 解决方案：减少批处理大小
training:
  batch_size: 32  # 从64减少到32
```

**问题**: 训练不收敛
```yaml
# 解决方案：调整学习率和奖励权重
training:
  learning_rate: 1e-4  # 降低学习率

environment:
  reward_weights:
    balance_weight: 2.0  # 增加平衡权重
```

### 评估问题

**问题**: 基线方法运行失败
```bash
# 解决方案：检查scikit-learn版本
pip install scikit-learn>=1.0.0
```

## 📈 最佳实践

### 训练策略

1. **从小规模开始**: 先用IEEE14测试，再扩展到大规模网络
2. **参数调优**: 使用fast模式快速测试参数，用full模式获得最终结果
3. **多次运行**: 使用不同随机种子验证结果稳定性
4. **监控指标**: 关注训练过程中的奖励和分区质量指标

### 实验设计

1. **对照实验**: 与基线方法进行对比
2. **消融研究**: 分析不同组件的贡献
3. **参数敏感性**: 测试关键参数的影响
4. **可扩展性**: 在不同规模网络上测试

### 结果分析

1. **定量分析**: 关注负载平衡、电气解耦等关键指标
2. **定性分析**: 通过可视化分析分区的合理性
3. **统计显著性**: 使用多次运行的统计分析
4. **实际应用**: 考虑实际电力系统的约束和需求

## 🔗 相关文档

- **奖励系统**: 详见 `docs/REWARD.md`
- **基线方法**: 详见 `docs/BASELINE_METHODS.md`
- **强化学习**: 详见 `docs/REINFORCEMENT_LEARNING.md`
- **系统架构**: 详见 `docs/SYSTEM_ARCHITECTURE.md`

## 📞 技术支持

如遇问题，请按以下步骤排查：

1. **检查依赖**: `python train.py --check-deps`
2. **查看日志**: 检查 `logs/` 目录中的错误信息
3. **配置验证**: 确认 `config.yaml` 配置正确
4. **清理缓存**: `python code/clean.py --all` 清理可能损坏的缓存
5. **重新安装**: 重新安装依赖包解决版本冲突

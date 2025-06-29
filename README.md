# 🔋 电力网络分区强化学习系统

基于图注意力网络(GAT)和强化学习(PPO)的电力网络智能分区系统，支持增强奖励机制和场景生成。

## 🚀 快速开始

### 安装依赖

```bash
pip install torch torch-geometric numpy matplotlib pyyaml tqdm
pip install pandapower networkx plotly seaborn  # 可选依赖
```

### 立即开始训练

```bash
# 增强奖励训练（推荐）
python train.py --config enhanced_rewards_training

# 完整训练（2-4小时）
python train.py --mode full

# IEEE 118节点大规模训练
python train.py --mode ieee118
```

### 模型评估测试

```bash
# 完整评估流程（包含所有测试对比）
python test.py --mode full
```

## 📋 系统架构

### 🏗️ 双模块设计

- **🚀 `train.py`** - 训练模块：专注于模型训练、数据处理、基础可视化
- **🧪 `test.py`** - 评估模块：专注于A/B测试、奖励分析、性能评估

### 训练模式对比

| 模式 | 训练时间 | 回合数 | 适用场景 | 特色功能 |
|------|----------|--------|----------|----------|
| `enhanced_rewards_training` | 45分钟 | 1500 | **推荐模式**、高质量训练 | 三阶段增强奖励 |
| `full` | 2-4小时 | 5000 | 高质量结果、论文实验 | 完整训练流程 |
| `ieee118` | 4-8小时 | 3000 | 大规模系统测试 | 并行+场景生成 |

### 评估模式

| 模式 | 评估时间 | 功能 | 适用场景 | 输出结果 |
|------|----------|------|----------|----------|
| `full` | 30-60分钟 | **完整评估流程**（包含A/B测试、奖励分析、性能分析） | 论文实验、系统验证 | 综合HTML报告 |

## 🎯 增强奖励系统

### 三阶段训练流程

```bash
# 完整三阶段训练（推荐）
python train.py --config enhanced_rewards_training
```

### 场景生成功能

**所有训练模式默认启用场景生成**，包括：
- 🔧 **N-1故障**：随机断开线路模拟设备故障
- 🔧 **负荷扰动**：随机调整负荷大小模拟需求变化
- 🎭 **组合场景**：同时应用多种扰动类型
- 💾 **智能缓存**：自动缓存不同场景的图数据

## 🔧 参数配置

```bash
# 指定电力系统算例
python train.py --config enhanced_rewards_training --case ieee30 --partitions 5

# 调整训练参数
python train.py --mode full --episodes 3000 --lr 0.001

# 保存训练结果
python train.py --config enhanced_rewards_training --save-results

# 完整评估（包含所有测试对比）
python test.py --mode full --output-dir my_evaluation
```

## 🔄 完整工作流程

### 🎯 推荐的一站式命令

```bash
# 🚀 训练：使用增强奖励系统进行高质量训练
python train.py --config enhanced_rewards_training --save-results

# 🧪 评估：完整评估流程（包含A/B测试、奖励分析、性能分析）
python test.py --mode full --output-dir evaluation_results

# 📊 结果：查看生成的HTML报告
# 报告位置：evaluation_results/evaluation_report_YYYYMMDD_HHMMSS.html
```

### 💡 使用建议

1. **日常研究**：使用 `enhanced_rewards_training` 获得高质量结果
2. **完整评估**：使用 `test.py --mode full` 包含所有测试对比功能
3. **大规模验证**：使用 `ieee118` 模式测试大规模性能

详细使用说明请参考 [USAGE.md](USAGE.md)

## 📁 项目结构

```
├── train.py                  # 训练模块（专注模型训练功能）
├── test.py                   # 评估模块（专注模型评估分析）
├── config.yaml              # 统一配置文件（包含增强奖励配置）
├── code/                     # 核心代码
│   ├── src/                  # 源代码
│   │   ├── data_processing.py    # 数据处理
│   │   ├── gat.py                # GAT编码器
│   │   └── rl/                   # 强化学习模块
│   │       ├── environment.py    # 环境定义
│   │       ├── agent.py          # PPO智能体
│   │       ├── reward.py         # 增强奖励系统
│   │       ├── ab_testing.py     # A/B测试框架
│   │       └── reward_analyzer.py # 奖励分析器
│   └── baseline/             # 基线方法
├── cache/                    # 数据缓存
├── logs/                     # 训练日志
├── checkpoints/              # 模型检查点
├── figures/                  # 可视化图表
└── test_results/             # 测试结果
```

## 🎉 核心特性

- **🧠 智能分区**：基于GAT的图神经网络编码器
- **🎯 增强奖励**：三阶段渐进式奖励设计
- **🎭 场景生成**：自动生成多样化训练场景
- **📊 A/B测试**：完整的基线方法对比框架
- **📈 深度分析**：奖励函数和性能的全面分析
- **🔧 易于使用**：简洁的命令行接口
- **📋 完整评估**：从训练到评估的全流程支持
- **🎨 美化输出**：使用 Rich 库提供彩色进度条和优雅的终端显示

## 📞 技术支持

如遇问题，请检查：
1. 依赖安装是否完整：`python train.py --check-deps`
2. 配置文件是否正确：`config.yaml`
3. 数据缓存是否正常：`cache/` 目录

## 🎨 终端输出美化

### 输出控制功能

系统使用 **Rich** 库提供美化的终端输出，支持：
- 🌈 **彩色进度条**：实时显示训练进度和关键指标
- 📊 **格式化表格**：优雅展示训练摘要和评估结果
- ⚡ **智能过滤**：只显示重要信息，减少终端冗余输出

### 快速配置

**🔥 推荐设置**（只显示报错和进度条）：
```yaml
# config.yaml
debug:
  training_output:
    only_show_errors: true    # 简洁模式，只显示关键信息
    use_rich_output: true     # 启用彩色美化输出
```

**🔍 调试设置**（显示详细信息）：
```yaml
# config.yaml  
debug:
  training_output:
    only_show_errors: false           # 显示所有信息
    show_attention_collection: true   # 显示GAT注意力收集过程
    show_scenario_generation: true    # 显示场景生成详情
    show_cache_loading: true          # 显示数据缓存过程
```

### 使用建议

| 场景 | 设置 | 效果 |
|------|------|------|
| **日常训练** | `only_show_errors: true` | 干净整洁，只显示进度条和错误 |
| **问题调试** | `only_show_errors: false` | 显示详细调试信息 |
| **性能监控** | 保持默认设置 | 平衡信息量和可读性 |

---

**开始您的电力网络分区研究之旅！** 🚀

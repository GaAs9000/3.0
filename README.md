# 🔋 电力网络分区强化学习系统

基于图注意力网络(GAT)和强化学习(PPO)的电力网络智能分区系统，采用基于势函数理论的奖励机制和场景生成。

## 🚀 快速开始

### 安装依赖

```bash
pip install torch torch-geometric numpy matplotlib pyyaml tqdm
pip install pandapower networkx plotly seaborn  # 可选依赖
```

### 立即开始训练

```bash
# 快速训练（推荐）
python train.py --mode fast

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
- **🧪 `test.py`** - 评估模块：专注于模型性能评估、基线对比、可视化分析

### 训练模式对比

| 模式 | 训练时间 | 回合数 | 适用场景 | 特色功能 |
|------|----------|--------|----------|----------|
| `fast` | 45分钟 | 1500 | **推荐模式**、日常训练 | 场景生成 |
| `adaptive` | 1-2小时 | 2000 | **智能训练**、自适应优化 | 🧠智能导演+场景生成 |
| `full` | 2-4小时 | 5000 | 高质量结果、论文实验 | 并行+场景生成 |
| `ieee118` | 4-8小时 | 3000 | 大规模系统训练 | 并行+场景生成 |



## 🎯 奖励系统

基于势函数理论的双层激励结构，提供理论最优的训练效果：

**核心特性：**
- 🔬 **理论基础**：严格遵循势函数奖励塑造理论
- ⚡ **即时奖励**：基于增量改善的密集反馈
- 🎯 **终局奖励**：基于最终质量的综合评估
- 🛡️ **数值稳定**：全面的NaN/inf保护机制
- 📊 **详细分解**：提供奖励组件的详细分析

详细说明请参考：[docs/REWARD.md](docs/REWARD.md)

## 🧠 智能自适应课程学习

**从"固定剧本"升级为"智能导演"模式**，实现真正的自适应训练：

**核心特性：**
- 🎯 **性能驱动转换**：基于多维度指标的智能阶段转换
- 📈 **平滑参数演化**：连续参数调整，避免训练震荡
- 🛡️ **安全保障机制**：训练崩溃检测与自动恢复
- 🔄 **向后兼容**：保持现有系统稳定性

**4阶段智能课程：**
1. **探索期**：高度放松约束，鼓励探索
2. **过渡期**：逐渐收紧约束，平滑演化
3. **精炼期**：严格约束，追求高质量
4. **微调期**：学习率退火，精细优化

```bash
# 启用智能自适应课程学习
python train.py --mode adaptive
```

详细说明请参考：[docs/ADAPTIVE_CURRICULUM.md](docs/ADAPTIVE_CURRICULUM.md)

### 场景生成功能

**所有训练模式默认启用场景生成**，包括：
- 🔧 **N-1故障**：随机断开线路模拟设备故障
- 🔧 **负荷扰动**：随机调整负荷大小模拟需求变化
- 🎭 **组合场景**：同时应用多种扰动类型
- 💾 **智能缓存**：自动缓存不同场景的图数据

## 🔧 参数配置

```bash
# 指定电力系统算例
python train.py --mode fast --case ieee30 --partitions 5

# 调整训练参数
python train.py --mode full --episodes 3000 --lr 0.001

# 保存训练结果
python train.py --mode fast --save-results

# 完整评估（包含所有测试对比）
python test.py --mode full --output-dir my_evaluation
```

## 🔄 完整工作流程

### 🎯 推荐的一站式命令

```bash
# 🚀 训练：快速训练获得高质量结果
python train.py --mode fast --save-results

# 📊 结果：查看训练日志和模型输出
# 结果位置：logs/ 和 checkpoints/ 目录
```

### 💡 使用建议

1. **日常训练**：使用 `fast` 模式进行快速训练
2. **高质量结果**：使用 `full` 模式获得最佳性能
3. **大规模训练**：使用 `ieee118` 模式训练大规模系统
4. **奖励系统详情**：参考 `docs/REWARD.md` 了解奖励系统原理



## 📁 项目结构

```
├── train.py                  # 训练模块（专注模型训练功能）
├── test.py                   # 评估模块（专注模型评估分析）
├── config.yaml              # 统一配置文件（包含奖励系统配置）
├── code/                     # 核心代码
│   ├── src/                  # 源代码
│   │   ├── data_processing.py    # 数据处理
│   │   ├── gat.py                # GAT编码器
│   │   └── rl/                   # 强化学习模块
│   │       ├── environment.py    # 环境定义
│   │       ├── agent.py          # PPO智能体
│   │       └── reward.py         # 奖励系统
│   └── baseline/             # 基线方法
├── docs/                     # 文档目录
│   └── REWARD.md             # 双层奖励系统详细说明
├── cache/                    # 数据缓存
├── logs/                     # 训练日志
├── checkpoints/              # 模型检查点
├── figures/                  # 可视化图表
└── test_results/             # 测试结果
```

## 🎉 核心特性

- **🧠 智能分区**：基于GAT的图神经网络编码器
- **🎯 奖励系统**：基于势函数理论的理论最优奖励系统
- **🛡️ 数值稳定**：全面的NaN/inf保护和异常处理
- **🎭 场景生成**：自动生成多样化训练场景
- **📊 基线对比**：完整的基线方法对比框架
- **📈 性能分析**：可扩展性、鲁棒性和收敛性分析
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

tensorboard --logdir=logs --port=6006
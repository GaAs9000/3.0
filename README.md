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
# 快速测试（5分钟）
python train.py --mode quick

# 标准训练（30分钟）
python train.py --mode standard

# 增强奖励训练（推荐）
python train.py --config enhanced_rewards_training

# 完整训练（2-4小时）
python train.py --mode full

# IEEE 118节点大规模训练
python train.py --mode ieee118
```

### 模型评估测试

```bash
# 快速A/B测试对比
python test.py --mode ab_test

# 奖励函数深度分析
python test.py --mode reward_analysis

# 性能分析评估
python test.py --mode performance

# 完整评估流程（推荐）
python test.py --mode full

# 指定模型路径评估
python test.py --mode full --model checkpoints/best_model.pth
```

## 📋 系统架构

### 🏗️ 双模块设计

- **🚀 `train.py`** - 训练模块：专注于模型训练、数据处理、基础可视化
- **🧪 `test.py`** - 评估模块：专注于A/B测试、奖励分析、性能评估

### 训练模式对比

| 模式 | 训练时间 | 回合数 | 适用场景 | 特色功能 |
|------|----------|--------|----------|----------|
| `quick` | 5分钟 | 100 | 功能验证、快速测试 | 轻量化配置 |
| `standard` | 30分钟 | 1000 | 日常研究、算法验证 | 平衡性能与时间 |
| `enhanced_rewards_training` | 45分钟 | 1500 | **推荐模式**、高质量训练 | 三阶段增强奖励 |
| `full` | 2-4小时 | 5000 | 高质量结果、论文实验 | 完整训练流程 |
| `ieee118` | 4-8小时 | 3000 | 大规模系统测试 | 并行+场景生成 |

### 评估模式对比

| 模式 | 评估时间 | 功能 | 适用场景 | 输出结果 |
|------|----------|------|----------|----------|
| `ab_test` | 10-20分钟 | A/B测试对比 | 方法对比、基线验证 | 对比图表、统计报告 |
| `reward_analysis` | 5-10分钟 | 奖励函数分析 | 奖励设计优化 | 组件分析、相关性图 |
| `performance` | 15-30分钟 | 性能分析 | 系统优化、扩展性测试 | 性能指标、可扩展性图 |
| `full` | 30-60分钟 | **完整评估流程** | 论文实验、系统验证 | 综合HTML报告 |

## 🎯 增强奖励系统

### 三阶段训练流程

```bash
# 第一阶段：稠密奖励（解决稀疏奖励问题）
python train.py --config stage1_dense_rewards

# 第二阶段：智能探索（平衡探索与利用）
python train.py --config stage2_smart_exploration

# 第三阶段：自适应物理约束（融入领域知识）
python train.py --config stage3_adaptive_physics

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

### 训练参数

```bash
# 指定电力系统算例
python train.py --mode standard --case ieee30 --partitions 5

# 调整训练参数
python train.py --mode standard --episodes 2000 --lr 0.001

# 启用并行训练
python train.py --mode parallel --workers 8

# 保存训练结果
python train.py --mode standard --save-results --output-dir my_experiment
```

### 评估参数

```bash
# 指定测试案例和回合数
python test.py --mode full --case ieee30 --episodes 100

# 指定模型路径进行评估
python test.py --mode ab_test --model checkpoints/ieee30_model.pth

# 自定义输出目录
python test.py --mode full --output-dir my_evaluation

# 生成简化报告（无可视化）
python test.py --mode performance --no-viz --no-report
```

## 🔄 完整工作流程

### 典型研究流程

```bash
# 1️⃣ 第一步：快速验证系统功能
python train.py --mode quick --case ieee14
python test.py --mode ab_test --case ieee14

# 2️⃣ 第二步：标准训练获得基线模型
python train.py --mode standard --case ieee30 --save-results

# 3️⃣ 第三步：增强奖励训练优化性能
python train.py --config enhanced_rewards_training --case ieee30 --save-results

# 4️⃣ 第四步：完整评估对比分析
python test.py --mode full --case ieee30 --model checkpoints/enhanced_model.pth

# 5️⃣ 第五步：大规模系统验证
python train.py --mode ieee118 --save-results
python test.py --mode performance --case ieee118
```

### 🎯 推荐的一站式命令

```bash
# 🚀 训练：使用增强奖励系统进行高质量训练
python train.py --config enhanced_rewards_training --save-results

# 🧪 评估：完整评估流程生成综合报告
python test.py --mode full --output-dir evaluation_results

# 📊 结果：查看生成的HTML报告
# 报告位置：evaluation_results/evaluation_report_YYYYMMDD_HHMMSS.html
```

### 💡 使用建议

1. **新手用户**：先运行 `quick` 模式熟悉系统
2. **研究人员**：使用 `enhanced_rewards_training` + `full` 评估
3. **工程应用**：使用 `ieee118` 模式测试大规模性能
4. **方法对比**：重点使用 `ab_test` 模式进行基线对比
5. **系统优化**：使用 `performance` 模式分析瓶颈

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

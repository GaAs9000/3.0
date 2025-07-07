# 🔋 电力网络分区强化学习系统 2.0

基于图注意力网络(GAT)和强化学习(PPO)的电力网络智能分区系统，采用智能自适应训练和场景感知奖励机制。

## ⚡ 最新架构升级

**🎉 架构级重构已完成！**
- ✅ **官方数据转换**: 迁移至pandapower官方转换器，确保数据质量和一致性
- ✅ **智能奖励系统**: 全面启用HybridRewardFunction，支持场景感知和历史基线
- ✅ **两级查询基线**: 精确场景匹配 → 类别平均 → 全局回退的智能机制
- ✅ **探索奖励加成**: 激励Agent探索新场景，解决冷启动问题
- ✅ **GPU计算优化**: 关键计算迁移到GPU，性能大幅提升
- ✅ **全面兼容性**: 支持IEEE14/30/57/118所有标准测试系统

## 🚀 快速开始

### 环境要求
```bash
# Python 3.8+，推荐使用conda环境
conda create -n rl python=3.8
conda activate rl

# 核心依赖（必需）
pip install torch torch-geometric numpy matplotlib pyyaml tqdm
pip install pandapower networkx seaborn rich

# GPU支持（推荐）
# 访问 https://pytorch.org/get-started/locally/ 选择合适的PyTorch版本
```

### 立即开始训练
```bash
# 查看配置（不启动训练）
python train.py --mode ieee118

# 开始训练
python train.py --mode ieee118 --run

# 其他训练选项
python train.py --mode fast --run      # 快速训练 - IEEE14网络
python train.py --mode full --run      # 完整训练 - IEEE30网络
python train.py --mode ieee57 --run    # 中等规模 - IEEE57网络
python train.py --mode ieee118 --run   # 大规模 - IEEE118网络
```

### 模型测试和评估
```bash
# 🔬 Agent性能测试（保留的专业测试脚本）
python test.py --quick

# 详细性能评估
python test.py --episodes 20
```

### 监控训练进度
```bash
# TensorBoard实时监控（推荐）
tensorboard --logdir=data/logs --port=6006

# 查看生成的训练图表
# 结果位置：data/figures/ 和 output/dashboards/
```

## 📊 训练模式与网络兼容性

| 模式 | 网络规模 | 节点数 | 分区数 | 训练时间 | 特点 |
|------|----------|--------|--------|----------|------|
| `fast` | IEEE14 | 14 | 3 | ~30分钟 | 快速验证，原型测试 |
| `full` | IEEE30 | 30 | 4 | ~1-2小时 | 中等规模，完整功能 |
| `ieee57` | IEEE57 | 57 | 6 | ~3-4小时 | 复杂网络，高级测试 |
| `ieee118` | IEEE118 | 118 | 8 | ~6-8小时 | 大规模系统，生产级 |

> ✅ **全面兼容性**: 所有IEEE标准测试系统均已验证兼容

## 🎯 核心技术特性

### 🧠 智能场景感知奖励系统
- **场景分类**: 自动识别N-1故障、负荷波动等场景类型
- **历史基线**: 为每个场景类别维护独立的性能历史
- **两级查询**: 精确匹配 → 类别平均 → 全局回退的智能基线查询
- **探索奖励**: 激励Agent探索新场景，有效解决冷启动问题

### 🔧 数据处理与性能优化
- **官方转换器**: 使用pandapower官方`to_mpc`转换器，确保数据质量
- **智能导纳提取**: 上下文感知的导纳验证和提取机制
- **GPU计算**: 关键计算路径迁移到GPU，性能大幅提升
- **预计算缓存**: 总导纳等不变量的预计算和缓存

### 🎭 场景生成与自适应
- **多样化场景**: N-1故障、负荷扰动、组合场景
- **自适应课程**: 4阶段智能课程学习
- **平台期检测**: 自动识别训练停滞并调整策略

### 📊 监控与可视化
- **TensorBoard集成**: 分组记录训练指标，清晰的可视化结构
- **Rich输出**: 彩色终端显示，实时训练进度
- **HTML仪表板**: 自动生成交互式训练报告

## 📁 项目结构

```
├── train.py              # 主训练脚本（完整CLI支持）
├── test.py               # Agent性能测试脚本
├── config.yaml           # 统一配置文件
├── code/src/             # 核心代码模块
│   ├── rl/               # 强化学习核心
│   │   ├── agent.py      # PPO智能体
│   │   ├── environment.py # 分区环境
│   │   ├── reward.py     # 智能奖励系统
│   │   ├── scenario_*.py # 场景感知组件
│   │   └── utils.py      # 工具函数
│   ├── gat.py            # 图注意力网络
│   ├── data_processing.py # 数据处理（官方转换器）
│   └── visualization.py  # 可视化系统
├── data/                 # 数据、日志、缓存
│   ├── logs/             # TensorBoard日志
│   ├── cache/            # 预处理缓存
│   └── figures/          # 训练图表
└── docs/                 # 技术文档
```

## 🔧 配置说明

系统采用分层配置设计，支持多种预设：

```yaml
# 场景感知奖励系统（默认启用）
scenario_aware_reward:
  enabled: true
  relative_reward:
    enabled: true
    base_reward_weight: 0.5    # 基础奖励权重
    improvement_bonus: 0.1     # 改进奖励加成

# 数据处理配置
data:
  normalize: true              # 启用特征归一化
  cache_dir: data/cache       # 缓存目录

# 智能自适应训练
adaptive_curriculum:
  enabled: true
  plateau_detection:
    enabled: true
    confidence_threshold: 0.75

# 监控与日志
logging:
  use_tensorboard: true
  generate_html_dashboard: true
  log_dir: data/logs
```

## 🚀 最佳实践

### 开发流程
1. **开始开发**: `python train.py --mode fast --run` （快速验证）
2. **中等测试**: `python train.py --mode ieee57 --run` （功能测试）
3. **生产部署**: `python train.py --mode ieee118 --run` （完整训练）

### 性能监控
```bash
# 实时监控训练进度
tensorboard --logdir=data/logs --port=6006

# 检查GPU使用情况
nvidia-smi

# 查看系统资源
htop
```

### 故障排除
```bash
# 检查配置合并结果
python train.py --mode ieee118

# 验证数据加载
python -c "from train import load_power_grid_data; print('✅ IEEE118:', len(load_power_grid_data('ieee118')['bus']), 'nodes')"

# 清理缓存（如遇问题）
rm -rf data/cache/
```

## 📈 性能基准

基于Intel i7 + RTX 3080的测试结果：

| 网络 | 训练速度 | 内存使用 | 收敛质量 |
|------|----------|----------|----------|
| IEEE14 | ~2.1 s/episode | 2.5GB | 优秀 |
| IEEE30 | ~4.3 s/episode | 3.2GB | 优秀 |
| IEEE57 | ~8.7 s/episode | 4.1GB | 良好 |
| IEEE118 | ~13.4 s/episode | 6.8GB | 良好 |

> 💡 **性能提示**: 启用GPU可获得3-5x加速比

## 🔬 技术细节

### 场景感知奖励机制
系统实现了三层智能基线查询：
1. **精确匹配**: 查找特定场景ID的历史记录
2. **类别匹配**: 使用同类场景的平均表现
3. **全局回退**: 使用所有场景的全局基线

### 数据质量保证
- 使用pandapower官方转换器，确保物理参数准确性
- 智能导纳验证，根据归一化状态自适应调整验证范围
- 全面的数值稳定性保护（NaN/inf检测）

## 📞 技术支持

### 常见问题
1. **设备错误**: 确保PyTorch和CUDA版本匹配
2. **内存不足**: 降低batch_size或使用更小的网络
3. **收敛问题**: 检查学习率和奖励权重配置

### 获取帮助
```bash
# 查看详细帮助
python train.py --help

# 检查依赖版本
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

---

**🚀 一键启动:**
```bash
python train.py --mode ieee118 --run
```

> 🎯 **目标**: 构建世界级的电力网络智能分区系统
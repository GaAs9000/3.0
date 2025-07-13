# 🔋 电力网络分区强化学习系统 2.0

基于双塔架构和图注意力网络(GAT)的电力网络智能分区系统，解决固定K值问题，提升拓扑泛化能力。

## ⚡ 最新架构升级 - 双塔架构

**🎉 双塔架构全面集成！**
- ✅ **动作嵌入机制**: 策略向量与分区嵌入匹配，解耦K值依赖
- ✅ **动态分区数支持**: 训练时K=3，推理时可以是K=8或任意值
- ✅ **拓扑泛化增强**: 多尺度数据生成+GNN预训练
- ✅ **连通性保证**: 增强环境确保所有动作保持分区连通性
- ✅ **必需的GNN预训练**: 自监督学习提升泛化能力
- ✅ **生产级实现**: 无临时代码，完整工程实现

### 核心技术突破
- **状态塔**: EnhancedActorNetwork输出固定维度策略向量
- **动作塔**: PartitionEncoder编码分区物理特征
- **相似度决策**: 策略向量与分区嵌入的余弦相似度
- **渐进式训练**: 从小规模到大规模的课程学习

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
# 查看配置（不启动训练，仅加载参考拓扑）
python train.py --show-config

# 开始跨拓扑训练（推荐）
python train.py --run

# 单拓扑复现（可选）
python train.py --single --case ieee57 --run

# 传统预设仍支持：
python train.py --mode fast --run      # 快速验证 (IEEE14 参考网)
python train.py --mode full --run      # 中等规模 (IEEE30 参考网)
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

### 🧠 双塔架构系统
- **策略向量生成**: 状态塔输出128维策略向量，表示"理想分区特征"
- **分区特征编码**: 动作塔将分区的6个物理特征编码为嵌入向量
- **动态K值支持**: 训练和推理时的分区数可以不同
- **连通性保证**: 所有建议的动作都保证分区连通性

### 🔧 多尺度数据生成
- **拓扑变形**: 自动添加/删除节点和边，生成新拓扑
- **负荷扰动**: 真实的负荷和发电变化模拟
- **渐进式课程**: 从小规模逐步过渡到大规模网络
- **物理验证**: 确保生成的电网满足物理约束

### 🎭 GNN预训练（必需）
- **平滑性损失**: 相邻节点特征相似性
- **对比学习**: 子图级别的表示学习
- **物理一致性**: 确保嵌入符合电力流约束
- **早停机制**: 避免过拟合

### 📊 监控与可视化
- **TensorBoard集成**: 实时监控训练进度
- **Rich输出**: 彩色终端显示
- **HTML仪表板**: 交互式训练报告
- **分区可视化**: 实时查看分区结果

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
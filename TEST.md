# 智能体综合评估系统使用文档

## 概述

本评估系统为电力网络分区强化学习智能体提供全面的性能评估，包含baseline方法对比、跨网络泛化测试和自动化报告生成。

## 核心功能

### 1. 评估指标
- **分区间平衡度** (Inter-region CV): 各分区总负载的变异系数
- **区域内平衡度** (Intra-region CV): 各分区内部节点负载的平均变异系数 ⭐ **新增核心指标**
- **电气解耦度** (Electrical Decoupling): 分区内部边占总边数的比例
- **综合质量分数**: 基于三个核心指标的加权综合分数

### 2. Baseline方法
- **随机分区** (Random): 随机分配节点到各分区
- **度中心性分区** (Degree-based): 基于节点度数进行分区
- **谱聚类分区** (Spectral): 使用图的拉普拉斯矩阵特征向量

### 3. 测试场景
- **正常场景** (Normal): 原始负载数据
- **高负载场景** (High Load): 所有负载×1.5，模拟高峰期
- **不平衡场景** (Unbalanced): 随机30%节点负载×2，其他×0.8
- **故障场景** (Fault): 移除度数最高的1条边，模拟N-1故障

## 快速开始

### 1. 查看可用模式
```bash
python test.py --list-modes
```

### 2. 快速Baseline对比测试
```bash
# 在IEEE14网络上进行快速对比测试
python test.py --baseline --network ieee14 --mode quick

# 在IEEE30网络上进行标准对比测试
python test.py --baseline --network ieee30 --mode standard

# 使用预训练模型进行测试（推荐）
python test.py --baseline --network ieee30 --mode quick --model data/checkpoints/models/agent_ieee57_adaptive_best.pth
```

### 3. 泛化能力测试
```bash
# 在IEEE14上训练，测试在其他网络的泛化能力
python test.py --generalization --train-network ieee14 --mode quick

# 更全面的泛化测试
python test.py --generalization --train-network ieee14 --mode standard

# 使用预训练模型进行泛化测试（推荐）
python test.py --generalization --train-network ieee14 --mode quick --model data/checkpoints/models/agent_ieee57_adaptive_best.pth
```

### 4. 完整评估
```bash
# 运行完整评估（包含baseline对比和泛化测试）
python test.py --comprehensive --mode standard

# 最全面的评估
python test.py --comprehensive --mode comprehensive

# 使用预训练模型进行完整评估（推荐）
python test.py --comprehensive --mode standard --model data/checkpoints/models/agent_ieee57_adaptive_best.pth
```

## 评估模式说明

### Quick模式 (快速评估)
- **Baseline对比**: 2个网络, 2种场景, 每个5次运行
- **泛化测试**: 1个测试网络, 每个10次运行
- **适用场景**: 快速验证、调试测试

### Standard模式 (标准评估)
- **Baseline对比**: 3个网络, 3种场景, 每个10次运行
- **泛化测试**: 2个测试网络, 每个20次运行
- **适用场景**: 常规性能评估

### Comprehensive模式 (全面评估)
- **Baseline对比**: 3个网络, 4种场景, 每个15次运行
- **泛化测试**: 3个测试网络, 每个30次运行
- **适用场景**: 论文发表、深度分析

## 模型管理

### 训练模型
训练完成后，模型会自动保存到：
```
data/checkpoints/models/
├── agent_ieee14_adaptive_best.pth  # IEEE14网络训练模型
├── agent_ieee30_adaptive_best.pth  # IEEE30网络训练模型
├── agent_ieee57_adaptive_best.pth  # IEEE57网络训练模型
└── ...
```

**说明**: 所有训练都使用智能自适应算法，`--mode` 参数只是控制训练强度(episodes数量)，不改变训练算法。

### 使用预训练模型
```bash
# 查看可用模型
ls data/checkpoints/models/

# 使用特定模型进行评估
python test.py --baseline --model data/checkpoints/models/agent_ieee57_adaptive_best.pth
```

**优势**：
- ⚡ **快速评估**：跳过训练时间，直接使用已训练好的模型
- 🔄 **可重复性**：使用相同模型确保结果一致性
- 📊 **对比分析**：可以测试不同训练方法的模型效果

## 命令行参数

### 基本参数
- `--config`: 配置文件路径 (可选)
- `--mode`: 评估模式 (quick/standard/comprehensive)
- `--model`: 预训练模型路径 (可选，如不提供则重新训练)

### 测试类型 (必选其一)
- `--baseline`: 运行Baseline对比测试
- `--generalization`: 运行泛化能力测试
- `--comprehensive`: 运行完整评估
- `--list-modes`: 列出可用的评估模式

### Baseline测试参数
- `--network`: 测试网络 (ieee14/ieee30/ieee57/ieee118)

### 泛化测试参数
- `--train-network`: 训练网络 (ieee14/ieee30/ieee57)

## 输出结果

### 1. 控制台输出
- 实时进度显示
- 即时结果反馈
- 性能改善统计
- 泛化能力评级

### 2. 可视化图表
保存在 `evaluation_results/` 目录：
- `comparison_visualization.png`: Baseline对比图表
- `generalization_visualization.png`: 泛化能力图表

### 3. 评估报告
- `evaluation_report.txt`: 详细的文本报告
- 包含数值对比表、改善幅度、总体结论

## 结果解读

### Baseline对比结果
```
🏆 性能改善总结:
平均性能提升: 15.4%
最小性能提升: 8.2%
最大性能提升: 23.1%

normal: 提升 18.4% - 🎉 优秀
high_load: 提升 12.7% - ✅ 良好
```

### 泛化能力结果
```
🌐 泛化能力测试结果:
测试网络    质量分数    性能下降    泛化评级
IEEE30      0.68       4.2%       🎉 优秀泛化
IEEE57      0.64       9.9%       🎉 优秀泛化
IEEE118     0.58       18.3%      ✅ 良好泛化
```

### 评级标准
- **性能提升**:
  - 🎉 优秀: >15%
  - ✅ 良好: 5-15%
  - ⚠️ 有限: <5%

- **泛化能力**:
  - 🎉 优秀: <10%性能下降
  - ✅ 良好: 10-20%性能下降
  - ⚠️ 不足: >20%性能下降

## 系统架构

```
code/test/
├── comprehensive_evaluator.py    # 核心评估器
├── evaluation_config.py          # 配置管理
test.py                           # 主入口脚本
evaluation_results/              # 输出目录
├── comparison_visualization.png
├── generalization_visualization.png
└── evaluation_report.txt
```

## 故障排除

### 常见问题

1. **训练时间过长**
   - 使用 `--mode quick` 减少测试次数
   - 选择较小的网络进行测试

2. **内存不足**
   - 减少并行运行数量
   - 使用CPU模式训练

3. **依赖缺失**
   ```bash
   pip install scikit-learn matplotlib numpy
   ```

## 扩展功能

### 添加新的Baseline方法
在 `BaselinePartitioner` 类中添加新的静态方法：
```python
@staticmethod
def new_method_partition(adjacency_matrix: np.ndarray, num_partitions: int) -> np.ndarray:
    # 实现新的分区方法
    pass
```

### 自定义评估指标
在 `MetricsCalculator` 类中添加新的指标计算方法：
```python
@staticmethod
def calculate_new_metric(partition: np.ndarray, additional_data) -> float:
    # 实现新的指标计算
    pass
```

### 修改权重配置
编辑 `evaluation_config.py` 中的权重设置：
```python
'metrics_weights': {
    'inter_region_balance': 0.3,
    'intra_region_balance': 0.3,  # 新增指标权重
    'electrical_decoupling': 0.4
}
```

## 技术支持

如遇问题，请检查：
1. Python环境和依赖包版本
2. 数据文件完整性
3. 配置文件格式
4. 系统内存和GPU状态

更多技术细节请参考源代码注释和配置文件说明。

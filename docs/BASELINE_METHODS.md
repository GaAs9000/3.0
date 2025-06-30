# 基线方法对比系统

本文档描述了用于电网分区的各种基线方法实现，用于与强化学习方法进行对比分析。

## 📁 系统结构

```
baseline/
├── __init__.py               # 包初始化文件
├── baseline.py               # 主函数文件（包含基础类和兼容接口）
├── spectral_clustering.py    # 谱聚类分区方法
├── kmeans_clustering.py      # K-means聚类分区方法
├── evaluator.py              # 分区方案评估器
├── comparison.py             # 方法对比功能
└── (集成到主系统)            # 已集成到主训练系统
```

## 📋 核心组件功能

### 核心文件

#### 1. `baseline.py` - 主函数文件
- **功能**: 包含基础类和兼容性接口
- **主要内容**:
  - `set_baseline_seed()`: 设置随机种子函数
  - `BasePartitioner`: 基础分区器抽象类
  - `BaselinePartitioner`: 兼容性包装器类（支持原有接口）

#### 2. `spectral_clustering.py` - 谱聚类方法
- **功能**: 基于图谱理论的分区方法
- **主要类**: `SpectralPartitioner`
- **特点**: 
  - 使用电网的导纳矩阵构建邻接矩阵
  - 自动处理NaN和无穷大值
  - 使用稳定的标签分配算法

#### 3. `kmeans_clustering.py` - K-means聚类方法
- **功能**: 基于节点特征的聚类分区
- **主要类**: `KMeansPartitioner`
- **特点**:
  - 使用节点的电气特征（有功功率、无功功率、电压等）
  - 支持多种距离度量
  - 包含内存管理和线程安全机制

#### 4. `evaluator.py` - 分区评估器
- **功能**: 统一的分区质量评估
- **主要类**: `PartitionEvaluator`
- **评估指标**:
  - 负载平衡度 (load_cv)
  - 电气耦合度 (total_coupling)
  - 连通性 (connectivity)
  - 模块度 (modularity)

#### 5. `comparison.py` - 方法对比功能
- **功能**: 多方法性能对比和分析
- **主要函数**:
  - `compare_methods()`: 执行多方法对比
  - `evaluate_partition_method()`: 单方法评估
  - `run_baseline_comparison()`: 完整对比流程

## 🚀 使用方法

### 方式1: 兼容性接口（推荐用于现有代码）

```python
from baseline import BaselinePartitioner, run_baseline_comparison

# 创建分区器
spectral = BaselinePartitioner('spectral', seed=42)
kmeans = BaselinePartitioner('kmeans', seed=42)

# 执行分区
spectral_result = spectral.partition(env)
kmeans_result = kmeans.partition(env)

# 完整对比（包含RL方法）
comparison_df = run_baseline_comparison(env, agent, seed=42)
```

### 方式2: 模块化接口（推荐用于新代码）

```python
from baseline import (
    SpectralPartitioner, 
    KMeansPartitioner,
    evaluate_partition_method,
    compare_methods
)

# 创建分区器实例
spectral = SpectralPartitioner(seed=42)
kmeans = KMeansPartitioner(seed=42)

# 执行分区
spectral_labels = spectral.partition(env)
kmeans_labels = kmeans.partition(env)

# 评估单个方法
spectral_metrics = evaluate_partition_method(env, spectral_labels)

# 对比多个方法
methods = {'Spectral': spectral_labels, 'K-means': kmeans_labels}
comparison_df = compare_methods(env, methods)
```

## 📊 评估指标

### 核心指标

1. **负载平衡度 (load_cv)**
   - 计算各分区负载的变异系数
   - 值越小表示负载分布越均匀

2. **电气耦合度 (total_coupling)**
   - 计算跨分区连接的电气耦合强度
   - 值越小表示分区间电气独立性越好

3. **连通性 (connectivity)**
   - 评估各分区内部的连通性
   - 值越大表示分区内部连接越紧密

4. **模块度 (modularity)**
   - 基于图论的社区结构评估
   - 值越大表示分区质量越好

### 综合评分

系统使用加权综合评分对方法进行排名：
```
overall_score = w1*load_cv + w2*total_coupling + w3*connectivity + w4*modularity
```

## 📈 对比分析结果示例

### 方法对比结果
```
📋 分区方法对比结果:
                    load_cv  total_coupling  connectivity  modularity
method                                                               
RL (PPO)             0.1234          0.5678        0.9876      0.7890
Spectral Clustering  0.2345          0.6789        0.8765      0.6789
K-means              0.3456          0.7890        0.7654      0.5678

🏆 综合评分排名:
                    overall_score
method                          
RL (PPO)                  0.1234
Spectral Clustering      -0.0123
K-means                  -0.1234
```

## 🔧 扩展新方法

要添加新的基线方法，请按以下步骤：

1. **创建新的分区器类**:
```python
# 在新文件中，例如 new_method.py
from .baseline import BasePartitioner

class NewMethodPartitioner(BasePartitioner):
    def partition(self, env):
        # 实现你的分区算法
        self._set_seed()  # 设置随机种子
        # ... 你的算法逻辑
        return labels  # 返回1-based的标签
```

2. **更新导入**:
在`__init__.py`和`baseline.py`中添加新方法的导入

3. **更新对比函数**:
在`comparison.py`中添加新方法到对比列表

## 📋 依赖要求

```python
# 必需依赖
numpy>=1.20.0
torch>=1.9.0
scikit-learn>=1.0.0
pandas>=1.3.0

# 可选依赖（用于特定功能）
matplotlib>=3.4.0  # 用于可视化
networkx>=2.6.0    # 用于图分析
```

## ⚠️ 注意事项

1. **随机种子**: 所有方法都支持设置随机种子以确保结果可重现
2. **内存管理**: K-means方法包含自动内存清理机制
3. **错误处理**: 所有方法都包含异常处理，失败时会抛出错误
4. **GPU支持**: 自动检测并使用GPU加速（如果可用）
5. **线程安全**: K-means方法设置了单线程环境变量避免冲突

## 🎯 最佳实践

1. **方法选择**: 根据电网规模和特性选择合适的基线方法
2. **参数调优**: 使用网格搜索或贝叶斯优化调整方法参数
3. **结果验证**: 使用多个随机种子验证结果的稳定性
4. **性能分析**: 关注计算时间和内存使用情况
5. **可视化分析**: 使用分区可视化验证结果的合理性

## 🔗 与主系统集成

基线方法已完全集成到主训练和评估系统中：

- **训练阶段**: 可通过配置文件启用基线方法对比
- **评估阶段**: `test.py`自动包含基线方法对比
- **可视化**: 支持基线方法结果的可视化展示
- **日志记录**: 基线方法结果自动记录到训练日志中

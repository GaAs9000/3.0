# Baseline Methods

本文件夹包含了用于电网分区的各种基线方法实现，用于与强化学习方法进行对比分析。

## 📁 文件夹结构

```
baseline/
├── README.md                  # 本说明文件
├── __init__.py               # 包初始化文件
├── baseline.py               # 主函数文件（包含基础类和兼容接口）
├── spectral_clustering.py    # 谱聚类分区方法
├── kmeans_clustering.py      # K-means聚类分区方法
├── evaluator.py              # 分区方案评估器
├── comparison.py             # 方法对比功能
└── example_usage.py          # 使用示例
```

## 📋 文件功能详解

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
- **功能**: 基于节点嵌入的K-means聚类
- **主要类**: `KMeansPartitioner`
- **特点**:
  - 使用MiniBatchKMeans提高稳定性
  - 自动内存管理和清理
  - 失败时会抛出错误，而不是使用低质量的回退方案

#### 4. `evaluator.py` - 评估器
- **功能**: 评估分区方案的性能指标
- **主要函数**: `evaluate_partition_method()`
- **评估指标**:
  - `load_cv`: 负载变异系数
  - `load_gini`: 负载基尼系数
  - `total_coupling`: 总耦合度
  - `inter_region_lines`: 跨区域线路数
  - `connectivity`: 连通性
  - `power_balance`: 功率平衡
  - `modularity`: 模块度

#### 5. `comparison.py` - 方法对比
- **功能**: 比较不同分区方法的性能
- **主要函数**:
  - `compare_methods()`: 执行多种方法对比
  - `run_baseline_comparison()`: 完整对比流程（包含RL方法）

### 辅助文件

#### 6. `__init__.py` - 包初始化
- **功能**: 定义包的公共接口
- **导出**: 所有主要类和函数

#### 7. `example_usage.py` - 使用示例
- **功能**: 展示三种不同的使用方式
- **示例类型**:
  - 原有接口使用（向后兼容）
  - 新模块化接口使用
  - 单独方法使用

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

# 创建分区器
spectral = SpectralPartitioner(seed=42)
kmeans = KMeansPartitioner(seed=42)

# 执行分区
spectral_result = spectral.partition(env)

# 评估单个方法
metrics = evaluate_partition_method(env, spectral_result)
print(f"谱聚类指标: {metrics}")

# 对比多种方法
comparison_df = compare_methods(env, agent, seed=42)
```

### 方式3: 直接导入特定模块

```python
from baseline.spectral_clustering import SpectralPartitioner
from baseline.evaluator import evaluate_partition_method

# 只使用谱聚类
partitioner = SpectralPartitioner(seed=42)
result = partitioner.partition(env)
metrics = evaluate_partition_method(env, result)
```

## 📊 输出示例

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


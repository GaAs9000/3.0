# 自适应质量导向奖励系统设计文档

## 概述

本文档描述了电力网络分区强化学习系统中的自适应质量导向奖励函数设计。该系统实现了基于势函数理论的自适应质量导向激励结构，具备跨网络适应性、平台期检测和智能早停机制。

## 🎯 核心设计理念

### 两层奖励结构 + 平台期检测
1. **主奖励层**：纯势函数，鼓励质量改善，零固定阈值
2. **效率奖励层**：仅在质量平台期激活，鼓励快速收敛
3. **检测机制**：基于相对改善率的自适应判断

### 完全相对化设计
- **零固定阈值依赖**：所有评估基于相对改善，自动适应不同网络规模
- **跨网络适应性**：同一套配置适用于IEEE14到IEEE118的所有网络
- **势函数奖励**：基于质量分数的相对变化，理论保证最优策略不变

### 核心特性
- **完全相对化**：自动适应不同网络的质量水平
- **平台期检测**：基于改善率、稳定性和历史表现的综合判断
- **效率激励**：仅在质量平台期激活，避免质量牺牲
- **数值稳定性**：全面的NaN/inf保护和异常处理

## 📊 核心算法设计

### 1. 综合质量分数计算

```
Q(s) = 1 - normalize(w₁·CV + w₂·coupling_ratio + w₃·power_imbalance)

其中:
- CV ∈ [0, ∞), 越小越好
- coupling_ratio ∈ [0, 1], 越小越好
- power_imbalance ∈ [0, ∞), 越小越好
- normalize(): 映射到 [0, 1], 使Q(s) ∈ [0, 1], 越大越好
```

### 2. 势函数主奖励

```
Φ(s) = Q(s)  // 直接使用质量分作为势函数

主奖励 = γ · Φ(s_{t+1}) - Φ(s_t)
      = γ · Q(s_{t+1}) - Q(s_t)
```

**关键优势**: 完全相对化，自动适应不同网络的质量水平

### 3. 平台期检测算法

```
ALGORITHM: QualityPlateauDetection
INPUT: recent_scores[window_size], all_history_scores
OUTPUT: plateau_detected, confidence

// 1. 改善率检测
slope = linear_regression_slope(recent_scores)
improvement_rate = |slope|

// 2. 稳定性检测
variance = var(recent_scores)
stability_score = 1 / (1 + variance)

// 3. 相对水平检测
current_score = recent_scores.last()
historical_percentile = percentile_rank(all_history_scores, current_score)

// 4. 综合判断
plateau_detected = (
    improvement_rate < min_improvement_threshold AND
    stability_score > stability_threshold AND
    historical_percentile > min_percentile_threshold
)

confidence = weighted_average(
    1 - improvement_rate/max_rate,     # 改善越慢，置信度越高
    stability_score,                   # 越稳定，置信度越高
    historical_percentile              # 表现越好，置信度越高
)
```

### 4. 效率奖励机制

```
IF plateau_detected AND confidence > confidence_threshold:
    efficiency_reward = λ · (max_steps - current_step) / max_steps
ELSE:
    efficiency_reward = 0

总奖励 = 主奖励 + efficiency_reward
```

## 🏗️ 系统架构

### RewardFunction类

```python
class RewardFunction:
    """自适应质量导向奖励函数系统"""

    def __init__(self, hetero_data, config, device):
        """初始化自适应质量导向奖励函数"""
        self.plateau_detector = self._create_plateau_detector()
        self.adaptive_quality_config = self._load_adaptive_quality_config()

    def compute_incremental_reward(self, partition, action):
        """计算自适应质量导向即时奖励"""
        # 返回 (总奖励, 平台期检测结果)

    def _compute_quality_score(self, partition):
        """计算统一质量分数"""

    def should_early_stop(self, partition):
        """判断是否应该早停"""
```

### 核心组件

1. **QualityPlateauDetector**: 平台期检测器，实现三层检测算法
2. **_compute_quality_score()**: 统一质量分数计算，支持跨网络适应性
3. **plateau_detector**: 平台期检测器实例，管理检测状态
4. **adaptive_quality_config**: 自适应质量配置，控制检测参数

## ⚙️ 配置参数设计

### 新增配置节

```yaml
adaptive_quality:
  enabled: true

  # 平台期检测参数
  plateau_detection:
    window_size: 15                    # 观察窗口大小
    min_improvement_rate: 0.005        # 最小改善率阈值
    stability_threshold: 0.8           # 稳定性要求
    min_percentile: 0.7                # 历史表现要求(前70%)
    confidence_threshold: 0.8          # 早停置信度要求

  # 效率奖励参数
  efficiency_reward:
    lambda: 0.5                        # 效率奖励权重
    early_stop_confidence: 0.85        # 早停置信度阈值

  # 质量分数权重(替代原有固定阈值)
  quality_weights:
    cv_weight: 0.4
    coupling_weight: 0.3
    power_weight: 0.3
```

### 移除的配置

```yaml
# 删除这些固定阈值配置
thresholds:
  excellent_cv: 0.1          # ❌ 删除
  good_cv: 0.2               # ❌ 删除
  excellent_coupling: 0.3    # ❌ 删除
  good_coupling: 0.5         # ❌ 删除
  excellent_power: 10.0      # ❌ 删除
  good_power: 50.0           # ❌ 删除
## 🚀 使用示例

### 1. 基本配置

```python
# 在config.yaml中启用自适应质量系统
adaptive_quality:
  enabled: true
  plateau_detection:
    window_size: 15
    min_improvement_rate: 0.005
    stability_threshold: 0.8
    min_percentile: 0.7
    confidence_threshold: 0.8
  efficiency_reward:
    lambda: 0.5
    early_stop_confidence: 0.85
  quality_weights:
    cv_weight: 0.4
    coupling_weight: 0.3
    power_weight: 0.3
```

### 2. 环境集成

```python
# 创建支持自适应质量的环境
env = PowerGridPartitioningEnv(
    hetero_data=hetero_data,
    node_embeddings=node_embeddings,
    num_partitions=3,
    reward_weights={},  # 统一使用自适应质量导向奖励系统
    max_steps=200,
    device=device,
    config=config  # 包含adaptive_quality配置
)

# 训练循环
obs, info = env.reset()
for step in range(max_steps):
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    # 检查早停
    if info.get('early_termination', False):
        print(f"早停触发于第{step}步，置信度={info['plateau_confidence']:.3f}")
        break
```

### 3. 奖励函数直接使用

```python
# 创建奖励函数
reward_function = RewardFunction(
    hetero_data,
    config={'adaptive_quality': adaptive_config},
    device=device
)

# 计算奖励
reward, plateau_result = reward_function.compute_incremental_reward(
    current_partition, action
)

# 检查平台期
if plateau_result and plateau_result.plateau_detected:
    print(f"检测到平台期，置信度: {plateau_result.confidence:.3f}")
```

## 📈 预期效果与验证

### 1. 训练效果改善

**指标监控**:
- `average_episode_length`: 应该随训练下降（更快收敛）
- `early_termination_rate`: 应该随训练上升（更多早停成功）
- `plateau_confidence_mean`: 检测器的平均置信度
- `quality_score_final`: 最终质量分数分布

### 2. 跨网络适应性验证

**测试方案**:
```
FOR each_network IN [IEEE14, IEEE30, IEEE57, IEEE118]:
    使用完全相同的配置参数训练

    记录：
    - 收敛的quality_score范围
    - 平台期检测的触发时机
    - 早停成功率

    验证：
    - 不同网络都能稳定训练
    - 质量改善曲线形状相似
    - 无需针对性调参
```

### 3. 消融实验

```
实验组A: 仅主奖励(势函数)
实验组B: 主奖励 + 固定阈值早停
实验组C: 主奖励 + 自适应平台期检测 (完整方案)

对比维度:
- 训练稳定性
- 最终质量
- 收敛速度
- 跨网络泛化性
```

## 🔧 实施优先级

### Phase 1: 核心算法 (必须)
1. ✅ 实现 `compute_quality_score()` 函数
2. ✅ 修改主奖励为势函数形式
3. ✅ 基础的平台期检测逻辑

### Phase 2: 集成与调优 (重要)
1. ✅ 环境层集成早停逻辑
2. ✅ 效率奖励机制
3. ✅ 配置文件重构

### Phase 3: 监控与验证 (建议)
1. ✅ 详细的日志和可视化
2. 🔄 跨网络测试
3. 🔄 超参数敏感性分析

## 💡 关键成功因素

1. **归一化函数设计**: 确保质量分数在不同网络上可比
2. **窗口大小调优**: 平衡检测敏感性和稳定性
3. **权重配置**: 主奖励vs效率奖励的平衡
4. **渐进式部署**: 先在单一网络验证，再推广到多网络

## 🧪 测试验证

运行测试脚本验证系统功能：

```bash
python test_adaptive_quality.py
```

测试内容：
- ✅ 平台期检测算法的正确性
- ✅ 质量分数计算的跨网络适应性
- ✅ 早停机制的触发条件
- ✅ 系统集成的完整性

## 📚 理论基础

### 势函数奖励塑造理论

基于Ng et al. (1999)的势函数奖励塑造理论：
```
R'(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s)
```

在我们的系统中：
- **Φ(s) = Q(s)**：势函数直接使用质量分数
- **R(s,a,s') = 0**：原始环境奖励为0
- **R'(s,a,s') = γQ(s') - Q(s)**：塑形后的奖励

**理论保证**：势函数奖励塑造不改变最优策略，只改变学习速度。

### 平台期检测的数学基础

#### 1. 改善率检测
使用线性回归分析质量分数序列的趋势：
```
slope = argmin_β Σ(Q_i - (α + β·i))²
improvement_rate = |slope|
```

#### 2. 稳定性检测
基于方差的稳定性度量：
```
stability_score = 1 / (1 + Var(Q_recent))
```

#### 3. 历史百分位数
评估当前表现在历史中的相对位置：
```
percentile = |{Q_hist ≤ Q_current}| / |Q_hist|
```

### 跨网络适应性的数学原理

通过归一化确保质量分数的可比性：
```
normalized_cv = CV / (1 + CV)           ∈ [0, 1)
normalized_coupling = coupling_ratio     ∈ [0, 1]
normalized_power = power_imb / (1 + power_imb) ∈ [0, 1)

Q(s) = 1 - (w₁·norm_cv + w₂·norm_coupling + w₃·norm_power) / (w₁ + w₂ + w₃)
```

这种设计确保：
- 不同网络规模的质量分数具有可比性
- 权重配置在所有网络上保持一致的语义
- 无需针对特定网络调整阈值参数

---

**总结**: 这套自适应质量导向奖励系统彻底移除了固定阈值依赖，通过势函数+平台期检测实现真正的自适应质量导向训练。同一套配置可以适用于任意规模的电网，既保证质量改善又鼓励训练效率。

# 势函数奖励系统

## 🎯 核心问题

电力网络分区的评价指标复杂且相互耦合，传统稀疏奖励难以有效指导训练。不同场景（正常运行vs故障状态）的难度差异导致训练偏向问题。

## 🔬 技术创新：自适应质量导向奖励

### 设计理念

1. **势函数理论**: 保持最优策略不变的密集奖励
2. **相对改进**: 解决跨场景训练偏向问题
3. **平台期检测**: 智能效率激励机制
4. **完全自适应**: 零固定阈值，自动适应不同网络规模

## 📊 数学框架

### 质量分数计算

**综合质量函数:**
```
Q(s) = w₁·R_balance(CV) + w₂·R_decoupling + w₃·R_power

其中:
R_balance = exp(-2.0 · CV)
R_decoupling = 0.5·σ(5(r_edge-0.5)) + 0.5·σ(5(r_admittance-0.5))
R_power = exp(-3.0 · I_normalized)

CV = std(partition_sizes) / mean(partition_sizes)
σ(x) = sigmoid函数
```

### 势函数主奖励

**相对改进奖励:**
```
R_potential = (Q(s_{t+1}) - Q(s_t)) / Q(s_t)

数值稳定性处理:
- 当 Q(s_t) > 0.01: 使用相对改进公式
- 当 Q(s_t) ≤ 0.01: 使用绝对改进 Q(s_{t+1}) - Q(s_t)
- 结果裁剪: R_potential ∈ [-2.0, 2.0]
```

### 完整奖励结构

**总奖励组成:**
```
R_total = R_potential + λ·η_efficiency(t)

其中:
R_potential: 势函数主奖励（始终激活）
η_efficiency: 效率奖励（仅在平台期激活）
λ: 效率奖励权重
```

**效率奖励设计:**
```
η_efficiency(t) = {
    0,                           if not in_plateau
    (max_steps - t) / max_steps, if in_plateau
}

目的: 在质量平台期鼓励快速收敛
```

## 🎯 跨场景训练偏向问题

### 问题分析

**场景难度差异:**
```
正常场景: 0.70→0.72 (改进0.02) → 宽松条件，容易实现
故障场景: 0.40→0.42 (改进0.02) → N-1约束，困难实现

传统绝对奖励缺陷:
- 算法偏向简单场景
- 忽略困难场景学习
- 关键场景泛化能力差
```

### 相对改进解决方案

**公平激励机制:**
```
故障场景: (0.42-0.40)/0.40 = 5%改进 → 奖励+0.05
正常场景: (0.714-0.68)/0.68 = 5%改进 → 奖励+0.05

结果: 相同相对努力获得相同奖励
```

## 🔧 实现细节

### 核心奖励计算

<augment_code_snippet path="code/src/rl/reward.py" mode="EXCERPT">
````python
def _compute_simple_relative_reward(self, prev_quality: float, curr_quality: float) -> float:
    """计算相对改进奖励"""
    if prev_quality > 0.01:
        # 使用相对改进
        relative_improvement = (curr_quality - prev_quality) / prev_quality
    else:
        # 使用绝对改进（避免除零）
        relative_improvement = curr_quality - prev_quality

    # 数值稳定性保护
    return np.clip(relative_improvement, -2.0, 2.0)
````
</augment_code_snippet>

### 质量分数计算

<augment_code_snippet path="code/src/rl/reward.py" mode="EXCERPT">
````python
def _compute_balance_reward(self, cv: float) -> float:
    """平衡性奖励: R_balance = exp(-2.0 * CV)"""
    cv = max(0.0, min(cv, 10.0))  # 防止极值
    exp_arg = np.clip(-2.0 * cv, -500, 0)  # 防止exp溢出
    return np.clip(np.exp(exp_arg), 0.0, 1.0)

def _compute_decoupling_reward(self, edge_ratio: float, coupling_ratio: float) -> float:
    """解耦奖励: 基于边界比例和耦合强度"""
    edge_score = self._sigmoid_transform(edge_ratio, center=0.5, steepness=5)
    coupling_score = self._sigmoid_transform(coupling_ratio, center=0.5, steepness=5)
    return 0.5 * edge_score + 0.5 * coupling_score
````
</augment_code_snippet>

## 📈 平台期检测算法

### 检测逻辑

```
ALGORITHM: QualityPlateauDetection
INPUT: recent_scores[window_size], all_history_scores
OUTPUT: plateau_detected, confidence

1. 改善率检测:
   slope = linear_regression_slope(recent_scores)
   improvement_rate = |slope|

2. 稳定性检测:
   variance = var(recent_scores)
   stability_score = 1 / (1 + variance)

3. 综合判断:
   plateau_detected = (
       improvement_rate < threshold AND
       stability_score > min_stability
   )
### 平台期检测实现

<augment_code_snippet path="code/src/rl/reward.py" mode="EXCERPT">
````python
def detect_plateau(self, current_score: float, scenario_context=None):
    """检测质量平台期"""
    # 更新历史记录
    self.quality_history.append(current_score)

    # 计算改善率
    recent_scores = self.quality_history[-self.window_size:]
    improvement_rate = self._calculate_improvement_rate(recent_scores)

    # 计算稳定性
    stability_score = self._calculate_stability(recent_scores)

    # 综合判断
    plateau_detected = (
        improvement_rate < self.min_improvement_threshold and
        stability_score > self.stability_threshold
    )

    return PlateauResult(
        is_plateau=plateau_detected,
        confidence=stability_score,
        improvement_rate=improvement_rate
    )
````
</augment_code_snippet>

## 🛠️ 数值稳定性处理

### 防止数值异常

```python
def _safe_divide(self, numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """安全除法，防止除零和数值溢出"""
    if abs(denominator) < 1e-8:
        return fallback
    result = numerator / denominator
    return np.clip(result, -1e6, 1e6)

def _safe_exp(self, x: float) -> float:
    """安全指数函数，防止溢出"""
    x = np.clip(x, -500, 500)
    return np.exp(x)
```

## 📊 理论保证

### 势函数理论基础

基于Ng等人的势函数理论：
- **最优策略保持不变**: π*(原环境) = π*(增强环境)
- **收敛性保证**: Q学习算法仍然收敛到最优Q函数
- **训练加速**: 提供密集的中间指导信号

### 相对奖励优势

1. **跨场景公平性**: 相同努力获得相同奖励
2. **自动适应性**: 无需针对不同网络调参
3. **训练稳定性**: 避免奖励尺度差异导致的不稳定
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

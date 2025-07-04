# TensorBoard 指标说明文档

## 📊 概述

本文档详细说明了电力系统分区优化训练过程中TensorBoard记录的所有指标，帮助理解训练进度和模型性能。

## 🎯 指标分类体系

**新版本指标分组** (2025-07-03 更新)

TensorBoard指标现已重新组织为清晰的分组结构，便于监控和分析：

### 1. 核心训练指标 (Core/)
监控训练进度的最重要指标，应重点关注。

#### Core/Episode_Reward
- **含义**: 每个回合的总奖励
- **计算**: 即时奖励累积 + 终局奖励
- **取值范围**: [-∞, +∞]，通常在 [-5, 5] 范围内
- **优化目标**: 越大越好
- **趋势解读**:
  - 上升趋势：模型学习效果良好
  - 震荡但整体上升：正常的探索学习过程
  - 持续下降：可能存在训练问题

#### Core/Episode_Length
- **含义**: 每个回合的步数长度
- **计算**: 从开始到终止的动作步数
- **取值范围**: [1, max_steps_per_episode]
- **优化目标**: 适中为好（过短可能未充分探索，过长可能效率低）
- **趋势解读**:
  - 逐渐缩短：模型学会更快找到好解
  - 保持稳定：模型收敛到稳定策略

#### Core/Success_Rate
- **含义**: 回合成功标志
- **取值**: 0（失败）或 1（成功）
- **成功条件**: 质量分数超过阈值且满足连通性要求
- **优化目标**: 越大越好

#### Core/Current_Step
- **含义**: 当前训练步数
- **用途**: 时间轴参考，用于分析训练进程

### 2. 奖励组件详情 (Reward/)
展示奖励函数的内部构成，用于深入分析奖励机制。

#### Reward/Balance_Component
- **含义**: 负载平衡奖励组件
- **计算公式**: `R_balance = exp(-2.0 × CV)`
- **数学定义**: CV为负载变异系数，CV = σ/μ
- **取值范围**: [0, 1]
- **优化目标**: 越大越好（接近1表示负载完全平衡）
- **物理意义**: 衡量各分区负载分布的均匀程度

#### Reward/Decoupling_Component
- **含义**: 电气解耦奖励组件
- **计算公式**: `R_decoupling = exp(-4.0 × coupling_ratio)`
- **数学定义**: coupling_ratio为跨分区连接比例
- **取值范围**: [0, 1]
- **优化目标**: 越大越好（接近1表示分区间耦合最小）
- **物理意义**: 衡量分区间电气连接的紧密程度

#### Reward/Power_Component
- **含义**: 功率平衡奖励组件
- **计算公式**: `R_power = exp(-3.0 × I_normalized)`
- **数学定义**: I_normalized为归一化功率不平衡度
- **取值范围**: [0, 1]
- **优化目标**: 越大越好（接近1表示功率完全平衡）
- **物理意义**: 衡量各分区功率供需平衡程度

#### Reward/Quality_Total
- **含义**: 综合质量奖励
- **计算**: 上述三个组件的加权和
- **公式**: `quality_reward = w1×balance + w2×decoupling + w3×power`
- **取值范围**: [0, 3]（理论最大值）
- **优化目标**: 越大越好
- **权重配置**: 可在配置文件中调整各组件权重

#### Reward/Final_Total
- **含义**: 回合结束时的终局奖励
- **计算**: 基于最终分区状态的综合评估
- **取值范围**: 通常在 [0, 3] 范围内
- **优化目标**: 越大越好
- **触发条件**: 仅在回合结束时计算

#### Reward/Threshold_Bonus
- **含义**: 阈值奖励加成
- **用途**: 当达到特定质量阈值时的额外奖励

#### Reward/Termination_Discount
- **含义**: 终止条件折扣因子
- **用途**: 根据终止原因调整最终奖励

### 3. 质量评估指标 (Quality/)

#### Quality/Unified_Quality_Score
- **含义**: 统一质量分数
- **计算**: 基于CV、耦合比、功率平衡的综合评分
- **公式**: `Q = (1-CV_norm) × (1-coupling_norm) × (1-power_norm)`
- **取值范围**: [0, 1]
- **优化目标**: 越大越好（1表示完美分区）
- **用途**: 跨场景比较的标准化指标

#### Quality/Plateau_Confidence
- **含义**: 平台期检测置信度
- **计算**: 基于质量分数变化趋势的统计分析
- **取值范围**: [0, 1]
- **意义**:
  - 接近1：高置信度认为已达到质量平台期
  - 接近0：质量仍在显著改善中
- **用途**: 自适应训练和早停机制的依据

### 4. 训练算法指标 (Training/)

#### Training/Actor_Loss
- **含义**: 策略网络（Actor）损失
- **优化目标**: 适度最小化（过小可能导致策略过于确定）
- **正常范围**: [-1.0, 1.0]

#### Training/Critic_Loss
- **含义**: 价值网络（Critic）损失
- **优化目标**: 最小化
- **正常范围**: [0.0, 10.0]

#### Training/Policy_Entropy
- **含义**: 策略熵，衡量探索程度
- **优化目标**: 平衡（太高过度探索，太低缺乏探索）
- **正常范围**: [0.0, 2.0]

### 5. 调试诊断信息 (Debug/)

#### Debug/Current_Reward_Mode
- **含义**: 当前奖励模式标识
- **取值**: 数值编码的字符串哈希
- **用途**: 区分不同的奖励计算策略

## 🔍 指标解读指南

### 训练健康度检查
1. **Episode/Reward** 整体上升 ✅
2. **Episode/quality_score** 逐步提高 ✅  
3. **Episode/success** 成功率增加 ✅
4. **Episode/plateau_confidence** 适度波动 ✅

### 常见问题诊断

#### 奖励不收敛
- 检查各奖励组件是否平衡
- 观察quality_score是否有改善趋势
- 确认success率是否过低

#### 训练过慢
- 观察Episode/Length是否过长
- 检查plateau_confidence是否过早触发
- 分析各奖励组件的贡献度

#### 性能退化
- 对比不同时期的奖励组件分布
- 检查是否出现灾难性遗忘
- 观察quality_score的稳定性

## 📈 最佳实践

### 监控重点（按优先级）

#### 🔥 一级监控（必须关注）
1. **Core/Episode_Reward**: 训练进度的主要指标
2. **Core/Success_Rate**: 模型性能的直接体现
3. **Quality/Unified_Quality_Score**: 分区质量的综合评估

#### 📊 二级监控（深入分析）
1. **Reward/组件**: 了解奖励来源和平衡性
2. **Core/Episode_Length**: 训练效率指标
3. **Training/Policy_Entropy**: 探索与利用平衡

#### 🔧 三级监控（问题诊断）
1. **Quality/Plateau_Confidence**: 训练停滞检测
2. **Training/Actor_Loss & Critic_Loss**: 算法收敛性
3. **Debug/Current_Reward_Mode**: 配置验证

### TensorBoard 使用技巧

#### 分组查看策略
```bash
# 启动TensorBoard
tensorboard --logdir=data/logs --port=6006

# 推荐的查看顺序：
# 1. 先看Core/组，了解整体训练状态
# 2. 再看Quality/组，分析质量改善趋势
# 3. 最后看Reward/组，深入分析奖励机制
```

#### 自定义仪表板
在TensorBoard中创建自定义标量组合：
1. **训练概览**: Core/Episode_Reward + Core/Success_Rate + Quality/Unified_Quality_Score
2. **奖励分析**: 所有Reward/组件的对比
3. **算法诊断**: Training/组的所有指标
4. **质量趋势**: Quality/组 + Core/Episode_Length

#### 平滑参数建议
- **Core/指标**: 平滑度 0.6（观察长期趋势）
- **Reward/指标**: 平滑度 0.3（保留细节变化）
- **Training/指标**: 平滑度 0.8（过滤噪声）

### 异常检测与诊断

#### 常见异常模式
1. **Core/Episode_Reward 持续下降**
   - 检查 Training/Policy_Entropy 是否过低（缺乏探索）
   - 观察 Reward/组件是否平衡

2. **Core/Success_Rate 长期为0**
   - 检查 Quality/Unified_Quality_Score 是否有改善
   - 分析 Reward/组件的数值范围

3. **Quality/Plateau_Confidence 过早达到1.0**
   - 可能需要调整平台期检测参数
   - 检查是否陷入局部最优

#### 性能优化指导
- **Reward/Balance_Component 过低**: 调整负载平衡权重
- **Reward/Decoupling_Component 过低**: 增加解耦奖励系数
- **Training/Policy_Entropy 过高**: 降低探索率或调整熵正则化

## 🧮 数学公式详解

### 负载平衡指标
```
CV (变异系数) = σ/μ
其中：
- σ = sqrt(Σ(Pi - P_avg)²/n)  # 标准差
- μ = P_avg = Σ(Pi)/n         # 平均负载
- Pi = 分区i的总负载
- n = 分区数量

balance_reward = exp(-2.0 × CV)
```

### 电气解耦指标
```
coupling_ratio = 跨分区边数 / 总边数
edge_decoupling_ratio = 1 - coupling_ratio

decoupling_reward = exp(-4.0 × coupling_ratio)
```

### 功率平衡指标
```
power_imbalance = |Σ(Pgen_i - Pload_i)|  # 各分区功率不平衡绝对值之和
I_normalized = power_imbalance / total_power  # 归一化

power_reward = exp(-3.0 × I_normalized)
```

### 综合质量分数
```
quality_score = (1 - CV_norm) × (1 - coupling_norm) × (1 - power_norm)
其中：
- CV_norm = min(CV, 1.0)           # CV归一化到[0,1]
- coupling_norm = coupling_ratio    # 已在[0,1]范围
- power_norm = min(I_normalized, 1.0)  # 功率不平衡归一化
```

## 🎛️ 配置参数影响

### 奖励权重配置
```yaml
reward:
  weights:
    final_balance_weight: 1.0    # 影响balance_reward的重要性
    final_decoupling_weight: 1.0 # 影响decoupling_reward的重要性
    final_power_weight: 1.0      # 影响power_reward的重要性
```

### 自适应质量配置
```yaml
adaptive_quality:
  efficiency_reward:
    lambda: 0.1                  # 效率奖励系数
    early_stop_confidence: 0.8   # 早停置信度阈值
```

## 📊 指标关系图

### 新版分组结构
```
Core/ (核心训练指标)
├── Episode_Reward (总奖励)
├── Episode_Length (回合长度)
├── Success_Rate (成功率)
└── Current_Step (当前步数)

Reward/ (奖励组件详情)
├── Balance_Component = exp(-2.0 × CV)
├── Decoupling_Component = exp(-4.0 × coupling_ratio)
├── Power_Component = exp(-3.0 × I_normalized)
├── Quality_Total (加权和)
├── Final_Total (终局奖励)
├── Threshold_Bonus (阈值加成)
└── Termination_Discount (终止折扣)

Quality/ (质量评估)
├── Unified_Quality_Score = (1-CV_norm) × (1-coupling_norm) × (1-power_norm)
└── Plateau_Confidence (平台期置信度)

Training/ (训练算法)
├── Actor_Loss (策略网络损失)
├── Critic_Loss (价值网络损失)
└── Policy_Entropy (策略熵)

Debug/ (调试信息)
└── Current_Reward_Mode (奖励模式)
```

### 指标依赖关系
```
Core/Episode_Reward ← 汇总所有奖励
├── 即时奖励 (每步)
│   ├── 相对改进奖励 = (Q_curr - Q_prev) / Q_prev
│   └── 效率奖励 (平台期激活)
└── 终局奖励 (回合结束)
    └── Reward/Final_Total
        ├── Reward/Balance_Component
        ├── Reward/Decoupling_Component
        ├── Reward/Power_Component
        └── Reward/Quality_Total

Quality/Unified_Quality_Score ← 统一质量评估
├── CV (负载平衡) → Reward/Balance_Component
├── coupling_ratio (电气解耦) → Reward/Decoupling_Component
└── power_imbalance (功率平衡) → Reward/Power_Component
```

## 🚨 异常值处理

### 数值稳定性保护
- 所有奖励组件都有NaN/Inf检查
- 奖励值被限制在合理范围内：`clip(reward, -2.0, 2.0)`
- 指数函数参数被限制防止溢出：`clip(exp_arg, -500, 0)`

### 异常情况处理
1. **CV计算异常**: 当负载为0时，设置CV=1.0（最差情况）
2. **功率不平衡异常**: 当总功率为0时，设置归一化值=1.0
3. **连通性检查失败**: 返回最小奖励值

## 🔧 调试技巧

### 使用TensorBoard Profiler
```bash
# 启动TensorBoard时启用profiler
tensorboard --logdir=data/logs --port=6006 --load_fast=false
```

### 自定义标量组合
在TensorBoard中创建自定义标量组合：
1. 奖励总览：Episode/Reward + Episode/quality_score
2. 奖励分解：所有reward组件
3. 训练效率：Episode/Length + Episode/success

### 导出数据分析
```python
# 从TensorBoard日志中提取数据进行深度分析
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator('data/logs/training_20250703_123456')
ea.Reload()

# 获取特定指标数据
reward_data = ea.Scalars('Episode/Reward')
quality_data = ea.Scalars('Episode/quality_score')
```

---

*本文档随系统更新而维护，如有疑问请参考源代码或联系开发团队。*
